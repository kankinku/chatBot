"""
Delta Fetcher
신규/수정 데이터만 수집하는 Fetcher

책임:
- State 기반 Delta 수집
- 재시도 및 백오프
- 부분 실패 리포트
"""
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from src.shared.schemas import (
    SourceSpec, 
    DeltaMethod, 
    Observation, 
    Event,
    FetchState,
)
from src.ingestion.source_registry import SourceRegistry
from src.ingestion.fetch_state_store import FetchStateStore

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Fetch 결과"""
    success: bool
    source_id: str
    stream: str = "default"
    
    # 수집된 데이터
    observations: List[Observation] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    
    # 다음 수집을 위한 상태
    new_timestamp: Optional[datetime] = None
    new_cursor: Optional[str] = None
    new_etag: Optional[str] = None
    new_hash: Optional[str] = None
    
    # 에러 정보
    error_message: Optional[str] = None
    partial_failure: bool = False  # 일부만 실패
    retry_count: int = 0


class BaseFetcher(ABC):
    """추상 Fetcher 베이스 클래스"""
    
    @abstractmethod
    def fetch(
        self,
        source: SourceSpec,
        state: Optional[FetchState],
    ) -> FetchResult:
        """데이터 수집"""
        ...


class DeltaFetcher:
    """
    Delta Fetcher
    
    State 기반으로 신규/수정 데이터만 수집합니다.
    재시도 및 백오프를 지원합니다.
    """
    
    def __init__(
        self,
        source_registry: SourceRegistry,
        state_store: FetchStateStore,
        fetchers: Optional[Dict[str, BaseFetcher]] = None,
        max_retries: int = 3,
        base_backoff_seconds: float = 1.0,
    ):
        """
        Args:
            source_registry: 소스 레지스트리
            state_store: 상태 저장소
            fetchers: 소스 타입별 Fetcher 구현체
            max_retries: 최대 재시도 횟수
            base_backoff_seconds: 기본 백오프 시간 (초)
        """
        self.source_registry = source_registry
        self.state_store = state_store
        self.fetchers = fetchers or {}
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
    
    def register_fetcher(self, source_type: str, fetcher: BaseFetcher) -> None:
        """소스 타입별 Fetcher 등록"""
        self.fetchers[source_type] = fetcher
        logger.debug(f"Registered fetcher for type: {source_type}")
    
    def fetch_source(
        self,
        source_id: str,
        stream: str = "default",
        force_full: bool = False,
    ) -> FetchResult:
        """
        단일 소스 수집
        
        Args:
            source_id: 소스 ID
            stream: 스트림 ID
            force_full: True면 전체 수집 (State 무시)
        
        Returns:
            FetchResult
        """
        source = self.source_registry.get(source_id)
        if not source:
            return FetchResult(
                success=False,
                source_id=source_id,
                stream=stream,
                error_message=f"Source not found: {source_id}",
            )
        
        if not source.is_active:
            return FetchResult(
                success=False,
                source_id=source_id,
                stream=stream,
                error_message=f"Source is not active: {source_id}",
            )
        
        fetcher = self.fetchers.get(source.source_type.value)
        if not fetcher:
            return FetchResult(
                success=False,
                source_id=source_id,
                stream=stream,
                error_message=f"No fetcher for source type: {source.source_type.value}",
            )
        
        # State 조회
        state = None if force_full else self.state_store.get(source_id, stream)
        
        # 재시도 루프
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = fetcher.fetch(source, state)
                result.retry_count = attempt
                
                if result.success:
                    # 성공 시 State 업데이트
                    self.state_store.update_success(
                        source_id=source_id,
                        stream=stream,
                        fetched_count=len(result.observations) + len(result.events),
                        last_timestamp=result.new_timestamp,
                        last_cursor=result.new_cursor,
                        last_etag=result.new_etag,
                        last_hash=result.new_hash,
                    )
                    logger.info(
                        f"Fetched {source_id}/{stream}: "
                        f"{len(result.observations)} observations, "
                        f"{len(result.events)} events"
                    )
                    return result
                else:
                    last_error = result.error_message
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Fetch attempt {attempt + 1} failed: {e}")
            
            # 백오프
            if attempt < self.max_retries:
                backoff = self.base_backoff_seconds * (2 ** attempt)
                logger.debug(f"Backing off for {backoff:.1f}s")
                time.sleep(backoff)
        
        # 모든 재시도 실패
        self.state_store.update_failure(source_id, stream)
        
        return FetchResult(
            success=False,
            source_id=source_id,
            stream=stream,
            error_message=f"All retries failed: {last_error}",
            retry_count=self.max_retries,
        )
    
    def fetch_all(
        self,
        force_full: bool = False,
        stop_on_error: bool = False,
    ) -> Tuple[List[FetchResult], Dict[str, Any]]:
        """
        모든 활성 소스 수집
        
        Args:
            force_full: True면 전체 수집
            stop_on_error: True면 에러 시 중단
        
        Returns:
            (결과 리스트, 통계)
        """
        sources = self.source_registry.get_source_for_refresh()
        results = []
        
        total_observations = 0
        total_events = 0
        success_count = 0
        failure_count = 0
        
        for source in sources:
            result = self.fetch_source(source.source_id, force_full=force_full)
            results.append(result)
            
            if result.success:
                success_count += 1
                total_observations += len(result.observations)
                total_events += len(result.events)
            else:
                failure_count += 1
                if stop_on_error:
                    logger.warning(f"Stopping fetch due to error: {result.error_message}")
                    break
        
        stats = {
            "total_sources": len(sources),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_observations": total_observations,
            "total_events": total_events,
        }
        
        logger.info(
            f"Fetch all complete: {success_count}/{len(sources)} succeeded, "
            f"{total_observations} observations, {total_events} events"
        )
        
        return results, stats
    
    def fetch_series(
        self,
        series_ids: List[str],
        force_full: bool = False,
    ) -> Tuple[List[FetchResult], Dict[str, Any]]:
        """
        특정 시계열만 수집
        
        Args:
            series_ids: 시계열 ID 리스트
            force_full: True면 전체 수집
        
        Returns:
            (결과 리스트, 통계)
        """
        # 시계열 → 소스 매핑
        source_ids = set()
        for series_id in series_ids:
            source = self.source_registry.get_by_series(series_id)
            if source:
                source_ids.add(source.source_id)
            else:
                logger.warning(f"No source for series: {series_id}")
        
        results = []
        for source_id in source_ids:
            result = self.fetch_source(source_id, force_full=force_full)
            results.append(result)
        
        stats = {
            "requested_series": len(series_ids),
            "sources_fetched": len(source_ids),
            "success_count": sum(1 for r in results if r.success),
            "failure_count": sum(1 for r in results if not r.success),
        }
        
        return results, stats


# =============================================================================
# 샘플 Fetcher 구현 (API용)
# =============================================================================

class MockApiFetcher(BaseFetcher):
    """
    Mock API Fetcher (테스트 및 개발용)
    
    실제 구현 시 이 클래스를 참고하여 각 API별 Fetcher를 구현하세요.
    """
    
    def __init__(self, mock_data: Optional[Dict[str, List[Observation]]] = None):
        self.mock_data = mock_data or {}
    
    def fetch(
        self,
        source: SourceSpec,
        state: Optional[FetchState],
    ) -> FetchResult:
        """Mock 데이터 반환"""
        # 실제 구현에서는 source.endpoint로 API 호출
        observations = self.mock_data.get(source.source_id, [])
        
        # Delta 필터링 (since_timestamp 방식)
        if state and state.last_timestamp and source.delta_method == DeltaMethod.SINCE_TIMESTAMP:
            observations = [
                obs for obs in observations 
                if obs.timestamp > state.last_timestamp
            ]
        
        new_timestamp = None
        if observations:
            new_timestamp = max(obs.timestamp for obs in observations)
        
        return FetchResult(
            success=True,
            source_id=source.source_id,
            observations=observations,
            new_timestamp=new_timestamp,
        )
