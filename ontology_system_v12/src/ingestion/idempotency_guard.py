"""
Idempotency Guard
중복 데이터 방지 모듈

책임:
- 시계열 중복 체크 (series_id, timestamp, as_of)
- 이벤트 중복 체크 (source_id, external_id) 또는 (content_hash)
- 멱등성 보장
"""
import logging
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field

from src.shared.schemas import Observation, Event, generate_content_hash

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """중복 제거 결과"""
    original_count: int = 0
    unique_count: int = 0
    duplicate_count: int = 0
    observations: List[Observation] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    duplicate_keys: List[str] = field(default_factory=list)


class IdempotencyGuard:
    """
    멱등성 보장 모듈
    
    동일 데이터 중복 저장 방지.
    같은 배치를 2번 넣어도 결과 동일.
    """
    
    def __init__(self):
        """
        Note: 실제 운영에서는 외부 저장소(Redis, DB)에서 키를 관리해야 합니다.
              여기서는 메모리 기반 캐시를 사용합니다.
        """
        # 메모리 캐시 (실제로는 DB 조회로 대체)
        self._observation_keys: Set[Tuple[str, datetime, datetime]] = set()
        self._event_keys: Set[str] = set()
    
    def filter_observations(
        self,
        observations: List[Observation],
        check_external: bool = True,
    ) -> DeduplicationResult:
        """
        Observation 중복 필터링
        
        Key: (series_id, timestamp, as_of)
        
        Args:
            observations: 관측값 리스트
            check_external: 외부 저장소 확인 여부 (미구현)
        
        Returns:
            DeduplicationResult
        """
        result = DeduplicationResult(original_count=len(observations))
        seen_in_batch: Set[Tuple[str, datetime, datetime]] = set()
        
        for obs in observations:
            key = (obs.series_id, obs.timestamp, obs.as_of)
            key_str = f"{obs.series_id}|{obs.timestamp.isoformat()}|{obs.as_of.isoformat()}"
            
            # 배치 내 중복 체크
            if key in seen_in_batch:
                result.duplicate_count += 1
                result.duplicate_keys.append(key_str)
                continue
            
            # 기존 저장소 중복 체크
            if key in self._observation_keys:
                result.duplicate_count += 1
                result.duplicate_keys.append(key_str)
                continue
            
            # 고유 데이터
            seen_in_batch.add(key)
            result.observations.append(obs)
            result.unique_count += 1
        
        if result.duplicate_count > 0:
            logger.debug(f"Filtered {result.duplicate_count} duplicate observations")
        
        return result
    
    def filter_events(
        self,
        events: List[Event],
        check_external: bool = True,
    ) -> DeduplicationResult:
        """
        Event 중복 필터링
        
        Key: (source_id, external_id) 또는 content_hash
        
        Args:
            events: 이벤트 리스트
            check_external: 외부 저장소 확인 여부 (미구현)
        
        Returns:
            DeduplicationResult
        """
        result = DeduplicationResult(original_count=len(events))
        seen_in_batch: Set[str] = set()
        
        for event in events:
            # 키 생성
            if event.external_id:
                key = f"{event.source_id}|{event.external_id}"
            else:
                # content_hash 사용
                if not event.content_hash and event.content:
                    content_hash = generate_content_hash(event.content)
                else:
                    content_hash = event.content_hash or generate_content_hash(event.title)
                key = f"hash|{content_hash}"
            
            # 배치 내 중복 체크
            if key in seen_in_batch:
                result.duplicate_count += 1
                result.duplicate_keys.append(key)
                continue
            
            # 기존 저장소 중복 체크
            if key in self._event_keys:
                result.duplicate_count += 1
                result.duplicate_keys.append(key)
                continue
            
            # 고유 데이터
            seen_in_batch.add(key)
            result.events.append(event)
            result.unique_count += 1
        
        if result.duplicate_count > 0:
            logger.debug(f"Filtered {result.duplicate_count} duplicate events")
        
        return result
    
    def mark_as_stored(
        self,
        observations: Optional[List[Observation]] = None,
        events: Optional[List[Event]] = None,
    ) -> None:
        """
        저장 완료된 데이터 키 등록
        
        Args:
            observations: 저장된 관측값
            events: 저장된 이벤트
        """
        if observations:
            for obs in observations:
                key = (obs.series_id, obs.timestamp, obs.as_of)
                self._observation_keys.add(key)
        
        if events:
            for event in events:
                if event.external_id:
                    key = f"{event.source_id}|{event.external_id}"
                else:
                    content_hash = event.content_hash or generate_content_hash(event.title)
                    key = f"hash|{content_hash}"
                self._event_keys.add(key)
    
    def check_observation_exists(
        self,
        series_id: str,
        timestamp: datetime,
        as_of: datetime,
    ) -> bool:
        """단일 Observation 존재 여부 확인"""
        key = (series_id, timestamp, as_of)
        return key in self._observation_keys
    
    def check_event_exists(
        self,
        source_id: str,
        external_id: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> bool:
        """단일 Event 존재 여부 확인"""
        if external_id:
            key = f"{source_id}|{external_id}"
        elif content_hash:
            key = f"hash|{content_hash}"
        else:
            return False
        
        return key in self._event_keys
    
    def clear_cache(self) -> Tuple[int, int]:
        """
        캐시 클리어
        
        Returns:
            (삭제된 observation 키 수, 삭제된 event 키 수)
        """
        obs_count = len(self._observation_keys)
        evt_count = len(self._event_keys)
        
        self._observation_keys.clear()
        self._event_keys.clear()
        
        logger.info(f"Cleared idempotency cache: {obs_count} obs, {evt_count} events")
        return obs_count, evt_count
    
    def load_from_repository(
        self,
        observation_keys: Optional[List[Tuple[str, datetime, datetime]]] = None,
        event_keys: Optional[List[str]] = None,
    ) -> None:
        """
        저장소에서 기존 키 로드
        
        실제 구현에서는 TimeSeriesRepository, EventRepository에서 
        기존 키를 조회하여 로드합니다.
        """
        if observation_keys:
            self._observation_keys.update(observation_keys)
            logger.debug(f"Loaded {len(observation_keys)} observation keys")
        
        if event_keys:
            self._event_keys.update(event_keys)
            logger.debug(f"Loaded {len(event_keys)} event keys")
    
    def get_stats(self) -> Dict:
        """통계 조회"""
        return {
            "observation_keys_cached": len(self._observation_keys),
            "event_keys_cached": len(self._event_keys),
        }
