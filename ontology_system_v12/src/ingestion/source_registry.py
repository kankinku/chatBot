"""
Source Registry
데이터 소스별 Delta 수집 방식 정의 및 관리

책임:
- 소스 스펙 관리 (CRUD)
- Delta 방식 조회
- 소스별 제공 시계열 목록 조회
"""
import logging
from typing import Dict, List, Optional
from pathlib import Path
import yaml

from src.shared.schemas import SourceSpec, SourceType, DeltaMethod

logger = logging.getLogger(__name__)


class SourceRegistry:
    """
    데이터 소스 레지스트리
    
    소스별 Delta 수집 방식과 메타데이터를 관리합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: sources.yaml 경로 (None이면 기본 경로)
        """
        self._sources: Dict[str, SourceSpec] = {}
        self._series_to_source: Dict[str, str] = {}  # series_id -> source_id
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> int:
        """
        YAML 파일에서 소스 정의 로드
        
        Returns:
            로드된 소스 수
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Source config not found: {config_path}")
            return 0
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            sources = config.get('sources', [])
            loaded = 0
            
            for source_data in sources:
                spec = self._parse_source_spec(source_data)
                if spec:
                    self.register(spec)
                    loaded += 1
            
            logger.info(f"Loaded {loaded} sources from {config_path}")
            return loaded
            
        except Exception as e:
            logger.error(f"Failed to load source config: {e}")
            return 0
    
    def _parse_source_spec(self, data: Dict) -> Optional[SourceSpec]:
        """Dict에서 SourceSpec 파싱"""
        try:
            return SourceSpec(
                source_id=data['source_id'],
                name=data.get('name', data['source_id']),
                source_type=SourceType(data.get('source_type', 'api')),
                delta_method=DeltaMethod(data.get('delta_method', 'since_timestamp')),
                endpoint=data.get('endpoint'),
                credentials_key=data.get('credentials_key'),
                refresh_interval_minutes=data.get('refresh_interval_minutes', 60),
                retry_count=data.get('retry_count', 3),
                timeout_seconds=data.get('timeout_seconds', 30),
                provides_series=data.get('provides_series', []),
                is_active=data.get('is_active', True),
            )
        except Exception as e:
            logger.error(f"Failed to parse source spec: {e}")
            return None
    
    def register(self, spec: SourceSpec) -> None:
        """소스 등록"""
        self._sources[spec.source_id] = spec
        
        # series -> source 매핑 업데이트
        for series_id in spec.provides_series:
            self._series_to_source[series_id] = spec.source_id
        
        logger.debug(f"Registered source: {spec.source_id}")
    
    def unregister(self, source_id: str) -> bool:
        """소스 등록 해제"""
        if source_id not in self._sources:
            return False
        
        spec = self._sources[source_id]
        
        # series 매핑 제거
        for series_id in spec.provides_series:
            if self._series_to_source.get(series_id) == source_id:
                del self._series_to_source[series_id]
        
        del self._sources[source_id]
        logger.debug(f"Unregistered source: {source_id}")
        return True
    
    def get(self, source_id: str) -> Optional[SourceSpec]:
        """소스 조회"""
        return self._sources.get(source_id)
    
    def get_by_series(self, series_id: str) -> Optional[SourceSpec]:
        """시계열 ID로 소스 조회"""
        source_id = self._series_to_source.get(series_id)
        if source_id:
            return self._sources.get(source_id)
        return None
    
    def get_delta_method(self, source_id: str) -> Optional[DeltaMethod]:
        """소스의 Delta 수집 방식 조회"""
        spec = self._sources.get(source_id)
        return spec.delta_method if spec else None
    
    def list_sources(self, active_only: bool = True) -> List[SourceSpec]:
        """모든 소스 목록"""
        sources = list(self._sources.values())
        if active_only:
            sources = [s for s in sources if s.is_active]
        return sources
    
    def list_series(self, source_id: Optional[str] = None) -> List[str]:
        """시계열 목록 (소스별 또는 전체)"""
        if source_id:
            spec = self._sources.get(source_id)
            return spec.provides_series if spec else []
        return list(self._series_to_source.keys())
    
    def get_source_for_refresh(self) -> List[SourceSpec]:
        """리프레시가 필요한 소스 목록 (active 소스만)"""
        return [s for s in self._sources.values() if s.is_active]
    
    def get_stats(self) -> Dict:
        """레지스트리 통계"""
        active = sum(1 for s in self._sources.values() if s.is_active)
        return {
            "total_sources": len(self._sources),
            "active_sources": active,
            "total_series": len(self._series_to_source),
            "delta_methods": {
                method.value: sum(1 for s in self._sources.values() if s.delta_method == method)
                for method in DeltaMethod
            }
        }
