"""
Feature Spec Registry
Feature 정의(스펙) 관리

책임:
- Feature 스펙 등록/조회
- 스펙 유효성 검사
- Config 파일 로드
"""
import logging
from typing import Dict, List, Optional
from pathlib import Path
import yaml

from src.shared.schemas import FeatureSpec, FeatureType

logger = logging.getLogger(__name__)


class FeatureSpecRegistry:
    """
    Feature 정의 레지스트리
    
    스프레드, ROC, ZScore, YoY 등 다양한 Feature 정의를 관리합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: features.yaml 경로
        """
        self._specs: Dict[str, FeatureSpec] = {}
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self._register_default_specs()
    
    def load_from_file(self, config_path: str) -> int:
        """
        YAML 파일에서 스펙 로드
        
        Returns:
            로드된 스펙 수
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Feature config not found: {config_path}")
            self._register_default_specs()
            return 0
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            features = config.get('features', {})
            loaded = 0
            
            for feature_id, feature_data in features.items():
                spec = self._parse_spec(feature_id, feature_data)
                if spec:
                    self.register(spec)
                    loaded += 1
            
            logger.info(f"Loaded {loaded} feature specs from {config_path}")
            return loaded
            
        except Exception as e:
            logger.error(f"Failed to load feature config: {e}")
            self._register_default_specs()
            return 0
    
    def _parse_spec(self, feature_id: str, data: Dict) -> Optional[FeatureSpec]:
        """Dict에서 FeatureSpec 파싱"""
        try:
            return FeatureSpec(
                feature_id=feature_id,
                name=data.get('name', feature_id),
                feature_type=FeatureType(data.get('type', 'raw')),
                input_series=data.get('inputs', []),
                window_days=data.get('window_days', 0),
                params=data.get('params', {}),
                description=data.get('description'),
                unit=data.get('unit'),
            )
        except Exception as e:
            logger.error(f"Failed to parse feature spec {feature_id}: {e}")
            return None
    
    def _register_default_specs(self) -> None:
        """기본 스펙 등록"""
        defaults = [
            # 스프레드
            FeatureSpec(
                feature_id="SOFR_EFFR_SPREAD",
                name="SOFR-EFFR Spread",
                feature_type=FeatureType.SPREAD,
                input_series=["SOFR", "EFFR"],
                window_days=0,
                description="SOFR와 EFFR의 스프레드",
                unit="bps",
            ),
            # ROC (Rate of Change)
            FeatureSpec(
                feature_id="SOFR_ROC_30D",
                name="SOFR 30D ROC",
                feature_type=FeatureType.ROC,
                input_series=["SOFR"],
                window_days=30,
                description="SOFR 30일 변화율",
                unit="%",
            ),
            FeatureSpec(
                feature_id="VIX_ROC_5D",
                name="VIX 5D ROC",
                feature_type=FeatureType.ROC,
                input_series=["VIX"],
                window_days=5,
                description="VIX 5일 변화율",
                unit="%",
            ),
            # ZScore
            FeatureSpec(
                feature_id="SOFR_ZSCORE_90D",
                name="SOFR 90D Z-Score",
                feature_type=FeatureType.ZSCORE,
                input_series=["SOFR"],
                window_days=90,
                description="SOFR 90일 Z-Score",
            ),
            FeatureSpec(
                feature_id="VIX_ZSCORE_30D",
                name="VIX 30D Z-Score",
                feature_type=FeatureType.ZSCORE,
                input_series=["VIX"],
                window_days=30,
                description="VIX 30일 Z-Score",
            ),
            # YoY (Year over Year)
            FeatureSpec(
                feature_id="CPI_YOY",
                name="CPI YoY",
                feature_type=FeatureType.YOY,
                input_series=["CPI"],
                window_days=365,
                description="CPI 전년 대비 변화율",
                unit="%",
            ),
            # Volatility
            FeatureSpec(
                feature_id="SPY_VOL_20D",
                name="SPY 20D Volatility",
                feature_type=FeatureType.VOLATILITY,
                input_series=["SPY"],
                window_days=20,
                description="SPY 20일 변동성",
                unit="%",
            ),
            # Correlation
            FeatureSpec(
                feature_id="SOFR_SPY_CORR_60D",
                name="SOFR-SPY 60D Correlation",
                feature_type=FeatureType.CORRELATION,
                input_series=["SOFR", "SPY"],
                window_days=60,
                description="SOFR와 SPY의 60일 상관계수",
            ),
        ]
        
        for spec in defaults:
            self.register(spec)
        
        logger.debug(f"Registered {len(defaults)} default feature specs")
    
    def register(self, spec: FeatureSpec) -> None:
        """스펙 등록"""
        self._specs[spec.feature_id] = spec
        logger.debug(f"Registered feature spec: {spec.feature_id}")
    
    def unregister(self, feature_id: str) -> bool:
        """스펙 등록 해제"""
        if feature_id in self._specs:
            del self._specs[feature_id]
            return True
        return False
    
    def get(self, feature_id: str) -> Optional[FeatureSpec]:
        """스펙 조회"""
        return self._specs.get(feature_id)
    
    def get_by_input(self, series_id: str) -> List[FeatureSpec]:
        """
        특정 시계열을 입력으로 사용하는 모든 Feature 조회
        
        증분 업데이트에 필요: series가 업데이트되면 어떤 feature를 재계산해야 하는지
        """
        return [
            spec for spec in self._specs.values()
            if series_id in spec.input_series
        ]
    
    def get_by_type(self, feature_type: FeatureType) -> List[FeatureSpec]:
        """타입별 스펙 조회"""
        return [
            spec for spec in self._specs.values()
            if spec.feature_type == feature_type
        ]
    
    def list_all(self) -> List[FeatureSpec]:
        """모든 스펙 목록"""
        return list(self._specs.values())
    
    def list_feature_ids(self) -> List[str]:
        """모든 Feature ID 목록"""
        return list(self._specs.keys())
    
    def get_required_series(self) -> List[str]:
        """모든 Feature에 필요한 시계열 목록 (중복 제거)"""
        series = set()
        for spec in self._specs.values():
            series.update(spec.input_series)
        return list(series)
    
    def get_stats(self) -> Dict:
        """통계"""
        type_counts = {}
        for spec in self._specs.values():
            t = spec.feature_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "total_specs": len(self._specs),
            "required_series": len(self.get_required_series()),
            "by_type": type_counts,
        }
