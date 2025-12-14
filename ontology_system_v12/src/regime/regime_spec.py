"""
Regime Spec Manager
레짐 정의(스펙) 관리

책임:
- 레짐 조건 정의
- Config 로드
- 스펙 조회
"""
import logging
from typing import Dict, List, Optional
from pathlib import Path
import yaml

from src.shared.schemas import RegimeSpec, RegimeCondition, RegimeType

logger = logging.getLogger(__name__)


class RegimeSpecManager:
    """
    레짐 정의 관리자
    
    규칙 기반 레짐 정의를 관리합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: regimes.yaml 경로
        """
        self._specs: Dict[str, RegimeSpec] = {}
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self._register_default_specs()
    
    def load_from_file(self, config_path: str) -> int:
        """YAML에서 스펙 로드"""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Regime config not found: {config_path}")
            self._register_default_specs()
            return 0
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            regimes = config.get('regimes', [])
            loaded = 0
            
            for regime_data in regimes:
                spec = self._parse_spec(regime_data)
                if spec:
                    self.register(spec)
                    loaded += 1
            
            logger.info(f"Loaded {loaded} regime specs from {config_path}")
            return loaded
            
        except Exception as e:
            logger.error(f"Failed to load regime config: {e}")
            self._register_default_specs()
            return 0
    
    def _parse_spec(self, data: Dict) -> Optional[RegimeSpec]:
        """Dict에서 RegimeSpec 파싱"""
        try:
            conditions = []
            for cond_data in data.get('conditions', []):
                conditions.append(RegimeCondition(
                    feature=cond_data['feature'],
                    operator=cond_data['operator'],
                    threshold=cond_data['threshold'],
                ))
            
            return RegimeSpec(
                name=data['name'],
                regime_type=RegimeType(data['name']),  # name과 동일
                conditions=conditions,
                priority=data.get('priority', 1),
                description=data.get('description'),
                is_active=data.get('is_active', True),
            )
        except Exception as e:
            logger.error(f"Failed to parse regime spec: {e}")
            return None
    
    def _register_default_specs(self) -> None:
        """기본 스펙 등록"""
        defaults = [
            # Risk On
            RegimeSpec(
                name="risk_on",
                regime_type=RegimeType.RISK_ON,
                conditions=[
                    RegimeCondition(feature="VIX", operator="<", threshold=20),
                    RegimeCondition(feature="SPY_ROC_20D", operator=">", threshold=0),
                ],
                priority=1,
                description="위험 선호 국면: 낮은 VIX, 상승 추세",
            ),
            # Risk Off
            RegimeSpec(
                name="risk_off",
                regime_type=RegimeType.RISK_OFF,
                conditions=[
                    RegimeCondition(feature="VIX", operator=">=", threshold=25),
                ],
                priority=2,
                description="위험 회피 국면: 높은 VIX",
            ),
            # Inflation Up
            RegimeSpec(
                name="inflation_up",
                regime_type=RegimeType.INFLATION_UP,
                conditions=[
                    RegimeCondition(feature="CPI_YOY", operator=">", threshold=0.03),
                ],
                priority=1,
                description="인플레이션 상승 국면: CPI YoY > 3%",
            ),
            # Disinflation
            RegimeSpec(
                name="disinflation",
                regime_type=RegimeType.DISINFLATION,
                conditions=[
                    RegimeCondition(feature="CPI_YOY", operator="<", threshold=0.02),
                ],
                priority=2,
                description="디스인플레이션 국면: CPI YoY < 2%",
            ),
            # Growth Up
            RegimeSpec(
                name="growth_up",
                regime_type=RegimeType.GROWTH_UP,
                conditions=[
                    RegimeCondition(feature="SPY_ROC_90D", operator=">", threshold=0.05),
                ],
                priority=1,
                description="성장 상승 국면",
            ),
            # Growth Down
            RegimeSpec(
                name="growth_down",
                regime_type=RegimeType.GROWTH_DOWN,
                conditions=[
                    RegimeCondition(feature="SPY_ROC_90D", operator="<", threshold=-0.05),
                ],
                priority=2,
                description="성장 하락 국면",
            ),
            # Liquidity Abundant
            RegimeSpec(
                name="liquidity_abundant",
                regime_type=RegimeType.LIQUIDITY_ABUNDANT,
                conditions=[
                    RegimeCondition(feature="SOFR_ZSCORE_90D", operator="<", threshold=-1),
                ],
                priority=1,
                description="유동성 풍부 국면: 금리 낮은 상태",
            ),
            # Liquidity Tight
            RegimeSpec(
                name="liquidity_tight",
                regime_type=RegimeType.LIQUIDITY_TIGHT,
                conditions=[
                    RegimeCondition(feature="SOFR_ZSCORE_90D", operator=">", threshold=1),
                ],
                priority=2,
                description="유동성 긴축 국면: 금리 높은 상태",
            ),
        ]
        
        for spec in defaults:
            self.register(spec)
        
        logger.debug(f"Registered {len(defaults)} default regime specs")
    
    def register(self, spec: RegimeSpec) -> None:
        """스펙 등록"""
        self._specs[spec.regime_id] = spec
        logger.debug(f"Registered regime spec: {spec.name}")
    
    def get(self, regime_id: str) -> Optional[RegimeSpec]:
        """ID로 스펙 조회"""
        return self._specs.get(regime_id)
    
    def get_by_type(self, regime_type: RegimeType) -> Optional[RegimeSpec]:
        """타입으로 스펙 조회"""
        for spec in self._specs.values():
            if spec.regime_type == regime_type:
                return spec
        return None
    
    def list_all(self, active_only: bool = True) -> List[RegimeSpec]:
        """모든 스펙"""
        specs = list(self._specs.values())
        if active_only:
            specs = [s for s in specs if s.is_active]
        return sorted(specs, key=lambda s: s.priority, reverse=True)
    
    def get_required_features(self) -> List[str]:
        """모든 스펙에 필요한 Feature 목록"""
        features = set()
        for spec in self._specs.values():
            for cond in spec.conditions:
                features.add(cond.feature)
        return list(features)
    
    def get_stats(self) -> Dict:
        """통계"""
        return {
            "total_specs": len(self._specs),
            "active_specs": sum(1 for s in self._specs.values() if s.is_active),
            "required_features": len(self.get_required_features()),
        }
