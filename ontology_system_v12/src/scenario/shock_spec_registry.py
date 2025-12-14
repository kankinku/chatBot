"""
Shock Spec Registry
Shock 입력 표준화 및 관리

책임:
- Shock 스펙 정의
- 표준화된 Shock 입력
- 프리셋 시나리오 관리
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ShockType(str, Enum):
    """Shock 타입"""
    ABSOLUTE = "absolute"       # 절대값 변화 (예: +25bp)
    RELATIVE = "relative"       # 상대값 변화 (예: +10%)
    SIGMA = "sigma"            # 표준편차 단위 (예: -2σ)
    PERCENTILE = "percentile"  # 백분위 이동 (예: 90th percentile로)


class ShockMagnitude(str, Enum):
    """Shock 강도"""
    SMALL = "small"      # 1σ 이내
    MEDIUM = "medium"    # 1-2σ
    LARGE = "large"      # 2-3σ
    EXTREME = "extreme"  # 3σ 이상


@dataclass
class ShockSpec:
    """단일 Shock 스펙"""
    shock_id: str = field(default_factory=lambda: f"SHOCK_{uuid.uuid4().hex[:8]}")
    
    # 대상
    target_node: str = ""              # 영향받을 노드 (series_id, feature_id, or entity_id)
    target_type: str = "series"        # series, feature, entity
    
    # Shock 정의
    shock_type: ShockType = ShockType.RELATIVE
    shock_value: float = 0.0           # 변화량
    shock_direction: str = "+"         # + 또는 -
    
    # 강도
    magnitude: ShockMagnitude = ShockMagnitude.MEDIUM
    
    # 메타
    description: str = ""
    source: str = ""                   # 시나리오 출처 (예: "fed_rate_hike")


@dataclass
class ScenarioPreset:
    """프리셋 시나리오"""
    preset_id: str
    name: str
    description: str
    shocks: List[ShockSpec]
    break_conditions: List[Dict] = field(default_factory=list)  # 시나리오 무효화 조건
    applicable_regimes: List[str] = field(default_factory=list)


class ShockSpecRegistry:
    """
    Shock 스펙 레지스트리
    
    표준화된 Shock 입력을 관리합니다.
    """
    
    def __init__(self):
        self._specs: Dict[str, ShockSpec] = {}
        self._presets: Dict[str, ScenarioPreset] = {}
        
        self._register_default_presets()
    
    def register_shock(self, spec: ShockSpec) -> None:
        """Shock 스펙 등록"""
        self._specs[spec.shock_id] = spec
        logger.debug(f"Registered shock: {spec.shock_id}")
    
    def get_shock(self, shock_id: str) -> Optional[ShockSpec]:
        """Shock 스펙 조회"""
        return self._specs.get(shock_id)
    
    def create_shock(
        self,
        target_node: str,
        shock_value: float,
        shock_type: ShockType = ShockType.RELATIVE,
        shock_direction: str = "+",
        description: str = "",
    ) -> ShockSpec:
        """Shock 생성 및 등록"""
        magnitude = self._infer_magnitude(shock_value, shock_type)
        
        spec = ShockSpec(
            target_node=target_node,
            shock_type=shock_type,
            shock_value=shock_value,
            shock_direction=shock_direction,
            magnitude=magnitude,
            description=description,
        )
        
        self.register_shock(spec)
        return spec
    
    def _infer_magnitude(self, value: float, shock_type: ShockType) -> ShockMagnitude:
        """Shock 강도 추론"""
        abs_value = abs(value)
        
        if shock_type == ShockType.SIGMA:
            if abs_value < 1:
                return ShockMagnitude.SMALL
            elif abs_value < 2:
                return ShockMagnitude.MEDIUM
            elif abs_value < 3:
                return ShockMagnitude.LARGE
            else:
                return ShockMagnitude.EXTREME
        
        elif shock_type == ShockType.RELATIVE:
            if abs_value < 0.05:  # 5% 이내
                return ShockMagnitude.SMALL
            elif abs_value < 0.15:  # 15% 이내
                return ShockMagnitude.MEDIUM
            elif abs_value < 0.30:
                return ShockMagnitude.LARGE
            else:
                return ShockMagnitude.EXTREME
        
        return ShockMagnitude.MEDIUM
    
    def register_preset(self, preset: ScenarioPreset) -> None:
        """프리셋 등록"""
        self._presets[preset.preset_id] = preset
        # 포함된 Shock들도 등록
        for shock in preset.shocks:
            self._specs[shock.shock_id] = shock
        
        logger.debug(f"Registered preset: {preset.name}")
    
    def get_preset(self, preset_id: str) -> Optional[ScenarioPreset]:
        """프리셋 조회"""
        return self._presets.get(preset_id)
    
    def list_presets(self) -> List[ScenarioPreset]:
        """모든 프리셋"""
        return list(self._presets.values())
    
    def _register_default_presets(self) -> None:
        """기본 프리셋 등록"""
        presets = [
            # Fed 금리 인상 시나리오
            ScenarioPreset(
                preset_id="fed_rate_hike_50bp",
                name="Fed Rate Hike +50bp",
                description="연준 50bp 금리 인상 시나리오",
                shocks=[
                    ShockSpec(
                        target_node="SOFR",
                        target_type="series",
                        shock_type=ShockType.ABSOLUTE,
                        shock_value=0.50,
                        shock_direction="+",
                        magnitude=ShockMagnitude.MEDIUM,
                        description="SOFR +50bp",
                    ),
                    ShockSpec(
                        target_node="T10Y2Y",
                        target_type="series",
                        shock_type=ShockType.ABSOLUTE,
                        shock_value=-0.25,
                        shock_direction="-",
                        magnitude=ShockMagnitude.SMALL,
                        description="Yield Curve Flattening",
                    ),
                ],
                break_conditions=[
                    {"feature": "VIX", "operator": ">", "threshold": 40},
                ],
            ),
            
            # VIX 급등 시나리오
            ScenarioPreset(
                preset_id="vix_spike",
                name="VIX Spike",
                description="VIX 급등 (공포 확산) 시나리오",
                shocks=[
                    ShockSpec(
                        target_node="VIX",
                        target_type="series",
                        shock_type=ShockType.SIGMA,
                        shock_value=2.0,
                        shock_direction="+",
                        magnitude=ShockMagnitude.LARGE,
                        description="VIX +2σ",
                    ),
                ],
                break_conditions=[
                    {"feature": "VIX", "operator": ">", "threshold": 80},
                ],
                applicable_regimes=["risk_off"],
            ),
            
            # 인플레이션 가속 시나리오
            ScenarioPreset(
                preset_id="inflation_acceleration",
                name="Inflation Acceleration",
                description="인플레이션 가속 시나리오",
                shocks=[
                    ShockSpec(
                        target_node="CPI_YOY",
                        target_type="feature",
                        shock_type=ShockType.ABSOLUTE,
                        shock_value=0.01,  # +1%p
                        shock_direction="+",
                        magnitude=ShockMagnitude.MEDIUM,
                        description="CPI YoY +1%p",
                    ),
                    ShockSpec(
                        target_node="GLD",
                        target_type="series",
                        shock_type=ShockType.RELATIVE,
                        shock_value=0.05,
                        shock_direction="+",
                        magnitude=ShockMagnitude.SMALL,
                        description="Gold +5%",
                    ),
                ],
                applicable_regimes=["inflation_up"],
            ),
            
            # 유동성 위기 시나리오
            ScenarioPreset(
                preset_id="liquidity_crisis",
                name="Liquidity Crisis",
                description="유동성 위기 시나리오",
                shocks=[
                    ShockSpec(
                        target_node="SOFR",
                        target_type="series",
                        shock_type=ShockType.ABSOLUTE,
                        shock_value=1.0,
                        shock_direction="+",
                        magnitude=ShockMagnitude.EXTREME,
                        description="SOFR +100bp",
                    ),
                    ShockSpec(
                        target_node="VIX",
                        target_type="series",
                        shock_type=ShockType.SIGMA,
                        shock_value=3.0,
                        shock_direction="+",
                        magnitude=ShockMagnitude.EXTREME,
                        description="VIX +3σ",
                    ),
                    ShockSpec(
                        target_node="SPY",
                        target_type="series",
                        shock_type=ShockType.RELATIVE,
                        shock_value=-0.10,
                        shock_direction="-",
                        magnitude=ShockMagnitude.EXTREME,
                        description="SPY -10%",
                    ),
                ],
                break_conditions=[
                    {"feature": "SPY", "operator": "<", "threshold": -30},
                ],
            ),
            
            # 골디락스 시나리오 (지속)
            ScenarioPreset(
                preset_id="goldilocks_continuation",
                name="Goldilocks Continuation",
                description="저인플레, 적정성장 지속 시나리오",
                shocks=[
                    ShockSpec(
                        target_node="CPI_YOY",
                        target_type="feature",
                        shock_type=ShockType.ABSOLUTE,
                        shock_value=-0.005,  # -0.5%p
                        shock_direction="-",
                        magnitude=ShockMagnitude.SMALL,
                        description="CPI YoY -0.5%p",
                    ),
                    ShockSpec(
                        target_node="SPY",
                        target_type="series",
                        shock_type=ShockType.RELATIVE,
                        shock_value=0.03,
                        shock_direction="+",
                        magnitude=ShockMagnitude.SMALL,
                        description="SPY +3%",
                    ),
                ],
                applicable_regimes=["risk_on", "goldilocks"],
            ),
        ]
        
        for preset in presets:
            self.register_preset(preset)
        
        logger.debug(f"Registered {len(presets)} default scenario presets")
    
    def get_stats(self) -> Dict:
        """통계"""
        return {
            "total_shocks": len(self._specs),
            "total_presets": len(self._presets),
            "presets": [p.name for p in self._presets.values()],
        }
