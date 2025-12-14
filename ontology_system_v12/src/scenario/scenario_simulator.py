"""
Scenario Simulator
Shock를 노드/feature에 주입하고 영향 전파 시뮬레이션

책임:
- Shock 주입
- 경로추론/부호전파로 영향 전파
- 시나리오 결과 생성
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.scenario.shock_spec_registry import (
    ShockSpecRegistry, ShockSpec, ScenarioPreset, ShockType, ShockMagnitude
)

logger = logging.getLogger(__name__)


class ImpactStrength(str, Enum):
    """영향 강도"""
    NEGLIGIBLE = "negligible"  # 무시할 수준
    WEAK = "weak"              # 약함
    MODERATE = "moderate"      # 중간
    STRONG = "strong"          # 강함
    VERY_STRONG = "very_strong"  # 매우 강함


@dataclass
class NodeImpact:
    """노드별 영향"""
    node_id: str
    node_name: str
    impact_direction: str         # + 또는 -
    impact_strength: ImpactStrength
    impact_value: float           # 정량적 영향 (추정)
    propagation_path: List[str]   # 전파 경로
    confidence: float = 0.5
    explanation: str = ""


@dataclass
class BreakConditionResult:
    """Break Condition 평가 결과"""
    condition: Dict
    is_triggered: bool
    current_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class ScenarioResult:
    """시나리오 시뮬레이션 결과"""
    scenario_id: str
    scenario_name: str
    simulated_at: datetime
    
    # 적용된 Shock들
    applied_shocks: List[ShockSpec]
    
    # 영향 분석
    node_impacts: List[NodeImpact]
    summary_direction: str        # 전체적 방향 (+, -, mixed)
    summary_strength: ImpactStrength
    
    # Break Conditions
    break_conditions: List[BreakConditionResult] = field(default_factory=list)
    scenario_valid: bool = True   # break condition 미충족 시 True
    
    # 메타
    regime_context: Optional[str] = None
    execution_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)


class ScenarioSimulator:
    """
    시나리오 시뮬레이터
    
    Shock를 주입하고 KG 경로를 따라 영향을 전파합니다.
    """
    
    def __init__(
        self,
        shock_registry: ShockSpecRegistry,
        dependency_manager: Optional[Any] = None,
        edge_weight_fusion: Optional[Any] = None,
        current_feature_values: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            shock_registry: Shock 스펙 레지스트리
            dependency_manager: 의존성 그래프 관리자
            edge_weight_fusion: Edge 가중치 융합 엔진
            current_feature_values: 현재 feature 값들
        """
        self.shock_registry = shock_registry
        self.dependency_manager = dependency_manager
        self.edge_weight_fusion = edge_weight_fusion
        self.current_values = current_feature_values or {}
        
        # 관계 맵 (간단한 룰 기반)
        self._relation_map = self._build_default_relation_map()
    
    def simulate_preset(
        self,
        preset_id: str,
        regime: Optional[str] = None,
    ) -> ScenarioResult:
        """
        프리셋 시나리오 시뮬레이션
        
        Args:
            preset_id: 프리셋 ID
            regime: 현재 레짐 (optional)
        
        Returns:
            ScenarioResult
        """
        import time
        start_time = time.time()
        
        preset = self.shock_registry.get_preset(preset_id)
        if not preset:
            return ScenarioResult(
                scenario_id=f"SIM_{preset_id}",
                scenario_name=preset_id,
                simulated_at=datetime.now(),
                applied_shocks=[],
                node_impacts=[],
                summary_direction="unknown",
                summary_strength=ImpactStrength.NEGLIGIBLE,
                warnings=[f"Preset not found: {preset_id}"],
            )
        
        return self.simulate_shocks(
            shocks=preset.shocks,
            scenario_name=preset.name,
            break_conditions=preset.break_conditions,
            regime=regime,
        )
    
    def simulate_shocks(
        self,
        shocks: List[ShockSpec],
        scenario_name: str = "Custom Scenario",
        break_conditions: Optional[List[Dict]] = None,
        regime: Optional[str] = None,
    ) -> ScenarioResult:
        """
        Shock 리스트 시뮬레이션
        
        Args:
            shocks: Shock 스펙 리스트
            scenario_name: 시나리오 이름
            break_conditions: 무효화 조건들
            regime: 현재 레짐
        
        Returns:
            ScenarioResult
        """
        import time
        start_time = time.time()
        
        scenario_id = f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        node_impacts = []
        warnings = []
        
        # 각 Shock별 영향 전파
        for shock in shocks:
            impacts = self._propagate_shock(shock)
            node_impacts.extend(impacts)
        
        # 영향 집계
        summary_direction, summary_strength = self._aggregate_impacts(node_impacts)
        
        # Break Conditions 평가
        break_results = []
        scenario_valid = True
        
        if break_conditions:
            for cond in break_conditions:
                result = self._evaluate_break_condition(cond)
                break_results.append(result)
                if result.is_triggered:
                    scenario_valid = False
                    warnings.append(f"Break condition triggered: {cond}")
        
        elapsed = (time.time() - start_time) * 1000
        
        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            simulated_at=datetime.now(),
            applied_shocks=shocks,
            node_impacts=node_impacts,
            summary_direction=summary_direction,
            summary_strength=summary_strength,
            break_conditions=break_results,
            scenario_valid=scenario_valid,
            regime_context=regime,
            execution_time_ms=elapsed,
            warnings=warnings,
        )
    
    def _propagate_shock(self, shock: ShockSpec) -> List[NodeImpact]:
        """
        Shock 영향 전파
        
        간단한 룰 기반:
        - relation_map에서 관계 조회
        - 부호 전파
        - 강도 감쇠
        """
        impacts = []
        
        # 직접 영향 노드
        target = shock.target_node
        relations = self._relation_map.get(target, [])
        
        for rel in relations:
            affected_node = rel["tail"]
            rel_sign = rel["sign"]
            rel_strength = rel.get("strength", 0.5)
            
            # 영향 방향 계산 (Shock 방향 × 관계 부호)
            if shock.shock_direction == "+":
                impact_dir = rel_sign
            else:
                impact_dir = "-" if rel_sign == "+" else "+"
            
            # 영향 강도 계산
            shock_strength = self._magnitude_to_value(shock.magnitude)
            impact_value = shock_strength * rel_strength * abs(shock.shock_value)
            impact_strength = self._value_to_strength(impact_value)
            
            impacts.append(NodeImpact(
                node_id=affected_node,
                node_name=rel.get("tail_name", affected_node),
                impact_direction=impact_dir,
                impact_strength=impact_strength,
                impact_value=impact_value,
                propagation_path=[target, affected_node],
                confidence=rel_strength,
                explanation=f"{target} {shock.shock_direction}{shock.shock_value} → {affected_node} {impact_dir}",
            ))
            
            # 2차 전파 (간접 영향)
            secondary_relations = self._relation_map.get(affected_node, [])
            for sec_rel in secondary_relations[:2]:  # 최대 2개
                sec_node = sec_rel["tail"]
                sec_sign = sec_rel["sign"]
                
                # 부호 전파
                if impact_dir == "+":
                    sec_impact_dir = sec_sign
                else:
                    sec_impact_dir = "-" if sec_sign == "+" else "+"
                
                # 감쇠
                sec_impact_value = impact_value * 0.5
                sec_strength = self._value_to_strength(sec_impact_value)
                
                impacts.append(NodeImpact(
                    node_id=sec_node,
                    node_name=sec_rel.get("tail_name", sec_node),
                    impact_direction=sec_impact_dir,
                    impact_strength=sec_strength,
                    impact_value=sec_impact_value,
                    propagation_path=[target, affected_node, sec_node],
                    confidence=rel_strength * 0.5,
                    explanation=f"2차 영향: {affected_node} → {sec_node}",
                ))
        
        return impacts
    
    def _build_default_relation_map(self) -> Dict[str, List[Dict]]:
        """기본 관계 맵 구축"""
        return {
            "SOFR": [
                {"tail": "growth_stock", "tail_name": "성장주", "sign": "-", "strength": 0.7},
                {"tail": "value_stock", "tail_name": "가치주", "sign": "-", "strength": 0.4},
                {"tail": "TLT", "tail_name": "장기국채", "sign": "-", "strength": 0.6},
                {"tail": "HYG", "tail_name": "하이일드", "sign": "-", "strength": 0.5},
            ],
            "VIX": [
                {"tail": "SPY", "tail_name": "S&P500", "sign": "-", "strength": 0.8},
                {"tail": "growth_stock", "tail_name": "성장주", "sign": "-", "strength": 0.7},
                {"tail": "GLD", "tail_name": "금", "sign": "+", "strength": 0.4},
            ],
            "CPI_YOY": [
                {"tail": "SOFR", "tail_name": "SOFR", "sign": "+", "strength": 0.6},
                {"tail": "GLD", "tail_name": "금", "sign": "+", "strength": 0.5},
                {"tail": "TLT", "tail_name": "장기국채", "sign": "-", "strength": 0.6},
            ],
            "SPY": [
                {"tail": "growth_stock", "tail_name": "성장주", "sign": "+", "strength": 0.9},
                {"tail": "value_stock", "tail_name": "가치주", "sign": "+", "strength": 0.8},
            ],
            "GLD": [
                {"tail": "risk_sentiment", "tail_name": "위험 심리", "sign": "-", "strength": 0.3},
            ],
        }
    
    def _magnitude_to_value(self, magnitude: ShockMagnitude) -> float:
        """강도 → 수치"""
        mapping = {
            ShockMagnitude.SMALL: 0.3,
            ShockMagnitude.MEDIUM: 0.5,
            ShockMagnitude.LARGE: 0.7,
            ShockMagnitude.EXTREME: 1.0,
        }
        return mapping.get(magnitude, 0.5)
    
    def _value_to_strength(self, value: float) -> ImpactStrength:
        """수치 → 영향 강도"""
        abs_val = abs(value)
        if abs_val < 0.1:
            return ImpactStrength.NEGLIGIBLE
        elif abs_val < 0.25:
            return ImpactStrength.WEAK
        elif abs_val < 0.5:
            return ImpactStrength.MODERATE
        elif abs_val < 0.75:
            return ImpactStrength.STRONG
        else:
            return ImpactStrength.VERY_STRONG
    
    def _aggregate_impacts(
        self,
        impacts: List[NodeImpact],
    ) -> Tuple[str, ImpactStrength]:
        """영향 집계"""
        if not impacts:
            return "neutral", ImpactStrength.NEGLIGIBLE
        
        positive_sum = sum(i.impact_value for i in impacts if i.impact_direction == "+")
        negative_sum = sum(abs(i.impact_value) for i in impacts if i.impact_direction == "-")
        
        if positive_sum > negative_sum * 1.2:
            direction = "+"
            total = positive_sum
        elif negative_sum > positive_sum * 1.2:
            direction = "-"
            total = negative_sum
        else:
            direction = "mixed"
            total = (positive_sum + negative_sum) / 2
        
        strength = self._value_to_strength(total)
        
        return direction, strength
    
    def _evaluate_break_condition(self, condition: Dict) -> BreakConditionResult:
        """Break Condition 평가"""
        feature = condition.get("feature")
        operator = condition.get("operator", ">")
        threshold = condition.get("threshold", 0)
        
        current_value = self.current_values.get(feature)
        
        if current_value is None:
            return BreakConditionResult(
                condition=condition,
                is_triggered=False,
                current_value=None,
                threshold=threshold,
            )
        
        is_triggered = False
        if operator == ">":
            is_triggered = current_value > threshold
        elif operator == ">=":
            is_triggered = current_value >= threshold
        elif operator == "<":
            is_triggered = current_value < threshold
        elif operator == "<=":
            is_triggered = current_value <= threshold
        
        return BreakConditionResult(
            condition=condition,
            is_triggered=is_triggered,
            current_value=current_value,
            threshold=threshold,
        )
    
    def generate_report(self, result: ScenarioResult) -> str:
        """시나리오 결과 리포트 생성"""
        lines = [
            f"# 시나리오 분석 리포트: {result.scenario_name}",
            f"",
            f"**시뮬레이션 시점**: {result.simulated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**레짐 컨텍스트**: {result.regime_context or 'N/A'}",
            f"**시나리오 유효성**: {'✅ 유효' if result.scenario_valid else '⚠️ Break Condition 충족'}",
            f"",
            f"## 적용된 Shock",
        ]
        
        for shock in result.applied_shocks:
            lines.append(
                f"- **{shock.target_node}**: {shock.shock_direction}{shock.shock_value} "
                f"({shock.shock_type.value}, {shock.magnitude.value})"
            )
        
        lines.extend([
            f"",
            f"## 전체 영향 요약",
            f"- **방향**: {result.summary_direction}",
            f"- **강도**: {result.summary_strength.value}",
            f"",
            f"## 개별 노드 영향",
        ])
        
        for impact in result.node_impacts[:10]:  # 상위 10개
            lines.append(
                f"- **{impact.node_name}** ({impact.node_id}): "
                f"{impact.impact_direction} ({impact.impact_strength.value})"
            )
            if impact.explanation:
                lines.append(f"  - {impact.explanation}")
        
        if result.break_conditions:
            lines.extend([
                f"",
                f"## Break Conditions",
            ])
            for bc in result.break_conditions:
                status = "🔴 충족" if bc.is_triggered else "🟢 미충족"
                lines.append(
                    f"- {status}: {bc.condition.get('feature')} "
                    f"{bc.condition.get('operator')} {bc.threshold}"
                )
        
        if result.warnings:
            lines.extend([
                f"",
                f"## 주의사항",
            ])
            for warning in result.warnings:
                lines.append(f"- ⚠️ {warning}")
        
        return "\n".join(lines)
