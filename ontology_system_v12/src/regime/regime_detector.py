"""
Regime Detector
현재 레짐 탐지

책임:
- Feature 값 기반 레짐 판단
- 확률/불확실성 계산
- 레짐 전환 감지
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from src.shared.schemas import (
    RegimeType,
    RegimeSpec,
    RegimeCondition,
    RegimeDetectionResult,
    FeatureValue,
)
from src.regime.regime_spec import RegimeSpecManager
from src.features.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    레짐 탐지기
    
    Feature 값을 기반으로 현재 레짐을 판단합니다.
    
    특징:
    - 다중 레짐 동시 탐지 (확률 기반)
    - 조건 부분 충족 시 확률 반영
    - 불확실성 계산
    """
    
    def __init__(
        self,
        spec_manager: RegimeSpecManager,
        feature_store: FeatureStore,
    ):
        """
        Args:
            spec_manager: 레짐 스펙 관리자
            feature_store: Feature 저장소
        """
        self.spec_manager = spec_manager
        self.feature_store = feature_store
        
        # 이전 레짐 (전환 감지용)
        self._previous_regime: Optional[RegimeType] = None
    
    def detect(
        self,
        timestamp: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
    ) -> RegimeDetectionResult:
        """
        현재 레짐 탐지
        
        Args:
            timestamp: 평가 시점 (None이면 현재)
            as_of: 데이터 기준 시점 (리플레이용)
        
        Returns:
            RegimeDetectionResult
        """
        timestamp = timestamp or datetime.now()
        
        # 필요한 Feature 값 조회
        feature_snapshot = self._get_feature_snapshot(as_of)
        
        # 각 레짐별 확률 계산
        regime_probs: Dict[RegimeType, float] = {}
        
        for spec in self.spec_manager.list_all(active_only=True):
            prob = self._evaluate_regime(spec, feature_snapshot)
            if prob > 0:
                regime_probs[spec.regime_type] = prob
        
        # 주요 레짐 선택
        primary_regime = None
        primary_prob = 0.0
        
        if regime_probs:
            # 가장 높은 확률 (우선순위 고려)
            sorted_regimes = sorted(
                regime_probs.items(),
                key=lambda x: (x[1], self._get_priority(x[0])),
                reverse=True
            )
            primary_regime, primary_prob = sorted_regimes[0]
        
        # 불확실성 계산
        uncertainty = self._calculate_uncertainty(regime_probs, feature_snapshot)
        
        # 레짐 전환 체크
        regime_changed = (
            primary_regime != self._previous_regime and
            primary_prob > 0.5
        )
        if regime_changed:
            logger.info(
                f"Regime change detected: {self._previous_regime} -> {primary_regime}"
            )
            self._previous_regime = primary_regime
        
        return RegimeDetectionResult(
            timestamp=timestamp,
            detected_regimes=regime_probs,
            primary_regime=primary_regime,
            primary_probability=primary_prob,
            uncertainty=uncertainty,
            feature_snapshot=feature_snapshot,
        )
    
    def _get_feature_snapshot(
        self,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """필요한 Feature 최신값 조회"""
        required_features = self.spec_manager.get_required_features()
        snapshot = {}
        
        for feature_id in required_features:
            fv = self.feature_store.get_last(feature_id, as_of)
            if fv:
                snapshot[feature_id] = fv.value
        
        return snapshot
    
    def _evaluate_regime(
        self,
        spec: RegimeSpec,
        feature_snapshot: Dict[str, float],
    ) -> float:
        """
        단일 레짐 평가
        
        Returns:
            확률 (0~1)
        """
        if not spec.conditions:
            return 0.0
        
        satisfied = 0
        total = len(spec.conditions)
        
        for cond in spec.conditions:
            if self._evaluate_condition(cond, feature_snapshot):
                satisfied += 1
        
        # 부분 충족 시 비례 확률
        return satisfied / total
    
    def _evaluate_condition(
        self,
        condition: RegimeCondition,
        feature_snapshot: Dict[str, float],
    ) -> bool:
        """단일 조건 평가"""
        value = feature_snapshot.get(condition.feature)
        if value is None:
            return False
        
        threshold = condition.threshold
        op = condition.operator
        
        if op == ">":
            return value > threshold
        elif op == ">=":
            return value >= threshold
        elif op == "<":
            return value < threshold
        elif op == "<=":
            return value <= threshold
        elif op == "==":
            return abs(value - threshold) < 1e-9
        elif op == "between":
            # threshold가 [low, high] 리스트인 경우
            if isinstance(threshold, list) and len(threshold) == 2:
                return threshold[0] <= value <= threshold[1]
        
        return False
    
    def _calculate_uncertainty(
        self,
        regime_probs: Dict[RegimeType, float],
        feature_snapshot: Dict[str, float],
    ) -> float:
        """
        불확실성 계산
        
        높을수록 불확실:
        - 여러 레짐이 비슷한 확률
        - Feature 값 누락
        - 극단적 조건 근처
        """
        if not regime_probs:
            return 1.0  # 완전 불확실
        
        probs = list(regime_probs.values())
        
        # 확률 분산 (낮을수록 불확실)
        if len(probs) == 1:
            variance_factor = 0.0
        else:
            mean_prob = sum(probs) / len(probs)
            variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
            # 분산이 낮으면 불확실 (레짐들이 비슷)
            variance_factor = 1.0 / (1.0 + variance * 10)
        
        # Feature 누락 비율
        required = self.spec_manager.get_required_features()
        missing = sum(1 for f in required if f not in feature_snapshot)
        missing_factor = missing / len(required) if required else 0
        
        # 최고 확률 (높을수록 확실)
        max_prob = max(probs) if probs else 0
        prob_factor = 1.0 - max_prob
        
        # 종합 불확실성
        uncertainty = (
            variance_factor * 0.3 +
            missing_factor * 0.4 +
            prob_factor * 0.3
        )
        
        return min(1.0, max(0.0, uncertainty))
    
    def _get_priority(self, regime_type: RegimeType) -> int:
        """레짐 우선순위 조회"""
        spec = self.spec_manager.get_by_type(regime_type)
        return spec.priority if spec else 0
    
    def detect_transition(
        self,
        timestamps: List[datetime],
        as_of: Optional[datetime] = None,
    ) -> List[Tuple[datetime, RegimeType, RegimeType]]:
        """
        레짐 전환 시점 탐지
        
        Args:
            timestamps: 검사할 시점 리스트 (정렬됨)
            as_of: 데이터 기준 시점
        
        Returns:
            (시점, 이전 레짐, 새 레짐) 리스트
        """
        transitions = []
        prev_regime = None
        
        for ts in sorted(timestamps):
            result = self.detect(ts, as_of)
            
            if result.primary_regime and result.primary_probability > 0.5:
                if prev_regime and result.primary_regime != prev_regime:
                    transitions.append((ts, prev_regime, result.primary_regime))
                prev_regime = result.primary_regime
        
        return transitions
    
    def get_regime_applicability(
        self,
        regime: Optional[RegimeType] = None,
        relation_type: str = "Affect",
    ) -> Dict[str, float]:
        """
        현재/특정 레짐에서의 관계 타입별 적용 강도
        
        relation_types.yaml의 regime_applicability 참조
        """
        # TODO: relation_types.yaml 로드하여 반환
        # 현재는 기본값 반환
        if not regime:
            result = self.detect()
            regime = result.primary_regime
        
        # 기본 적용 강도
        default_applicability = {
            RegimeType.RISK_ON: 1.0,
            RegimeType.RISK_OFF: 1.0,
            RegimeType.INFLATION_UP: 1.0,
            RegimeType.DISINFLATION: 1.0,
            RegimeType.GROWTH_UP: 1.0,
            RegimeType.GROWTH_DOWN: 1.0,
            RegimeType.LIQUIDITY_ABUNDANT: 1.0,
            RegimeType.LIQUIDITY_TIGHT: 1.0,
        }
        
        return {k.value: v for k, v in default_applicability.items()}
    
    def get_stats(self) -> Dict:
        """통계"""
        return {
            "previous_regime": self._previous_regime.value if self._previous_regime else None,
            "available_specs": len(self.spec_manager.list_all()),
        }
