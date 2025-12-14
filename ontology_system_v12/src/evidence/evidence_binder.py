"""
Evidence Binder
Feature 값을 읽어 관계별 Evidence Score 산출

책임:
- Feature 값 조회
- pro_score / con_score / total_score 계산
- trace (기여도 추적) 생성
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from src.shared.schemas import (
    EvidenceSpec,
    EvidenceScore,
    EvidenceDirection,
    RegimeType,
    FeatureValue,
)
from src.evidence.evidence_spec_registry import EdgeEvidenceSpecRegistry
from src.features.feature_store import FeatureStore

logger = logging.getLogger(__name__)


@dataclass
class BindingResult:
    """바인딩 결과"""
    success: bool
    edge_id: str
    score: Optional[EvidenceScore] = None
    error_message: Optional[str] = None


class EvidenceBinder:
    """
    Evidence Binder
    
    Feature 값을 읽어 관계별 Evidence Score를 산출합니다.
    
    주요 기능:
    - 스펙에 정의된 Feature 값 조회
    - 방향에 따른 점수 계산
    - 시차(lag) 효과 탐색하여 최적 lag 선택
    - trace 생성 (설명 가능성)
    """
    
    def __init__(
        self,
        spec_registry: EdgeEvidenceSpecRegistry,
        feature_store: FeatureStore,
    ):
        """
        Args:
            spec_registry: Evidence 스펙 레지스트리
            feature_store: Feature 저장소
        """
        self.spec_registry = spec_registry
        self.feature_store = feature_store
    
    def bind(
        self,
        edge_id: str,
        head_id: str,
        head_name: str,
        head_type: str,
        tail_id: str,
        tail_name: str,
        tail_type: str,
        relation_type: str,
        polarity: str,
        timestamp: Optional[datetime] = None,
        regime: Optional[RegimeType] = None,
    ) -> BindingResult:
        """
        Edge에 대한 Evidence Score 계산
        
        Args:
            edge_id: Edge ID
            head_id, head_name, head_type: Head 엔티티 정보
            tail_id, tail_name, tail_type: Tail 엔티티 정보
            relation_type: 관계 타입
            polarity: 관계 부호
            timestamp: 평가 시점 (None이면 현재)
            regime: 현재 레짐
        
        Returns:
            BindingResult
        """
        timestamp = timestamp or datetime.now()
        
        # 매칭 스펙 찾기
        spec = self.spec_registry.find_matching_spec(
            head_id, head_name, head_type,
            tail_id, tail_name, tail_type,
            relation_type, polarity
        )
        
        if not spec:
            return BindingResult(
                success=False,
                edge_id=edge_id,
                error_message="No matching evidence spec found",
            )
        
        try:
            # Evidence Score 계산
            score = self._compute_score(
                edge_id, head_id, tail_id, relation_type,
                spec, timestamp, regime
            )
            
            return BindingResult(
                success=True,
                edge_id=edge_id,
                score=score,
            )
            
        except Exception as e:
            logger.error(f"Evidence binding failed: {e}")
            return BindingResult(
                success=False,
                edge_id=edge_id,
                error_message=str(e),
            )
    
    def bind_batch(
        self,
        edges: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None,
        regime: Optional[RegimeType] = None,
    ) -> List[BindingResult]:
        """배치 바인딩"""
        results = []
        for edge in edges:
            result = self.bind(
                edge_id=edge.get("edge_id", ""),
                head_id=edge.get("head_id", ""),
                head_name=edge.get("head_name", ""),
                head_type=edge.get("head_type", ""),
                tail_id=edge.get("tail_id", ""),
                tail_name=edge.get("tail_name", ""),
                tail_type=edge.get("tail_type", ""),
                relation_type=edge.get("relation_type", ""),
                polarity=edge.get("polarity", ""),
                timestamp=timestamp,
                regime=regime,
            )
            results.append(result)
        return results
    
    def _compute_score(
        self,
        edge_id: str,
        head_id: str,
        tail_id: str,
        relation_type: str,
        spec: EvidenceSpec,
        timestamp: datetime,
        regime: Optional[RegimeType],
    ) -> EvidenceScore:
        """Evidence Score 계산 로직"""
        trace = []
        
        # 각 lag에 대해 점수 계산하여 최적 lag 선택
        best_score = None
        best_lag = 0
        
        for lag in spec.lag_days:
            lag_timestamp = timestamp - timedelta(days=lag)
            
            pro_score = 0.0
            con_score = 0.0
            lag_trace = []
            total_weight = 0.0
            
            for ef in spec.evidence_features:
                feature_id = ef.get("feature", "")
                direction = ef.get("direction", "positive")
                weight = ef.get("weight", 1.0)
                
                # Feature 값 조회
                fv = self.feature_store.get_at(feature_id, lag_timestamp)
                if not fv:
                    # 최근값으로 대체
                    fv = self.feature_store.get_last(feature_id, lag_timestamp)
                
                if not fv:
                    lag_trace.append({
                        "feature": feature_id,
                        "value": None,
                        "contribution": 0,
                        "reason": "missing",
                    })
                    continue
                
                # 방향에 따른 기여도 계산
                contribution = self._calculate_contribution(
                    fv.value, direction, spec.thresholds
                )
                
                if contribution > 0:
                    pro_score += contribution * weight
                else:
                    con_score += abs(contribution) * weight
                
                total_weight += weight
                
                lag_trace.append({
                    "feature": feature_id,
                    "value": fv.value,
                    "direction": direction,
                    "weight": weight,
                    "contribution": contribution * weight,
                })
            
            # 정규화
            if total_weight > 0:
                pro_score /= total_weight
                con_score /= total_weight
            
            total_score = pro_score - con_score
            
            # 최적 lag 선택 (가장 높은 |total_score|)
            if best_score is None or abs(total_score) > abs(best_score):
                best_score = total_score
                best_lag = lag
                trace = lag_trace
        
        # Regime 조정
        regime_adjustment = 1.0
        if regime:
            regime_key = regime.value
            regime_adjustment = spec.regime_applicability.get(regime_key, 1.0)
        
        adjusted_score = (best_score or 0) * regime_adjustment
        
        # Confidence 계산 (trace에서 유효한 feature 비율)
        valid_features = sum(1 for t in trace if t.get("value") is not None)
        total_features = len(trace) if trace else 1
        confidence = valid_features / total_features
        
        return EvidenceScore(
            edge_id=edge_id,
            head_id=head_id,
            tail_id=tail_id,
            relation_type=relation_type,
            pro_score=max(0, best_score or 0) if best_score and best_score > 0 else 0,
            con_score=abs(best_score) if best_score and best_score < 0 else 0,
            total_score=adjusted_score,
            timestamp=timestamp,
            regime=regime,
            regime_adjustment=regime_adjustment,
            trace=trace,
            best_lag_days=best_lag,
            confidence=confidence,
        )
    
    def _calculate_contribution(
        self,
        value: float,
        direction: str,
        thresholds: Dict[str, float],
    ) -> float:
        """
        Feature 값의 기여도 계산
        
        Args:
            value: Feature 값
            direction: 기대 방향 ("positive" = 값↑→pro, "negative" = 값↓→pro)
            thresholds: 임계값
        
        Returns:
            기여도 (-1 ~ 1)
        """
        # 임계값 조회
        strong_pro = thresholds.get("strong_pro", 0.7)
        weak_pro = thresholds.get("weak_pro", 0.3)
        neutral_low = thresholds.get("neutral_low", -0.3)
        neutral_high = thresholds.get("neutral_high", 0.3)
        weak_con = thresholds.get("weak_con", -0.3)
        strong_con = thresholds.get("strong_con", -0.7)
        
        # 방향 반전 (negative면 값을 뒤집어서 평가)
        adjusted_value = value if direction == "positive" else -value
        
        # 구간별 기여도
        if adjusted_value >= strong_pro:
            return 1.0
        elif adjusted_value >= weak_pro:
            return 0.5
        elif adjusted_value >= neutral_high:
            return 0.2
        elif adjusted_value >= neutral_low:
            return 0.0  # 중립
        elif adjusted_value >= weak_con:
            return -0.2
        elif adjusted_value >= strong_con:
            return -0.5
        else:
            return -1.0
    
    def get_required_features(self, edge: Dict[str, Any]) -> List[str]:
        """특정 Edge에 필요한 Feature 목록"""
        spec = self.spec_registry.find_matching_spec(
            edge.get("head_id", ""),
            edge.get("head_name", ""),
            edge.get("head_type", ""),
            edge.get("tail_id", ""),
            edge.get("tail_name", ""),
            edge.get("tail_type", ""),
            edge.get("relation_type", ""),
            edge.get("polarity", ""),
        )
        
        if not spec:
            return []
        
        return [ef.get("feature", "") for ef in spec.evidence_features]
