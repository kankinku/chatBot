"""
Evidence Accumulator
단발성 Evidence Score를 누적/평활화하여 Edge Confidence 업데이트 값 생성

책임:
- 점수 누적 (EMA/EWMA)
- 변동성 계산
- 신뢰도 업데이트 권장값 산출
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import math

from src.shared.schemas import EvidenceScore, AccumulatedEvidence

logger = logging.getLogger(__name__)


@dataclass
class AccumulationResult:
    """누적 결과"""
    edge_id: str
    accumulated: AccumulatedEvidence
    confidence_delta: float  # edge confidence에 적용할 변화량
    is_significant: bool     # 유의미한 변화인지


class EvidenceAccumulator:
    """
    Evidence 누적기
    
    단발성 Evidence Score를 누적/평활화하여
    Edge Confidence 업데이트 값을 생성합니다.
    
    알고리즘:
    - EMA (Exponential Moving Average)로 점수 평활화
    - 점수 변동성 추적
    - 일정 관측 수 이상이면 유의미한 신호로 판단
    """
    
    def __init__(
        self,
        alpha: float = 0.2,              # EMA 계수 (작을수록 과거 영향↑)
        min_observations: int = 3,        # 유의미 판단 최소 관측 수
        significance_threshold: float = 0.1,  # 유의미 변화 임계값
    ):
        """
        Args:
            alpha: EMA 계수 (0~1)
            min_observations: 유의미 판단 최소 관측 수
            significance_threshold: 유의미 변화 임계값
        """
        self.alpha = alpha
        self.min_observations = min_observations
        self.significance_threshold = significance_threshold
        
        # Edge별 누적 상태
        self._accumulated: Dict[str, AccumulatedEvidence] = {}
    
    def accumulate(
        self,
        score: EvidenceScore,
    ) -> AccumulationResult:
        """
        새 Evidence Score 누적
        
        Args:
            score: 새 Evidence Score
        
        Returns:
            AccumulationResult
        """
        edge_id = score.edge_id
        
        # 기존 누적값 조회 또는 초기화
        if edge_id not in self._accumulated:
            self._accumulated[edge_id] = AccumulatedEvidence(
                edge_id=edge_id,
                accumulated_score=score.total_score,
                score_volatility=0.0,
                observation_count=1,
                positive_count=1 if score.total_score > 0 else 0,
                negative_count=1 if score.total_score < 0 else 0,
                first_observed=score.timestamp,
                last_updated=score.timestamp,
            )
            
            return AccumulationResult(
                edge_id=edge_id,
                accumulated=self._accumulated[edge_id],
                confidence_delta=0.0,  # 첫 관측은 변화 없음
                is_significant=False,
            )
        
        acc = self._accumulated[edge_id]
        prev_score = acc.accumulated_score
        
        # EMA 업데이트
        new_score = self.alpha * score.total_score + (1 - self.alpha) * prev_score
        
        # 변동성 업데이트 (이동 표준편차)
        diff = score.total_score - prev_score
        new_vol = self.alpha * abs(diff) + (1 - self.alpha) * acc.score_volatility
        
        # 통계 업데이트
        acc.accumulated_score = new_score
        acc.score_volatility = new_vol
        acc.observation_count += 1
        if score.total_score > 0:
            acc.positive_count += 1
        elif score.total_score < 0:
            acc.negative_count += 1
        acc.last_updated = score.timestamp
        
        # Confidence delta 계산
        confidence_delta = self._calculate_confidence_delta(acc, new_score, prev_score)
        acc.suggested_confidence_delta = confidence_delta
        
        # 유의미 여부 판단
        is_significant = (
            acc.observation_count >= self.min_observations and
            abs(confidence_delta) >= self.significance_threshold
        )
        
        return AccumulationResult(
            edge_id=edge_id,
            accumulated=acc,
            confidence_delta=confidence_delta,
            is_significant=is_significant,
        )
    
    def accumulate_batch(
        self,
        scores: List[EvidenceScore],
    ) -> List[AccumulationResult]:
        """배치 누적"""
        return [self.accumulate(s) for s in scores]
    
    def _calculate_confidence_delta(
        self,
        acc: AccumulatedEvidence,
        new_score: float,
        prev_score: float,
    ) -> float:
        """
        Confidence 변화량 계산
        
        고려 요소:
        - 점수 변화량
        - 관측 수 (많을수록 확신↑)
        - 변동성 (낮을수록 확신↑)
        - 양성/음성 비율
        """
        # 기본 변화량 = 점수 변화
        base_delta = new_score - prev_score
        
        # 관측 수 가중치 (최대 1.5까지)
        obs_weight = min(1.5, 1.0 + (acc.observation_count / 20))
        
        # 변동성 패널티 (변동성 높으면 감소)
        vol_penalty = 1.0 / (1.0 + acc.score_volatility * 2)
        
        # 일관성 보너스 (한 방향으로 일관되면 가중)
        total = acc.positive_count + acc.negative_count
        if total > 0:
            pos_ratio = acc.positive_count / total
            consistency = max(pos_ratio, 1 - pos_ratio)  # 0.5 ~ 1.0
            consistency_bonus = 0.5 + consistency  # 1.0 ~ 1.5
        else:
            consistency_bonus = 1.0
        
        # 최종 delta
        confidence_delta = (
            base_delta *
            obs_weight *
            vol_penalty *
            consistency_bonus *
            0.1  # 스케일링 (실제 confidence 변화폭 제한)
        )
        
        # 범위 제한 (-0.3 ~ 0.3)
        return max(-0.3, min(0.3, confidence_delta))
    
    def get(self, edge_id: str) -> Optional[AccumulatedEvidence]:
        """누적 상태 조회"""
        return self._accumulated.get(edge_id)
    
    def get_all(self) -> List[AccumulatedEvidence]:
        """모든 누적 상태"""
        return list(self._accumulated.values())
    
    def get_significant_edges(
        self,
        min_delta: float = 0.1,
    ) -> List[AccumulatedEvidence]:
        """
        유의미한 변화가 있는 Edge 목록
        
        Args:
            min_delta: 최소 변화량
        """
        return [
            acc for acc in self._accumulated.values()
            if abs(acc.suggested_confidence_delta) >= min_delta
               and acc.observation_count >= self.min_observations
        ]
    
    def reset(self, edge_id: str) -> bool:
        """특정 Edge 누적 초기화"""
        if edge_id in self._accumulated:
            del self._accumulated[edge_id]
            return True
        return False
    
    def reset_all(self) -> int:
        """모든 누적 초기화"""
        count = len(self._accumulated)
        self._accumulated.clear()
        return count
    
    def load_from_store(self, accumulated_list: List[AccumulatedEvidence]) -> int:
        """저장소에서 누적 상태 로드"""
        for acc in accumulated_list:
            self._accumulated[acc.edge_id] = acc
        return len(accumulated_list)
    
    def get_stats(self) -> Dict:
        """통계"""
        if not self._accumulated:
            return {
                "total_edges": 0,
                "significant_count": 0,
                "avg_observations": 0,
                "avg_volatility": 0,
            }
        
        significant = sum(
            1 for acc in self._accumulated.values()
            if abs(acc.suggested_confidence_delta) >= self.significance_threshold
               and acc.observation_count >= self.min_observations
        )
        
        total_obs = sum(acc.observation_count for acc in self._accumulated.values())
        total_vol = sum(acc.score_volatility for acc in self._accumulated.values())
        count = len(self._accumulated)
        
        return {
            "total_edges": count,
            "significant_count": significant,
            "avg_observations": total_obs / count,
            "avg_volatility": total_vol / count,
        }
