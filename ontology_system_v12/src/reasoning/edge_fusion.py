"""
Edge Weight Fusion Engine (EES) v2.0
"Domain/Personal/Semantic/Validation + Evidence + Regime 정보를 모두 합쳐 최종 관계 강도를 계산"

가중치 공식 (v2.0):
  W_base = domain_conf * (1 - decay) * semantic_score * evidence_bonus * gold_bonus
  W_evidence = evidence_score * evidence_weight
  W_regime = regime_applicability (1.0 기본, 레짐에 따라 조정)
  
  W_D = W_base * W_evidence * W_regime
  W_P = PCS * personal_weight 
        (Domain 존재 시 * 0.3 감쇠)
  
  추가 반영 요소:
  - evidence_count 가중 (많을수록 bonus)
  - gold_flag 보너스 (Gold Set 검증된 관계)
  - evidence_score (실데이터 기반 검증 점수)
  - regime_applicability (현재 레짐에서의 관계 유효성)
  
  최종: W = W_D + W_P
  (Domain-Personal sign 충돌 시 -> Personal 무시)
"""
import logging
from typing import List, Dict, Optional

from src.reasoning.models import RetrievedPath, FusedEdge, FusedPath

logger = logging.getLogger(__name__)


# Semantic score 매핑
SEMANTIC_SCORES = {
    "sem_confident": 1.0,
    "sem_weak": 0.7,
    "sem_ambiguous": 0.4,
    "sem_spurious": 0.2,
    "sem_wrong": 0.1,
}


class EdgeWeightFusion:
    """
    Edge Weight Fusion Engine (EES) v2.0
    Domain + Personal + Evidence + Regime 가중치 융합
    
    공식 (v2.0):
      W_base = domain_conf * (1 - decay) * semantic * evidence_bonus * gold_bonus
      W_evidence = evidence_score (0~1, 없으면 1.0)
      W_regime = regime_applicability (0~2, 기본 1.0)
      
      W_D = W_base * W_evidence * W_regime
      W_P = PCS * personal_weight * personal_discount
      W = W_D + W_P (충돌 시 Domain만)
    """
    
    def __init__(
        self,
        domain_priority: float = 0.7,
        personal_discount: float = 0.3,
        decay_weight: float = 0.1,
        evidence_bonus_rate: float = 0.02,
        gold_bonus: float = 1.2,
        # v2.0 신규 파라미터
        evidence_weight: float = 0.5,          # Evidence 영향력 (0~1)
        min_evidence_score: float = 0.1,       # Evidence 최소값 (0이면 무효화 방지)
        regime_sensitivity: float = 1.0,       # Regime 민감도 (높을수록 레짐 영향↑)
    ):
        self.domain_priority = domain_priority
        self.personal_discount = personal_discount
        self.decay_weight = decay_weight
        self.evidence_bonus_rate = evidence_bonus_rate
        self.gold_bonus = gold_bonus
        # v2.0
        self.evidence_weight = evidence_weight
        self.min_evidence_score = min_evidence_score
        self.regime_sensitivity = regime_sensitivity

    # ------------------------------------------------------------------ #
    # Policy injection helpers
    # ------------------------------------------------------------------ #
    def set_weights(
        self,
        domain_weight: float = None,
        personal_weight: float = None,
        semantic_penalty: float = None,
        decay_lambda: float = None,
    ) -> None:
        """
        정책 파라미터를 런타임에 주입.
        전달되지 않은 값은 기존 설정을 유지한다.
        """
        if domain_weight is not None:
            self.domain_priority = float(domain_weight)
        if personal_weight is not None:
            self.personal_discount = float(personal_weight)
        if decay_lambda is not None:
            self.decay_weight = float(decay_lambda)
        if semantic_penalty is not None:
            # semantic_penalty는 semantic_score를 조정하는 방식으로 사용
            self._semantic_penalty = float(semantic_penalty)
        else:
            self._semantic_penalty = getattr(self, "_semantic_penalty", 0.0)
    
    def fuse_path(
        self, 
        path: RetrievedPath,
        regime_applicability: Optional[float] = None,
    ) -> FusedPath:
        """
        경로의 모든 엣지 가중치 융합
        
        Args:
            path: 검색된 경로
            regime_applicability: 현재 레짐에서의 관계 적용 강도 (None이면 1.0)
        """
        fused_edges = []
        
        for edge in path.edges:
            fused = self._fuse_edge(edge, regime_applicability)
            fused_edges.append(fused)
        
        path_weight, path_sign = self._calculate_path_metrics(fused_edges)
        
        return FusedPath(
            path_id=path.path_id,
            nodes=path.nodes,
            fused_edges=fused_edges,
            path_weight=path_weight,
            path_sign=path_sign,
        )
    
    def _fuse_edge(
        self, 
        edge: Dict,
        regime_applicability: Optional[float] = None,
    ) -> FusedEdge:
        """
        단일 엣지 가중치 융합 (v2.0)
        
        W_base = domain_conf * (1 - decay) * semantic * evidence_bonus * gold_bonus
        W_evidence = evidence_score (조정됨)
        W_regime = regime_applicability
        W_D = W_base * W_evidence * W_regime
        W_P = PCS * personal_weight * discount
        """
        source = edge.get("source", "domain")
        
        domain_weight = 0.0
        domain_conf = 0.0
        decay_factor = 0.0
        semantic_score = 1.0
        evidence_bonus = 1.0
        gold_applied = False
        
        # v2.0: Evidence 및 Regime 관련
        evidence_score = edge.get("evidence_score", 1.0)  # 기본값 1.0 (미적용)
        evidence_trace = edge.get("evidence_trace", [])
        regime_app = regime_applicability if regime_applicability is not None else edge.get("regime_applicability", 1.0)
        
        if source == "domain":
            domain_conf = edge.get("domain_conf", 0.5)
            decay_factor = edge.get("decay_factor", 0.0)
            semantic_tag = edge.get("semantic_tag", "sem_confident")
            semantic_score = SEMANTIC_SCORES.get(semantic_tag, 0.7)
            if getattr(self, "_semantic_penalty", 0.0):
                semantic_score = max(
                    0.0, semantic_score - float(self._semantic_penalty)
                )
            
            # Evidence count 보너스 (체감)
            evidence_count = edge.get("evidence_count", 1)
            evidence_bonus = 1.0 + min(0.2, self.evidence_bonus_rate * evidence_count)
            
            # Gold flag 보너스
            gold_flag = edge.get("gold_flag", False)
            if gold_flag:
                gold_applied = True
                evidence_bonus *= self.gold_bonus
            
            # 기본 가중치
            base_weight = domain_conf * (1 - decay_factor) * semantic_score * evidence_bonus
            
            # v2.0: Evidence Score 반영
            # evidence_score가 1.0이 아닌 경우에만 적용
            if evidence_score != 1.0:
                # evidence_score를 [min, 1+bonus] 범위로 조정
                adjusted_evidence = max(
                    self.min_evidence_score,
                    self.evidence_weight * evidence_score + (1 - self.evidence_weight)
                )
                base_weight *= adjusted_evidence
            
            # v2.0: Regime Applicability 반영
            if regime_app != 1.0:
                # regime_sensitivity로 효과 조절
                regime_effect = 1.0 + (regime_app - 1.0) * self.regime_sensitivity
                regime_effect = max(0.1, min(2.0, regime_effect))  # 범위 제한
                base_weight *= regime_effect
            
            domain_weight = base_weight
        
        personal_weight = 0.0
        pcs_score = 0.0
        has_conflict = False
        
        if source == "personal":
            pcs_score = edge.get("pcs_score", 0.5)
            p_weight = edge.get("personal_weight", 0.5)
            
            personal_weight = pcs_score * p_weight
            
            if domain_weight > 0:
                personal_weight *= self.personal_discount
                has_conflict = True
        
        final_weight = domain_weight + personal_weight
        
        if has_conflict and domain_weight > 0:
            final_weight = domain_weight
        
        return FusedEdge(
            edge_id=edge.get("relation_id", ""),
            head_id=edge.get("head", ""),
            tail_id=edge.get("tail", ""),
            relation_type=edge.get("relation_type", "Affect"),
            sign=edge.get("sign", "+"),
            domain_weight=domain_weight,
            personal_weight=personal_weight,
            final_weight=final_weight,
            domain_conf=domain_conf,
            decay_factor=decay_factor,
            semantic_score=semantic_score,
            pcs_score=pcs_score,
            has_personal_conflict=has_conflict,
        )
    
    def _calculate_path_metrics(
        self,
        fused_edges: List[FusedEdge],
    ) -> tuple:
        """경로 전체 가중치와 sign 계산"""
        if not fused_edges:
            return 0.0, "+"
        
        path_weight = 1.0
        for edge in fused_edges:
            path_weight *= max(edge.final_weight, 0.01)
        
        path_sign = "+"
        for edge in fused_edges:
            if edge.sign == "-":
                path_sign = "-" if path_sign == "+" else "+"
        
        return path_weight, path_sign
    
    def fuse_multiple_paths(
        self,
        paths: List[RetrievedPath],
        regime_applicability: Optional[float] = None,
    ) -> List[FusedPath]:
        """여러 경로 융합"""
        return [self.fuse_path(p, regime_applicability) for p in paths]
    
    def fuse_path_with_evidence(
        self,
        path: RetrievedPath,
        edge_evidence_scores: Dict[str, float],
        regime_applicability: float = 1.0,
    ) -> FusedPath:
        """
        Evidence Score를 명시적으로 주입하여 경로 융합
        
        Args:
            path: 검색된 경로
            edge_evidence_scores: {edge_id: evidence_score} 매핑
            regime_applicability: 레짐 적용 강도
        
        Returns:
            FusedPath
        """
        fused_edges = []
        
        for edge in path.edges:
            # Edge에 evidence_score 주입
            edge_id = edge.get("relation_id", "")
            if edge_id in edge_evidence_scores:
                edge["evidence_score"] = edge_evidence_scores[edge_id]
            
            fused = self._fuse_edge(edge, regime_applicability)
            fused_edges.append(fused)
        
        path_weight, path_sign = self._calculate_path_metrics(fused_edges)
        
        return FusedPath(
            path_id=path.path_id,
            nodes=path.nodes,
            fused_edges=fused_edges,
            path_weight=path_weight,
            path_sign=path_sign,
        )
