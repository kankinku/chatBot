"""Edge weight fusion (domain-only)."""
import logging
from typing import List, Optional

from src.reasoning.models import RetrievedPath, FusedEdge, FusedPath

logger = logging.getLogger(__name__)

SEMANTIC_SCORES = {
    "sem_confident": 1.0,
    "sem_weak": 0.7,
    "sem_ambiguous": 0.4,
    "sem_spurious": 0.2,
    "sem_wrong": 0.1,
}


class EdgeWeightFusion:
    """Fuse edge weights using domain confidence and semantic tags."""

    def __init__(self, evidence_bonus_rate: float = 0.02):
        self.evidence_bonus_rate = evidence_bonus_rate
        self._semantic_penalty = 0.0

    def set_weights(self, semantic_penalty: Optional[float] = None) -> None:
        if semantic_penalty is not None:
            self._semantic_penalty = float(semantic_penalty)

    def fuse_path(self, path: RetrievedPath) -> FusedPath:
        fused_edges = [self._fuse_edge(edge) for edge in path.edges]
        path_weight, path_sign = self._calculate_path_metrics(fused_edges)
        return FusedPath(
            path_id=path.path_id,
            nodes=path.nodes,
            fused_edges=fused_edges,
            path_weight=path_weight,
            path_sign=path_sign,
        )

    def fuse_multiple_paths(self, paths: List[RetrievedPath]) -> List[FusedPath]:
        return [self.fuse_path(p) for p in paths]

    def _fuse_edge(self, edge: dict) -> FusedEdge:
        domain_conf = float(edge.get("domain_conf", 0.5))
        decay_factor = float(edge.get("decay_factor", 0.0))
        semantic_tag = edge.get("semantic_tag", "sem_confident")
        semantic_score = SEMANTIC_SCORES.get(semantic_tag, 0.7)
        if self._semantic_penalty:
            semantic_score = max(0.0, semantic_score - self._semantic_penalty)

        evidence_count = int(edge.get("evidence_count", 1))
        evidence_bonus = 1.0 + min(0.2, self.evidence_bonus_rate * evidence_count)

        final_weight = domain_conf * (1 - decay_factor) * semantic_score * evidence_bonus

        return FusedEdge(
            edge_id=edge.get("relation_id", ""),
            head_id=edge.get("head", ""),
            tail_id=edge.get("tail", ""),
            relation_type=edge.get("relation_type", "Affect"),
            sign=edge.get("sign", "+"),
            final_weight=final_weight,
            domain_conf=domain_conf,
            decay_factor=decay_factor,
            semantic_score=semantic_score,
            evidence_count=evidence_count,
        )

    def _calculate_path_metrics(self, fused_edges: List[FusedEdge]) -> tuple:
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
