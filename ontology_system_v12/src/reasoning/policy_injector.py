"""
Policy injector: applies PolicyBundle parameters to reasoning components.
"""
import logging
from typing import Optional

from src.policy.contracts import PolicyBundle

logger = logging.getLogger(__name__)


def apply(
    policy: PolicyBundle,
    *,
    edge_fusion: Optional[object] = None,
    graph_retrieval: Optional[object] = None,
    confidence_filter: Optional[object] = None,
) -> None:
    """
    주어진 PolicyBundle을 각 컴포넌트에 주입한다.
    존재하지 않는 속성에 대해서는 조용히 무시한다.
    """
    if not policy:
        logger.warning("PolicyInjector: no policy provided")
        return

    fusion_params = policy.get_param("fusion", {}) or {}
    routing_params = policy.get_param("routing", {}) or {}
    retrieval_params = policy.get_param("retrieval", {}) or {}

    if edge_fusion and hasattr(edge_fusion, "set_weights"):
        edge_fusion.set_weights(
            domain_weight=fusion_params.get("domain_weight"),
            personal_weight=fusion_params.get("personal_weight"),
            semantic_penalty=fusion_params.get("semantic_penalty"),
            decay_lambda=fusion_params.get("decay_lambda"),
        )
        logger.debug("PolicyInjector: applied fusion weights")

    if graph_retrieval and hasattr(graph_retrieval, "set_limits"):
        graph_retrieval.set_limits(
            max_depth=retrieval_params.get("max_depth"),
            max_paths=retrieval_params.get("max_paths"),
        )

    if confidence_filter and hasattr(confidence_filter, "set_thresholds"):
        confidence_filter.set_thresholds(
            domain_threshold=routing_params.get("domain_threshold"),
            personal_threshold=routing_params.get("personal_threshold"),
        )
