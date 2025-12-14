"""
Replay → Reasoning 파이프라인 연결 어댑터.
"""
import logging
import random
from typing import Tuple, Any

from src.replay.contracts import ReplayContext
from src.reasoning.pipeline import ReasoningPipeline
from src.reasoning.models import ReasoningConclusion
from src.policy.policy_registry import get_policy_registry
from src.reasoning import policy_injector
from src.replay.system_snapshot import SystemSnapshot
from src.common.asof_context import asof_guard

logger = logging.getLogger(__name__)


def run_query(
    query: str,
    ctx: ReplayContext,
    reasoning_pipeline: ReasoningPipeline,
    snapshot: SystemSnapshot,
) -> Tuple[ReasoningConclusion, dict]:
    """
    리플레이 컨텍스트를 적용해 추론을 실행한다.

    - 정책: ctx.policy_id 로드 후 주입
    - 시드: 동일 실행 재현을 위해 random seed 설정
    - 스냅샷: caller가 snapshot 기반 repo를 주입했다고 가정
    """
    random.seed(ctx.seed)

    policy = get_policy_registry().get_policy(ctx.policy_id) or get_policy_registry().get_active_policy()
    policy_injector.apply(
        policy,
        edge_fusion=reasoning_pipeline.edge_fusion,
        graph_retrieval=reasoning_pipeline.graph_retrieval,
        confidence_filter=getattr(reasoning_pipeline, "confidence_filter", None),
    )

    # 스냅샷 기반 도메인 뷰를 주입하여 미래 데이터 접근을 차단
    if hasattr(snapshot, "get_graph_view"):
        reasoning_pipeline.graph_retrieval.domain = snapshot.get_graph_view()

    logger.info(
        f"[ReplayAdapter] query='{query}', as_of={ctx.as_of}, "
        f"snapshot={ctx.snapshot_id}, policy={policy.policy_id}"
    )

    with asof_guard(ctx.as_of, mode=ctx.mode):
        conclusion = reasoning_pipeline.reason(
            query,
            as_of=ctx.as_of,
            context={"snapshot": snapshot}
        )
    trace = {
        "policy_id": policy.policy_id,
        "policy_version": policy.version,
        "snapshot_id": ctx.snapshot_id,
        "as_of": ctx.as_of.isoformat(),
        "seed": ctx.seed,
        "conclusion_direction": conclusion.direction.value if hasattr(conclusion, "direction") else None,
        "confidence": getattr(conclusion, "confidence", None),
    }
    return conclusion, trace
