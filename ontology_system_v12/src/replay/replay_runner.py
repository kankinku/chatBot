"""
Replay Runner
과거 기간을 순회하며 스냅샷 기반 추론을 실행하고, 실제 결과와 비교하여 메트릭/보상을 계산한다.
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from src.replay.snapshot_manager import SnapshotManager, SystemSnapshot
from src.replay.contracts import ReplayContext
from src.replay import reasoning_engine_adapter
from src.replay import metrics as replay_metrics
from src.replay import reward_fn
from src.policy.policy_registry import get_policy_registry
from src.replay.outcome_fetcher import Outcome

logger = logging.getLogger(__name__)


class ReplayMode(str, Enum):
    POINT_IN_TIME = "point_in_time"
    ROLLING = "rolling"
    FULL_PERIOD = "full_period"


@dataclass
class ReplayStep:
    step_date: datetime
    snapshot: Optional[SystemSnapshot]
    conclusion: Optional[Any] = None
    actual_outcome: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayResult:
    replay_id: str
    mode: ReplayMode
    start_date: datetime
    end_date: datetime
    steps: List[ReplayStep] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


from src.learning.policy_optimizer import PolicyOptimizer

class ReplayRunner:
    """
    스냅샷 기반 리플레이 실행기.
    """

    def __init__(
        self,
        snapshot_manager: SnapshotManager,
        reasoning_engine: Optional[Any] = None,
        outcome_fetcher: Optional[Callable[[datetime], Outcome]] = None,
        optimizer: Optional[PolicyOptimizer] = None,
    ):
        self.snapshot_manager = snapshot_manager
        self.reasoning_engine = reasoning_engine
        self.outcome_fetcher = outcome_fetcher
        self.optimizer = optimizer

    def run_point_in_time(
        self,
        target_date: datetime,
        query: Optional[str] = None,
        policy_id: Optional[str] = None,
        seed: int = 0,
        lookahead_days: int = 30,
    ) -> ReplayStep:
        """
        특정 시점(as_of)에서 스냅샷을 사용해 추론을 수행.
        """
        snapshot = self.snapshot_manager.get_snapshot_at(target_date)
        if not snapshot:
            snapshot = self.snapshot_manager.create_snapshot(target_date)

        registry = get_policy_registry()
        policy = registry.get_policy(policy_id) if policy_id else registry.get_active_policy()

        conclusion = None
        trace: Dict[str, Any] = {}
        if self.reasoning_engine and query:
            try:
                ctx = ReplayContext(
                    as_of=target_date,
                    snapshot_id=snapshot.snapshot_id,
                    policy_id=policy.policy_id,
                    seed=seed,
                    mode="REPLAY_SNAPSHOT_ONLY",
                )
                conclusion, trace = reasoning_engine_adapter.run_query(
                    query=query,
                    ctx=ctx,
                    reasoning_pipeline=self.reasoning_engine,
                    snapshot=snapshot,
                )
            except Exception as e:
                logger.error(f"Reasoning failed: {e}")

        actual_outcome = None
        step_metrics: Dict[str, float] = {}
        if self.outcome_fetcher:
            try:
                actual_outcome = self.outcome_fetcher(target_date + timedelta(days=lookahead_days))
                if conclusion and actual_outcome:
                    step_metrics = replay_metrics.compute_metrics(conclusion, actual_outcome)
                    logger.info(f"[ReplayRunner] Metrics computed for {target_date}: {step_metrics}")

            except Exception as e:
                logger.error(f"Outcome fetch failed: {e}")

        if conclusion and step_metrics:
            reward, breakdown = reward_fn.compute_reward(step_metrics)
            trace.update({"reward": reward, "reward_breakdown": breakdown})
            step_metrics["reward"] = reward
            logger.info(f"[ReplayRunner] Reward computed: {reward}, breakdown={breakdown}")

        return ReplayStep(
            step_date=target_date,
            snapshot=snapshot,
            conclusion=conclusion,
            actual_outcome=actual_outcome,
            metrics=step_metrics,
            trace=trace,
        )

    def run_rolling(
        self,
        start_date: datetime,
        end_date: datetime,
        step_days: int = 1,
        query: Optional[str] = None,
        policy_id: Optional[str] = None,
        lookahead_days: int = 30,
    ) -> ReplayResult:
        begin = time.time()
        replay_id = f"RPL_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        steps: List[ReplayStep] = []

        current_date = start_date
        while current_date <= end_date:
            step = self.run_point_in_time(
                current_date,
                query=query,
                policy_id=policy_id, # Use current (potentially updated) policy
                lookahead_days=lookahead_days,
            )
            steps.append(step)
            
            # Learning Loop
            if self.optimizer and step.metrics and "reward" in step.metrics:
                # Retrieve current policy object
                registry = get_policy_registry()
                current_policy = registry.get_policy(policy_id) or registry.get_active_policy()
                
                # Update
                new_policy = self.optimizer.update(current_policy, step.metrics["reward"], step.metrics)
                
                # Activate new version for next step
                registry.activate_policy(new_policy.policy_id)
                policy_id = new_policy.policy_id # Ensure next iteration uses it explicitly if passed
                
            current_date += timedelta(days=step_days)

        elapsed = time.time() - begin
        aggregate_metrics = self._calculate_aggregate_metrics(steps)

        return ReplayResult(
            replay_id=replay_id,
            mode=ReplayMode.ROLLING,
            start_date=start_date,
            end_date=end_date,
            steps=steps,
            aggregate_metrics=aggregate_metrics,
            elapsed_seconds=elapsed,
        )

    def run_full_period(
        self,
        start_date: datetime,
        end_date: datetime,
        queries: List[str],
        step_days: int = 5,
        policy_id: Optional[str] = None,
        lookahead_days: int = 30,
    ) -> ReplayResult:
        begin = time.time()
        replay_id = f"FULL_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        steps: List[ReplayStep] = []

        current_date = start_date
        while current_date <= end_date:
            for q in queries:
                step = self.run_point_in_time(
                    current_date,
                    query=q,
                    policy_id=policy_id,
                    lookahead_days=lookahead_days,
                )
                steps.append(step)
            current_date += timedelta(days=step_days)

        elapsed = time.time() - begin
        aggregate_metrics = self._calculate_aggregate_metrics(steps)

        return ReplayResult(
            replay_id=replay_id,
            mode=ReplayMode.FULL_PERIOD,
            start_date=start_date,
            end_date=end_date,
            steps=steps,
            aggregate_metrics=aggregate_metrics,
            elapsed_seconds=elapsed,
        )

    def _calculate_aggregate_metrics(self, steps: List[ReplayStep]) -> Dict[str, float]:
        if not steps:
            return {}
        total_steps = len(steps)
        success = sum(1 for s in steps if s.conclusion is not None)
        avg_reward = (
            sum(s.metrics.get("reward", 0.0) for s in steps if s.metrics) / max(success, 1)
        )
        return {
            "total_steps": total_steps,
            "success_rate": success / total_steps if total_steps else 0.0,
            "avg_reward": avg_reward,
        }

    def compare_with_live(
        self,
        replay_result: ReplayResult,
        live_conclusions: Dict[datetime, Dict],
    ) -> Dict[str, Any]:
        matches = 0
        mismatches = 0
        differences = []

        for step in replay_result.steps:
            live_conclusion = live_conclusions.get(step.step_date)

            if live_conclusion and step.conclusion:
                live_dir = live_conclusion.get("direction")
                replay_dir = getattr(step.conclusion, "direction", None)
                replay_dir_value = (
                    replay_dir.value if hasattr(replay_dir, "value") else replay_dir
                )

                if live_dir == replay_dir_value:
                    matches += 1
                else:
                    mismatches += 1
                    differences.append({"date": step.step_date, "live": live_dir, "replay": replay_dir_value})

        return {
            "matches": matches,
            "mismatches": mismatches,
            "differences": differences,
        }
