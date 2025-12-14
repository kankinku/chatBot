"""
Minimal orchestrator to connect replay rewards to policy updates.
"""
from typing import Any, Dict, Optional

from src.policy.policy_registry import get_policy_registry
from src.learning.policy_learner import PolicyWeightLearner
from src.replay.replay_runner import ReplayRunner


class ReplayTrainingOrchestrator:
    """
    Runs replay, collects rewards, and updates active policy.
    """

    def __init__(
        self,
        replay_runner: ReplayRunner,
        policy_learner: Optional[PolicyWeightLearner] = None,
    ):
        self.replay_runner = replay_runner
        self.policy_learner = policy_learner or PolicyWeightLearner()
        self.registry = get_policy_registry()

    def run(
        self,
        start_date,
        end_date,
        query: str,
        lookahead_days: int = 30,
    ) -> Dict[str, Any]:
        result = self.replay_runner.run_rolling(
            start_date=start_date,
            end_date=end_date,
            step_days=1,
            query=query,
            lookahead_days=lookahead_days,
        )
        # Reward 집계 (평균 reward 기반 간단 업데이트)
        avg_reward = result.aggregate_metrics.get("avg_reward", 0.0)
        delta = 0.05 if avg_reward >= 0 else -0.05
        new_policy = self.policy_learner.create_policy_variant(
            base_version=self.policy_learner.get_active_policy().version,
            ees_adj={"domain_weight": delta, "personal_weight": -delta},
        )
        new_policy_id = self.registry.save_policy_variant(
            parent_id=new_policy.version,
            params={
                "fusion": {
                    "domain_weight": new_policy.ees_weights.get("domain_weight", 0.7),
                    "personal_weight": new_policy.ees_weights.get("personal_weight", 0.3),
                }
            },
            metrics={"avg_reward": avg_reward},
        )
        self.registry.set_active_policy(new_policy_id)
        return {"avg_reward": avg_reward, "active_policy": new_policy_id}
