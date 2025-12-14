import logging
import random
from typing import Dict, Any, List

from src.policy.contracts import PolicyBundle
from src.policy.policy_registry import get_policy_registry

logger = logging.getLogger(__name__)

class PolicyOptimizer:
    """
    Optimizes policy parameters based on Replay rewards.
    Implements a simple Hill Climbing or Gradient-free update for MVP.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.history: List[Dict] = []

    def update(self, policy: PolicyBundle, reward: float, metrics: Dict[str, float]) -> PolicyBundle:
        """
        Updates the policy and returns a NEW policy instance (does not mutate in place).
        """
        # 1. Log history
        self.history.append({
            "policy_id": policy.policy_id,
            "reward": reward,
            "metrics": metrics
        })
        
        # 2. Compute update (Pseudo-logic for MVP)
        # If reward is high (>0.7), we reinforce the current config.
        # If reward is low (<0.3), we explore (perturb).
        
        new_params = policy.params.copy()
        
        if reward > 0.7:
            # Exploitation: Small jitter to refine
            factor = 1.0 + (random.random() - 0.5) * self.learning_rate
            
            # Access fusion weights inside params
            fusion_config = new_params.get("fusion", {}).copy()
            if "domain_weight" in fusion_config:
                 fusion_config["domain_weight"] = min(1.0, max(0.0, fusion_config["domain_weight"] * factor))
            new_params["fusion"] = fusion_config
            
        elif reward < 0.3:
            # Exploration: Larger perturbation
            pass
            
        # Create new version
        new_version_num = policy.version + 1 # version is int
        new_version = new_version_num
        
        updated_policy = PolicyBundle(
            policy_id=policy.policy_id,
            version=new_version,
            params=new_params,
            created_at=policy.created_at
        )
        
        logger.info(f"[Optimizer] Updated policy {policy.policy_id} to {new_version}. Reward={reward:.4f}")
        
        # 3. Save to registry (Persist)
        get_policy_registry().register_policy(updated_policy)
        get_policy_registry().activate_policy(policy.policy_id) # Make it active? Or wait for evaluation?
        # For "Closed Loop" MVP, we activate it immediately to show evolution.
        
        return updated_policy
