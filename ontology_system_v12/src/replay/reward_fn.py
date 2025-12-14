"""
Reward 계산 함수.
"""
from typing import Dict, Tuple


def compute_reward(metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    기본 보상:
    reward = +1 * direction_hit - 0.5 * overconfidence_penalty - 0.2 * risk_penalty
    """
    if not metrics:
        return 0.0, {}
    reward = (
        1.0 * metrics.get("direction_hit", 0.0)
        - 0.5 * metrics.get("overconfidence_penalty", 0.0)
        - 0.2 * metrics.get("risk_penalty", 0.0)
    )
    breakdown = {
        "direction_hit": metrics.get("direction_hit", 0.0),
        "overconfidence_penalty": metrics.get("overconfidence_penalty", 0.0),
        "risk_penalty": metrics.get("risk_penalty", 0.0),
    }
    return reward, breakdown
