"""
Replay metrics: 정량 평가 지표 모음.
"""
from typing import Dict

from src.replay.outcome_fetcher import Outcome
from src.reasoning.models import ReasoningConclusion


def compute_metrics(conclusion: ReasoningConclusion, outcome: Outcome) -> Dict[str, float]:
    """
    최소 3개 지표:
    - direction_hit: 예측 방향 일치 여부
    - overconfidence_penalty: 높은 confidence로 틀릴 때 패널티
    - risk_penalty: drawdown 발생 시 패널티
    """
    metrics: Dict[str, float] = {}
    if not conclusion or not outcome:
        return metrics

    pred_dir = getattr(conclusion.direction, "value", None) if hasattr(conclusion, "direction") else None
    conf = getattr(conclusion, "confidence", 0.0)

    # direction_hit
    hit = 0.0
    if pred_dir == "+" and outcome.direction_actual > 0:
        hit = 1.0
    elif pred_dir == "-" and outcome.direction_actual < 0:
        hit = 1.0
    elif pred_dir == "neutral" and outcome.direction_actual == 0:
        hit = 1.0
    metrics["direction_hit"] = hit

    # overconfidence_penalty
    wrong = 1.0 - hit
    metrics["overconfidence_penalty"] = wrong * conf

    # risk_penalty
    metrics["risk_penalty"] = abs(outcome.drawdown)

    return metrics
