"""
Layer 2: Trace & Verification (Part 2)
Scorecard, ScenarioDatasetBuilder
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import json

from src.scenario_inference.contracts import (
    ScenarioTrace, DecisionType, VerificationReport, RewardBreakdown,
)

logger = logging.getLogger(__name__)


class Scorecard:
    """스코어카드 - 추론 과정 품질 다면 채점"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "accuracy": 0.40, "calibration": 0.20, "robustness": 0.15,
            "cost_efficiency": 0.10, "constraint_penalty": 0.15,
        }
    
    def compute(self, trace: ScenarioTrace, 
                verification_report: Optional[VerificationReport],
                evidence_trace: Optional[Dict] = None) -> RewardBreakdown:
        breakdown = RewardBreakdown(
            trace_id=trace.trace_id,
            verification_report_id=verification_report.report_id if verification_report else None,
            weights=self.weights.copy(),
        )
        
        # Accuracy
        breakdown.accuracy = verification_report.accuracy if verification_report else 0.0
        
        # Calibration
        if verification_report and trace.confidence_breakdown:
            predicted = trace.confidence_breakdown.get("overall", 0.5)
            breakdown.calibration_error = abs(predicted - verification_report.accuracy)
            breakdown.calibration = max(0, 1 - breakdown.calibration_error * 2)
        else:
            breakdown.calibration = 0.5
        
        # Robustness
        if evidence_trace:
            scores = evidence_trace.get("robustness_scores", {})
            breakdown.robustness = sum(scores.values()) / max(1, len(scores)) if scores else 0.5
        else:
            breakdown.robustness = 0.5
        
        # Cost Efficiency
        breakdown.actual_cost = trace.total_cost
        breakdown.cost_efficiency = max(0, 1 - (trace.total_cost / 50.0))
        
        # Constraint Penalty
        breakdown.constraint_violations = trace.failure_flags.copy()
        breakdown.constraint_penalty = min(1.0, len(trace.failure_flags) * 0.15)
        
        breakdown.compute_scalar()
        return breakdown


@dataclass
class LearningDataset:
    dataset_id: str
    decision_type: DecisionType
    samples: List[Dict[str, Any]]
    created_at: datetime
    policy_version: str


class ScenarioDatasetBuilder:
    """시나리오 데이터셋 빌더 - 결정 단위별 학습 데이터 생성"""
    
    def __init__(self, db_path: str = "data/learning_datasets.db"):
        self.db_path = db_path
        self._ensure_db()
    
    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_samples (
                    sample_id TEXT PRIMARY KEY, dataset_id TEXT, decision_type TEXT,
                    trace_id TEXT, candidates TEXT, chosen_id TEXT, reward REAL,
                    reward_breakdown TEXT, policy_version TEXT, created_at TEXT
                )
            """)
            conn.commit()
    
    def build_from_traces(self, traces: List[ScenarioTrace],
                          reward_breakdowns: List[RewardBreakdown],
                          policy_version: str = "v1.0") -> Dict[DecisionType, LearningDataset]:
        datasets = {}
        reward_map = {rb.trace_id: rb for rb in reward_breakdowns}
        
        for dt in DecisionType:
            samples = []
            for trace in traces:
                reward = reward_map.get(trace.trace_id)
                if not reward:
                    continue
                for decision in trace.get_decisions_by_type(dt):
                    samples.append({
                        "trace_id": trace.trace_id,
                        "decision_id": decision.decision_id,
                        "candidates": [c.model_dump() for c in decision.candidates],
                        "chosen_id": decision.chosen_id,
                        "reward": reward.scalar_reward,
                        "reward_breakdown": reward.to_dict(),
                    })
            
            if samples:
                dataset = LearningDataset(
                    dataset_id=f"DS_{dt.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    decision_type=dt, samples=samples, created_at=datetime.now(),
                    policy_version=policy_version,
                )
                datasets[dt] = dataset
                self._save_dataset(dataset)
        
        return datasets
    
    def _save_dataset(self, dataset: LearningDataset) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for s in dataset.samples:
                conn.execute("""
                    INSERT INTO learning_samples VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (
                    s["decision_id"], dataset.dataset_id, dataset.decision_type.value,
                    s["trace_id"], json.dumps(s["candidates"], default=str),
                    s["chosen_id"], s["reward"], json.dumps(s["reward_breakdown"]),
                    dataset.policy_version, datetime.now().isoformat(),
                ))
            conn.commit()
    
    def get_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT decision_type, COUNT(*) FROM learning_samples GROUP BY decision_type
            """)
            by_type = {row[0]: row[1] for row in cursor.fetchall()}
        return {"by_decision_type": by_type, "total": sum(by_type.values())}
