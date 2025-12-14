"""
Layer 2: Trace & Verification (Part 1)
학습을 가능하게 만드는 핵심 모듈들
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import json

from src.scenario_inference.contracts import (
    ScenarioState, ScenarioTrace, DecisionRecord, DecisionType,
    VerificationReport, VerificationTarget, VerificationOutcome,
    RewardBreakdown,
)

logger = logging.getLogger(__name__)


class ScenarioTraceRecorder:
    """시나리오 트레이스 레코더 - 추론 중 모든 결정을 저장"""
    
    def __init__(self, db_path: str = "data/scenario_traces.db"):
        self.db_path = db_path
        self._current_trace: Optional[ScenarioTrace] = None
        self._ensure_db()
    
    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY, as_of_time TEXT NOT NULL,
                    query_id TEXT, state_hash TEXT, decision_records TEXT,
                    evidence_trace TEXT, final_scenarios TEXT,
                    failure_flags TEXT, total_cost REAL, created_at TEXT
                )
            """)
            conn.commit()
    
    def start_trace(self, state: ScenarioState, pipeline_version: str = "v1.0") -> ScenarioTrace:
        trace = ScenarioTrace(
            as_of_time=state.as_of_time, query_id=state.query_id,
            state_hash=state.compute_hash(), kg_snapshot_ref=state.kg_snapshot_ref,
            pipeline_version=pipeline_version,
        )
        self._current_trace = trace
        return trace
    
    def record_decision(self, record: DecisionRecord) -> None:
        if self._current_trace:
            self._current_trace.add_decision(record)
    
    def finalize_trace(self, total_cost: float = 0.0) -> ScenarioTrace:
        if not self._current_trace:
            raise ValueError("No active trace")
        self._current_trace.total_cost = total_cost
        self._save_trace(self._current_trace)
        trace = self._current_trace
        self._current_trace = None
        return trace
    
    def _save_trace(self, trace: ScenarioTrace) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                trace.trace_id, trace.as_of_time.isoformat(), trace.query_id,
                trace.state_hash,
                json.dumps([r.model_dump() for r in trace.decision_records], default=str),
                json.dumps(trace.evidence_trace, default=str),
                json.dumps(trace.final_scenarios, default=str),
                json.dumps(trace.failure_flags), trace.total_cost,
                trace.created_at.isoformat(),
            ))
            conn.commit()
    
    def get_trace(self, trace_id: str) -> Optional[ScenarioTrace]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM traces WHERE trace_id=?", (trace_id,)).fetchone()
        if not row:
            return None
        return ScenarioTrace(
            trace_id=row['trace_id'],
            as_of_time=datetime.fromisoformat(row['as_of_time']),
            query_id=row['query_id'] or "", state_hash=row['state_hash'] or "",
            decision_records=[DecisionRecord(**r) for r in json.loads(row['decision_records'] or "[]")],
            failure_flags=json.loads(row['failure_flags'] or "[]"),
            total_cost=row['total_cost'] or 0.0,
            created_at=datetime.fromisoformat(row['created_at']),
        )


@dataclass
class VerificationJob:
    job_id: str
    trace_id: str
    scenario_id: str
    scheduled_at: datetime
    targets: List[VerificationTarget]
    status: str = "pending"


class DelayedVerifier:
    """지연 검증기 - 미래/과거 시나리오 검증"""
    
    def __init__(self, db_path: str = "data/verifications.db", default_horizon_days: int = 30):
        self.db_path = db_path
        self.default_horizon_days = default_horizon_days
        self._ensure_db()
    
    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verification_reports (
                    report_id TEXT PRIMARY KEY, scenario_id TEXT, trace_id TEXT,
                    scenario_as_of TEXT, accuracy REAL, overall_outcome TEXT,
                    outcome_values TEXT, direction_hits TEXT, verified_at TEXT
                )
            """)
            conn.commit()
    
    def verify_now(self, trace: ScenarioTrace, scenario_id: str,
                   targets: List[VerificationTarget],
                   outcome_fetcher=None) -> VerificationReport:
        outcome_values, direction_hits = {}, {}
        for t in targets:
            if outcome_fetcher:
                val = outcome_fetcher(t.entity_id, trace.as_of_time + timedelta(days=t.window_days))
                if val is not None:
                    outcome_values[t.target_id] = val
                    if t.expected_direction == "+":
                        direction_hits[t.target_id] = val > 0
                    elif t.expected_direction == "-":
                        direction_hits[t.target_id] = val < 0
                    else:
                        direction_hits[t.target_id] = abs(val) < 0.02
        
        accuracy = sum(direction_hits.values()) / max(1, len(direction_hits))
        outcome = VerificationOutcome.PASS_DIRECTION if accuracy >= 0.6 else VerificationOutcome.FAIL_DIRECTION
        
        report = VerificationReport(
            scenario_id=scenario_id, trace_id=trace.trace_id,
            scenario_as_of=trace.as_of_time, targets=targets,
            outcome_values=outcome_values, direction_hits=direction_hits,
            overall_outcome=outcome, accuracy=accuracy,
        )
        self._save_report(report)
        return report
    
    def _save_report(self, report: VerificationReport) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO verification_reports VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                report.report_id, report.scenario_id, report.trace_id,
                report.scenario_as_of.isoformat() if report.scenario_as_of else None,
                report.accuracy, report.overall_outcome.value,
                json.dumps(report.outcome_values), json.dumps(report.direction_hits),
                report.verified_at.isoformat(),
            ))
            conn.commit()
    
    def get_pending_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        검증 대기 중인 Trace 목록 조회
        
        현재는 stub 구현 - 실제로는 `verification_jobs` 테이블을 조회해야 함
        """
        # Stub - 실제로는 pending 상태의 검증 작업 조회
        return []

