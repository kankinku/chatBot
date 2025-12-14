"""
Evidence Store with as_of validation.
"""
import logging
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from src.common.asof_context import validate_access
from src.shared.schemas import EvidenceScore, AccumulatedEvidence, RegimeType

logger = logging.getLogger(__name__)


class EvidenceStore:
    def __init__(self, db_path: str = "data/evidence.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_scores (
                    edge_id TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    pro_score REAL,
                    con_score REAL,
                    total_score REAL,
                    regime TEXT,
                    regime_confidence REAL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    PRIMARY KEY (edge_id, as_of)
                )
            """)
            conn.commit()

    def save_score(self, score: EvidenceScore) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO evidence_scores (
                    edge_id, as_of, pro_score, con_score, total_score,
                    regime, regime_confidence, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                score.edge_id,
                score.as_of.isoformat(),
                score.pro_score,
                score.con_score,
                score.total_score,
                score.regime.value if score.regime else None,
                score.regime_confidence,
                datetime.now().isoformat(),
                json.dumps(score.metadata or {}),
            ))
            conn.commit()

    def get_scores(
        self,
        edge_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
    ) -> List[EvidenceScore]:
        validate_access(as_of)
        query = "SELECT * FROM evidence_scores WHERE edge_id = ?"
        params = [edge_id]
        if start:
            query += " AND as_of >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND as_of <= ?"
            params.append(end.isoformat())
        if as_of:
            query += " AND as_of <= ?"
            params.append(as_of.isoformat())
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_score(r) for r in rows]

    def get_latest_score(self, edge_id: str, as_of: Optional[datetime] = None) -> Optional[EvidenceScore]:
        validate_access(as_of)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if as_of:
                cursor = conn.execute("""
                    SELECT * FROM evidence_scores
                    WHERE edge_id = ? AND as_of <= ?
                    ORDER BY as_of DESC
                    LIMIT 1
                """, (edge_id, as_of.isoformat()))
            else:
                cursor = conn.execute("""
                    SELECT * FROM evidence_scores
                    WHERE edge_id = ?
                    ORDER BY as_of DESC
                    LIMIT 1
                """, (edge_id,))
            row = cursor.fetchone()
        return self._row_to_score(row) if row else None

    def get_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) as cnt, COUNT(DISTINCT edge_id) as edges FROM evidence_scores")
            row = cursor.fetchone()
        return {"total_scores": row["cnt"], "edges": row["edges"]}

    def _row_to_score(self, row) -> EvidenceScore:
        return EvidenceScore(
            edge_id=row["edge_id"],
            as_of=datetime.fromisoformat(row["as_of"]),
            pro_score=row["pro_score"],
            con_score=row["con_score"],
            total_score=row["total_score"],
            regime=RegimeType(row["regime"]) if row["regime"] else None,
            regime_confidence=row["regime_confidence"],
            metadata=json.loads(row["metadata"] or "{}"),
        )
