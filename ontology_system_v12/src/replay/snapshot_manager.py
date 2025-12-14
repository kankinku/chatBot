"""
Snapshot Manager
as_of 기준 과거 스냅샷 생성 및 관리.
"""
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.replay.system_snapshot import SystemSnapshot

logger = logging.getLogger(__name__)


class SnapshotManager:
    """
    스냅샷 생성 및 조회를 담당.
    """

    def __init__(
        self,
        db_path: str = "data/snapshots.db",
        timeseries_repo: Optional[Any] = None,
        feature_store: Optional[Any] = None,
        evidence_store: Optional[Any] = None,
        regime_store: Optional[Any] = None,
        graph_repo: Optional[Any] = None,
    ):
        self.db_path = db_path
        self.timeseries_repo = timeseries_repo
        self.feature_store = feature_store
        self.evidence_store = evidence_store
        self.regime_store = regime_store
        self.graph_repo = graph_repo
        self._ensure_db()

    # ------------------------------------------------------------------ #
    # DB helpers
    # ------------------------------------------------------------------ #
    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    as_of TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    series_values TEXT,
                    feature_values TEXT,
                    evidence_scores TEXT,
                    regime_state TEXT,
                    edge_confidences TEXT,
                    graph_entities TEXT,
                    graph_relations TEXT,
                    metadata TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snap_as_of ON snapshots(as_of)")
            conn.commit()

    # ------------------------------------------------------------------ #
    def create_snapshot(
        self,
        as_of: datetime,
        series_ids: Optional[List[str]] = None,
        feature_ids: Optional[List[str]] = None,
        edge_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SystemSnapshot:
        snapshot_id = f"SNAP_{as_of.strftime('%Y%m%d_%H%M%S')}"

        series_values = self._fetch_series(series_ids, as_of)
        feature_values = self._fetch_features(feature_ids, as_of)
        evidence_scores = self._fetch_evidence(edge_ids)
        regime_state = self._fetch_regime(as_of)
        graph_entities, graph_relations = self._fetch_graph()

        snapshot = SystemSnapshot(
            snapshot_id=snapshot_id,
            as_of=as_of,
            created_at=datetime.utcnow(),
            series_values=series_values,
            feature_values=feature_values,
            evidence_scores=evidence_scores,
            regime_state=regime_state,
            edge_confidences={},
            graph_entities=graph_entities,
            graph_relations=graph_relations,
            metadata=metadata or {},
        )
        self._save_snapshot(snapshot)
        logger.info(f"Created snapshot: {snapshot_id} (as_of={as_of})")
        return snapshot

    # ------------------------------------------------------------------ #
    def get_snapshot_at(self, as_of: datetime) -> Optional[SystemSnapshot]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT snapshot_id, as_of, created_at,
                       series_values, feature_values, evidence_scores,
                       regime_state, edge_confidences,
                       graph_entities, graph_relations, metadata
                FROM snapshots
                WHERE as_of <= ?
                ORDER BY as_of DESC
                LIMIT 1
                """,
                (as_of.isoformat(),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_snapshot(row)

    # ------------------------------------------------------------------ #
    # Internal fetch helpers
    # ------------------------------------------------------------------ #
    def _fetch_series(self, series_ids: Optional[List[str]], as_of: datetime) -> Dict[str, float]:
        results = {}
        if not self.timeseries_repo:
            return results
        if series_ids:
            for sid in series_ids:
                obs = self.timeseries_repo.get_last(sid, as_of)
                if obs:
                    results[sid] = obs.value
        return results

    def _fetch_features(self, feature_ids: Optional[List[str]], as_of: datetime) -> Dict[str, float]:
        results = {}
        if not self.feature_store:
            return results
        if feature_ids:
            for fid in feature_ids:
                fv = self.feature_store.get_last(fid, as_of)
                if fv:
                    results[fid] = fv.value
        return results

    def _fetch_evidence(self, edge_ids: Optional[List[str]]) -> Dict[str, float]:
        results = {}
        if not self.evidence_store:
            return results
        if edge_ids:
            for eid in edge_ids:
                score = self.evidence_store.get_latest_score(eid)
                if score:
                    results[eid] = score.total_score
        return results

    def _fetch_regime(self, as_of: datetime) -> Dict[str, Any]:
        if not self.regime_store:
            return {}
        result = self.regime_store.get_at(as_of)
        if not result:
            return {}
        return {
            "primary_regime": result.primary_regime.value if result.primary_regime else None,
            "probability": result.primary_probability,
            "uncertainty": result.uncertainty,
        }

    def _fetch_graph(self) -> tuple:
        if not self.graph_repo:
            return [], []
        try:
            entities = self.graph_repo.get_all_entities()
            relations = self.graph_repo.get_all_relations()
            return entities, relations
        except Exception as e:
            logger.warning(f"Graph snapshot failed: {e}")
            return [], []

    # ------------------------------------------------------------------ #
    def _save_snapshot(self, snapshot: SystemSnapshot) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO snapshots (
                    snapshot_id, as_of, created_at,
                    series_values, feature_values, evidence_scores,
                    regime_state, edge_confidences,
                    graph_entities, graph_relations, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    series_values = excluded.series_values,
                    feature_values = excluded.feature_values,
                    evidence_scores = excluded.evidence_scores,
                    regime_state = excluded.regime_state,
                    edge_confidences = excluded.edge_confidences,
                    graph_entities = excluded.graph_entities,
                    graph_relations = excluded.graph_relations,
                    metadata = excluded.metadata
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.as_of.isoformat(),
                    snapshot.created_at.isoformat(),
                    json.dumps(snapshot.series_values),
                    json.dumps(snapshot.feature_values),
                    json.dumps(snapshot.evidence_scores),
                    json.dumps(snapshot.regime_state),
                    json.dumps(snapshot.edge_confidences),
                    json.dumps(snapshot.graph_entities),
                    json.dumps(snapshot.graph_relations),
                    json.dumps(snapshot.metadata),
                ),
            )
            conn.commit()

    def _row_to_snapshot(self, row) -> SystemSnapshot:
        (
            snapshot_id,
            as_of,
            created_at,
            series_values,
            feature_values,
            evidence_scores,
            regime_state,
            edge_confidences,
            graph_entities,
            graph_relations,
            metadata,
        ) = row
        return SystemSnapshot(
            snapshot_id=snapshot_id,
            as_of=datetime.fromisoformat(as_of),
            created_at=datetime.fromisoformat(created_at),
            series_values=json.loads(series_values or "{}"),
            feature_values=json.loads(feature_values or "{}"),
            evidence_scores=json.loads(evidence_scores or "{}"),
            regime_state=json.loads(regime_state or "{}"),
            edge_confidences=json.loads(edge_confidences or "{}"),
            graph_entities=json.loads(graph_entities or "[]"),
            graph_relations=json.loads(graph_relations or "[]"),
            metadata=json.loads(metadata or "{}"),
        )
