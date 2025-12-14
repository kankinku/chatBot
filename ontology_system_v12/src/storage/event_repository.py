"""
Simplified Event Repository with as_of validation.
"""
import logging
import sqlite3
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import json

from src.common.asof_context import validate_access
from src.shared.schemas import Event

logger = logging.getLogger(__name__)


class EventRepository:
    def __init__(self, db_path: str = "data/events.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    source TEXT,
                    external_id TEXT,
                    content_hash TEXT,
                    payload TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")
            conn.commit()

    def save(self, event: Event) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO events (
                    event_id, source, external_id, content_hash, payload, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.source,
                event.external_id,
                event.content_hash,
                json.dumps(event.payload or {}),
                event.timestamp.isoformat(),
                datetime.utcnow().isoformat(),
            ))
            conn.commit()

    def get_by_id(self, event_id: str, *, as_of: Optional[datetime] = None) -> Optional[Event]:
        validate_access(as_of)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        return self._row_to_event(row) if row else None

    def get_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
        *,
        as_of: Optional[datetime] = None,
    ) -> List[Event]:
        validate_access(as_of or end)
        query = "SELECT * FROM events WHERE 1=1"
        params: List[Any] = []
        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
        return {"total_events": row["cnt"] if row else 0}

    def _row_to_event(self, row) -> Event:
        return Event(
            event_id=row["event_id"],
            source=row["source"],
            external_id=row["external_id"],
            content_hash=row["content_hash"],
            payload=json.loads(row["payload"] or "{}"),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )
