"""
Feature Store with as_of validation.
"""
import logging
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from src.common.asof_context import validate_access
from src.shared.schemas import FeatureValue

logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, db_path: str = "data/features.db"):
        self.db_path = db_path
        self._ensure_db()
    
    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_values (
                    feature_value_id TEXT PRIMARY KEY,
                    feature_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    value REAL NOT NULL,
                    computation_time_ms REAL,
                    input_observations_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(feature_id, timestamp, as_of)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fv_feature_time
                ON feature_values(feature_id, timestamp)
            """)
            conn.commit()
    
    def save_batch(
        self,
        values: List[FeatureValue],
        skip_duplicates: bool = True,
    ) -> tuple:
        if not values:
            return 0, 0
        stored = 0
        skipped = 0
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            for fv in values:
                try:
                    conn.execute("""
                        INSERT INTO feature_values (
                            feature_value_id, feature_id, timestamp, as_of,
                            value, computation_time_ms, input_observations_count, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fv.feature_value_id,
                        fv.feature_id,
                        fv.timestamp.isoformat(),
                        fv.as_of.isoformat(),
                        fv.value,
                        fv.computation_time_ms,
                        fv.input_observations_count,
                        now,
                    ))
                    stored += 1
                except sqlite3.IntegrityError:
                    if skip_duplicates:
                        skipped += 1
                    else:
                        raise
            conn.commit()
        return stored, skipped
    
    def get_range(
        self,
        feature_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[FeatureValue]:
        validate_access(as_of)
        query = "SELECT * FROM feature_values WHERE feature_id = ?"
        params = [feature_id]
        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())
        if as_of:
            query += " AND as_of <= ?"
            params.append(as_of.isoformat())
            query = f"""
                SELECT * FROM feature_values fv1
                WHERE feature_id = ?
                AND fv1.as_of = (
                    SELECT MAX(fv2.as_of) FROM feature_values fv2
                    WHERE fv2.feature_id = fv1.feature_id
                    AND fv2.timestamp = fv1.timestamp
                    AND fv2.as_of <= ?
                )
            """
            params = [feature_id, as_of.isoformat()]
            if start:
                query += " AND fv1.timestamp >= ?"
                params.append(start.isoformat())
            if end:
                query += " AND fv1.timestamp <= ?"
                params.append(end.isoformat())
        query += f" ORDER BY timestamp ASC LIMIT {limit}"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_feature_value(row) for row in rows]
    
    def get_last(
        self,
        feature_id: str,
        as_of: Optional[datetime] = None,
    ) -> Optional[FeatureValue]:
        validate_access(as_of)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if as_of:
                cursor = conn.execute("""
                    SELECT * FROM feature_values
                    WHERE feature_id = ? AND as_of <= ?
                    ORDER BY timestamp DESC, as_of DESC
                    LIMIT 1
                """, (feature_id, as_of.isoformat()))
            else:
                cursor = conn.execute("""
                    SELECT * FROM feature_values
                    WHERE feature_id = ?
                    ORDER BY timestamp DESC, as_of DESC
                    LIMIT 1
                """, (feature_id,))
            row = cursor.fetchone()
        return self._row_to_feature_value(row) if row else None
    
    def get_at(
        self,
        feature_id: str,
        timestamp: datetime,
        as_of: Optional[datetime] = None,
    ) -> Optional[FeatureValue]:
        validate_access(as_of)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if as_of:
                cursor = conn.execute("""
                    SELECT * FROM feature_values
                    WHERE feature_id = ? AND timestamp = ? AND as_of <= ?
                    ORDER BY as_of DESC
                    LIMIT 1
                """, (feature_id, timestamp.isoformat(), as_of.isoformat()))
            else:
                cursor = conn.execute("""
                    SELECT * FROM feature_values
                    WHERE feature_id = ? AND timestamp = ?
                    ORDER BY as_of DESC
                    LIMIT 1
                """, (feature_id, timestamp.isoformat()))
            row = cursor.fetchone()
        return self._row_to_feature_value(row) if row else None
    
    def get_latest_values(
        self,
        feature_ids: List[str],
        as_of: Optional[datetime] = None,
    ) -> Dict[str, FeatureValue]:
        result = {}
        for fid in feature_ids:
            fv = self.get_last(fid, as_of)
            if fv:
                result[fid] = fv
        return result
    
    def delete_range(
        self,
        feature_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> int:
        query = "DELETE FROM feature_values WHERE feature_id = ?"
        params = [feature_id]
        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def list_features(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT feature_id FROM feature_values"
            )
            return [row[0] for row in cursor.fetchall()]
    
    def count(self, feature_id: Optional[str] = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            if feature_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM feature_values WHERE feature_id = ?",
                    (feature_id,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM feature_values")
            return cursor.fetchone()[0]
    
    def get_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT feature_id) as feature_count
                FROM feature_values
            """)
            row = cursor.fetchone()
            
            cursor = conn.execute("""
                SELECT feature_id, COUNT(*) as cnt
                FROM feature_values
                GROUP BY feature_id
                ORDER BY cnt DESC
                LIMIT 10
            """)
            top_features = {r[0]: r[1] for r in cursor.fetchall()}
        
        return {
            "total_values": row[0] or 0,
            "feature_count": row[1] or 0,
            "top_features": top_features,
        }
    
    def _row_to_feature_value(self, row) -> FeatureValue:
        return FeatureValue(
            feature_value_id=row['feature_value_id'],
            feature_id=row['feature_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            as_of=datetime.fromisoformat(row['as_of']),
            value=row['value'],
            computation_time_ms=row['computation_time_ms'],
            input_observations_count=row['input_observations_count'],
        )
    
    def clear(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM feature_values")
            conn.commit()
            return cursor.rowcount
