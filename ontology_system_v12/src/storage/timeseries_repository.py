"""
Time Series Repository
Append-only 시계열 저장소

책임:
- 시계열 데이터 저장 (Append-only)
- as_of 버전 관리
- Range 조회
"""
import logging
import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.shared.schemas import Observation, TimeSeriesMetadata
from src.common.asof_context import validate_access

logger = logging.getLogger(__name__)


class TimeSeriesRepository:
    """
    Append-only 시계열 저장소
    
    특징:
    - 한번 저장된 데이터는 수정/삭제 불가
    - 수정이 필요하면 새 as_of로 새 레코드 추가
    - is_revision=true로 수정 데이터 표시
    """
    
    def __init__(self, db_path: str = "data/timeseries.db"):
        """
        Args:
            db_path: SQLite 데이터베이스 경로
        """
        self.db_path = db_path
        self._ensure_db()
    
    def _ensure_db(self) -> None:
        """데이터베이스 및 테이블 생성"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # 관측값 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    series_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    source_id TEXT NOT NULL,
                    is_revision INTEGER DEFAULT 0,
                    original_value REAL,
                    quality_flag TEXT DEFAULT 'ok',
                    created_at TEXT NOT NULL,
                    UNIQUE(series_id, timestamp, as_of)
                )
            """)
            
            # 인덱스
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_obs_series_time 
                ON observations(series_id, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_obs_series_asof 
                ON observations(series_id, as_of)
            """)
            
            # 시계열 메타데이터 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS series_metadata (
                    series_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    source_id TEXT NOT NULL,
                    frequency TEXT DEFAULT 'daily',
                    unit TEXT,
                    start_date TEXT,
                    last_updated TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.commit()
        
        logger.debug(f"TimeSeriesRepository initialized at {self.db_path}")
    
    def append_batch(
        self,
        observations: List[Observation],
        skip_duplicates: bool = True,
    ) -> Tuple[int, int]:
        """
        배치 저장 (Append-only)
        
        Args:
            observations: 관측값 리스트
            skip_duplicates: 중복 시 스킵 (False면 에러)
        
        Returns:
            (저장된 수, 스킵된 수)
        """
        if not observations:
            return 0, 0
        
        stored = 0
        skipped = 0
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            for obs in observations:
                try:
                    conn.execute("""
                        INSERT INTO observations (
                            observation_id, series_id, timestamp, as_of, value, unit,
                            source_id, is_revision, original_value, quality_flag, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        obs.observation_id,
                        obs.series_id,
                        obs.timestamp.isoformat(),
                        obs.as_of.isoformat(),
                        obs.value,
                        obs.unit,
                        obs.source_id,
                        1 if obs.is_revision else 0,
                        obs.original_value,
                        obs.quality_flag,
                        now,
                    ))
                    stored += 1
                except sqlite3.IntegrityError:
                    if skip_duplicates:
                        skipped += 1
                    else:
                        raise
            
            conn.commit()
        
        logger.debug(f"Appended {stored} observations, skipped {skipped}")
        return stored, skipped
    
    def get_range(
        self,
        series_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[Observation]:
        """
        범위 조회
        
        Args:
            series_id: 시계열 ID
            start: 시작 시점 (포함)
            end: 종료 시점 (포함)
            as_of: 이 시점 기준 최신 버전만 조회
            limit: 최대 레코드 수
        
        Returns:
            Observation 리스트
        """
        validate_access(as_of or end)
        query = "SELECT * FROM observations WHERE series_id = ?"
        params = [series_id]
        
        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())
        
        if as_of:
            # as_of 기준 최신 버전만
            query = f"""
                SELECT * FROM observations o1
                WHERE series_id = ?
                AND o1.as_of = (
                    SELECT MAX(o2.as_of) FROM observations o2
                    WHERE o2.series_id = o1.series_id
                    AND o2.timestamp = o1.timestamp
                    AND o2.as_of <= ?
                )
            """
            params = [series_id, as_of.isoformat()]
            
            if start:
                query += " AND o1.timestamp >= ?"
                params.append(start.isoformat())
            if end:
                query += " AND o1.timestamp <= ?"
                params.append(end.isoformat())
        
        query += f" ORDER BY timestamp ASC LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        return [self._row_to_observation(row) for row in rows]
    
    def get_last(
        self,
        series_id: str,
        as_of: Optional[datetime] = None,
    ) -> Optional[Observation]:
        """
        최신 관측값 조회
        
        Args:
            series_id: 시계열 ID
            as_of: 이 시점 기준 최신
        
        Returns:
            최신 Observation 또는 None
        """
        validate_access(as_of)
        results = self.get_range(series_id, as_of=as_of, limit=1)
        
        if not results:
            return None
        
        # 가장 최근 timestamp
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if as_of:
                cursor = conn.execute("""
                    SELECT * FROM observations
                    WHERE series_id = ? AND as_of <= ?
                    ORDER BY timestamp DESC, as_of DESC
                    LIMIT 1
                """, (series_id, as_of.isoformat()))
            else:
                cursor = conn.execute("""
                    SELECT * FROM observations
                    WHERE series_id = ?
                    ORDER BY timestamp DESC, as_of DESC
                    LIMIT 1
                """, (series_id,))
            
            row = cursor.fetchone()
        
        return self._row_to_observation(row) if row else None
    
    def get_latest_version(
        self,
        series_id: str,
        timestamp: datetime,
    ) -> Optional[Observation]:
        """
        특정 시점의 최신 버전 조회
        
        Args:
            series_id: 시계열 ID
            timestamp: 관측 시점
        
        Returns:
            최신 버전 Observation 또는 None
        """
        validate_access(timestamp)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM observations
                WHERE series_id = ? AND timestamp = ?
                ORDER BY as_of DESC
                LIMIT 1
            """, (series_id, timestamp.isoformat()))
            row = cursor.fetchone()
        
        return self._row_to_observation(row) if row else None
    
    def get_all_versions(
        self,
        series_id: str,
        timestamp: datetime,
    ) -> List[Observation]:
        """
        특정 시점의 모든 버전 조회 (히스토리)
        
        Args:
            series_id: 시계열 ID
            timestamp: 관측 시점
        
        Returns:
            버전 순서대로 Observation 리스트
        """
        validate_access(timestamp)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM observations
                WHERE series_id = ? AND timestamp = ?
                ORDER BY as_of ASC
            """, (series_id, timestamp.isoformat()))
            rows = cursor.fetchall()
        
        return [self._row_to_observation(row) for row in rows]
    
    def register_series(self, metadata: TimeSeriesMetadata) -> None:
        """시계열 메타데이터 등록"""
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO series_metadata (
                    series_id, name, description, source_id, frequency,
                    unit, start_date, last_updated, tags, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(series_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    frequency = excluded.frequency,
                    unit = excluded.unit,
                    last_updated = excluded.last_updated,
                    tags = excluded.tags
            """, (
                metadata.series_id,
                metadata.name,
                metadata.description,
                metadata.source_id,
                metadata.frequency,
                metadata.unit,
                metadata.start_date.isoformat() if metadata.start_date else None,
                metadata.last_updated.isoformat() if metadata.last_updated else None,
                ",".join(metadata.tags),
                now,
            ))
            conn.commit()
    
    def get_series_metadata(self, series_id: str) -> Optional[TimeSeriesMetadata]:
        """시계열 메타데이터 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM series_metadata WHERE series_id = ?",
                (series_id,)
            )
            row = cursor.fetchone()
        
        if not row:
            return None
        
        return TimeSeriesMetadata(
            series_id=row['series_id'],
            name=row['name'],
            description=row['description'],
            source_id=row['source_id'],
            frequency=row['frequency'],
            unit=row['unit'],
            start_date=datetime.fromisoformat(row['start_date']) if row['start_date'] else None,
            last_updated=datetime.fromisoformat(row['last_updated']) if row['last_updated'] else None,
            tags=row['tags'].split(",") if row['tags'] else [],
        )
    
    def list_series(self) -> List[str]:
        """모든 시계열 ID 목록"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT series_id FROM observations")
            return [row[0] for row in cursor.fetchall()]
    
    def count(self, series_id: Optional[str] = None) -> int:
        """레코드 수 조회"""
        with sqlite3.connect(self.db_path) as conn:
            if series_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM observations WHERE series_id = ?",
                    (series_id,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM observations")
            return cursor.fetchone()[0]
    
    def get_stats(self) -> Dict:
        """통계 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT series_id) as series_count,
                    SUM(CASE WHEN is_revision = 1 THEN 1 ELSE 0 END) as revisions
                FROM observations
            """)
            row = cursor.fetchone()
        
        return {
            "total_observations": row[0] or 0,
            "series_count": row[1] or 0,
            "revision_count": row[2] or 0,
        }
    
    def _row_to_observation(self, row) -> Observation:
        """Row를 Observation으로 변환"""
        return Observation(
            observation_id=row['observation_id'],
            series_id=row['series_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            as_of=datetime.fromisoformat(row['as_of']),
            value=row['value'],
            unit=row['unit'],
            source_id=row['source_id'],
            is_revision=bool(row['is_revision']),
            original_value=row['original_value'],
            quality_flag=row['quality_flag'],
        )
    
    def clear(self) -> int:
        """모든 데이터 삭제 (테스트용)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM observations")
            conn.commit()
            return cursor.rowcount
