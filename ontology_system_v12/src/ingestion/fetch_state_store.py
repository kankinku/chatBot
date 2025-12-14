"""
Fetch State Store
소스별 수집 상태(마지막 위치) 관리

책임:
- 수집 위치 저장/조회
- 실패 상태 추적
- 통계 관리

저장소: SQLite (경량, 파일 기반)
"""
import logging
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from src.shared.schemas import FetchState

logger = logging.getLogger(__name__)


class FetchStateStore:
    """
    소스별 수집 상태 저장소
    
    (source_id, stream)별로 마지막 수집 위치를 저장합니다.
    Raw 데이터 저장소와 분리되어 덮어쓰기 가능합니다.
    """
    
    def __init__(self, db_path: str = "data/fetch_state.db"):
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fetch_state (
                    source_id TEXT NOT NULL,
                    stream TEXT NOT NULL DEFAULT 'default',
                    last_timestamp TEXT,
                    last_cursor TEXT,
                    last_etag TEXT,
                    last_hash TEXT,
                    last_success_at TEXT,
                    last_failure_at TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    total_fetched INTEGER DEFAULT 0,
                    total_errors INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source_id, stream)
                )
            """)
            conn.commit()
        
        logger.debug(f"FetchStateStore initialized at {self.db_path}")
    
    def get(self, source_id: str, stream: str = "default") -> Optional[FetchState]:
        """수집 상태 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM fetch_state WHERE source_id = ? AND stream = ?",
                (source_id, stream)
            )
            row = cursor.fetchone()
            
        if not row:
            return None
        
        return FetchState(
            source_id=row['source_id'],
            stream=row['stream'],
            last_timestamp=datetime.fromisoformat(row['last_timestamp']) if row['last_timestamp'] else None,
            last_cursor=row['last_cursor'],
            last_etag=row['last_etag'],
            last_hash=row['last_hash'],
            last_success_at=datetime.fromisoformat(row['last_success_at']) if row['last_success_at'] else None,
            last_failure_at=datetime.fromisoformat(row['last_failure_at']) if row['last_failure_at'] else None,
            consecutive_failures=row['consecutive_failures'],
            total_fetched=row['total_fetched'],
            total_errors=row['total_errors'],
        )
    
    def save(self, state: FetchState) -> None:
        """수집 상태 저장 (upsert)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO fetch_state (
                    source_id, stream, last_timestamp, last_cursor, last_etag, last_hash,
                    last_success_at, last_failure_at, consecutive_failures,
                    total_fetched, total_errors, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, stream) DO UPDATE SET
                    last_timestamp = excluded.last_timestamp,
                    last_cursor = excluded.last_cursor,
                    last_etag = excluded.last_etag,
                    last_hash = excluded.last_hash,
                    last_success_at = excluded.last_success_at,
                    last_failure_at = excluded.last_failure_at,
                    consecutive_failures = excluded.consecutive_failures,
                    total_fetched = excluded.total_fetched,
                    total_errors = excluded.total_errors,
                    updated_at = excluded.updated_at
            """, (
                state.source_id,
                state.stream,
                state.last_timestamp.isoformat() if state.last_timestamp else None,
                state.last_cursor,
                state.last_etag,
                state.last_hash,
                state.last_success_at.isoformat() if state.last_success_at else None,
                state.last_failure_at.isoformat() if state.last_failure_at else None,
                state.consecutive_failures,
                state.total_fetched,
                state.total_errors,
                datetime.now().isoformat(),
            ))
            conn.commit()
        
        logger.debug(f"Saved fetch state: {state.source_id}/{state.stream}")
    
    def update_success(
        self,
        source_id: str,
        stream: str = "default",
        fetched_count: int = 0,
        last_timestamp: Optional[datetime] = None,
        last_cursor: Optional[str] = None,
        last_etag: Optional[str] = None,
        last_hash: Optional[str] = None,
    ) -> None:
        """성공 상태 업데이트"""
        state = self.get(source_id, stream) or FetchState(source_id=source_id, stream=stream)
        
        state.last_success_at = datetime.now()
        state.consecutive_failures = 0
        state.total_fetched += fetched_count
        
        if last_timestamp:
            state.last_timestamp = last_timestamp
        if last_cursor:
            state.last_cursor = last_cursor
        if last_etag:
            state.last_etag = last_etag
        if last_hash:
            state.last_hash = last_hash
        
        self.save(state)
    
    def update_failure(self, source_id: str, stream: str = "default") -> None:
        """실패 상태 업데이트"""
        state = self.get(source_id, stream) or FetchState(source_id=source_id, stream=stream)
        
        state.last_failure_at = datetime.now()
        state.consecutive_failures += 1
        state.total_errors += 1
        
        self.save(state)
    
    def list_all(self) -> List[FetchState]:
        """모든 상태 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM fetch_state ORDER BY source_id, stream")
            rows = cursor.fetchall()
        
        result = []
        for row in rows:
            result.append(FetchState(
                source_id=row['source_id'],
                stream=row['stream'],
                last_timestamp=datetime.fromisoformat(row['last_timestamp']) if row['last_timestamp'] else None,
                last_cursor=row['last_cursor'],
                last_etag=row['last_etag'],
                last_hash=row['last_hash'],
                last_success_at=datetime.fromisoformat(row['last_success_at']) if row['last_success_at'] else None,
                last_failure_at=datetime.fromisoformat(row['last_failure_at']) if row['last_failure_at'] else None,
                consecutive_failures=row['consecutive_failures'],
                total_fetched=row['total_fetched'],
                total_errors=row['total_errors'],
            ))
        
        return result
    
    def delete(self, source_id: str, stream: str = "default") -> bool:
        """상태 삭제"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM fetch_state WHERE source_id = ? AND stream = ?",
                (source_id, stream)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def clear_all(self) -> int:
        """모든 상태 삭제"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM fetch_state")
            conn.commit()
            return cursor.rowcount
    
    def get_stats(self) -> Dict:
        """통계 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(total_fetched) as total_fetched,
                    SUM(total_errors) as total_errors,
                    SUM(CASE WHEN consecutive_failures > 0 THEN 1 ELSE 0 END) as failing
                FROM fetch_state
            """)
            row = cursor.fetchone()
        
        return {
            "total_states": row[0] or 0,
            "total_fetched": row[1] or 0,
            "total_errors": row[2] or 0,
            "failing_sources": row[3] or 0,
        }
