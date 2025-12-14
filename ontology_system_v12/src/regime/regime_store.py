"""
Regime Store
레짐 탐지 결과 저장소

책임:
- 레짐 탐지 결과 저장
- 시계열 조회
- 리플레이/검증 지원
"""
import logging
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from src.shared.schemas import RegimeDetectionResult, RegimeType

logger = logging.getLogger(__name__)


class RegimeStore:
    """
    레짐 탐지 결과 저장소
    
    레짐 탐지 결과를 시계열로 저장합니다.
    리플레이와 검증에 필요.
    """
    
    def __init__(self, db_path: str = "data/regimes.db"):
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
                CREATE TABLE IF NOT EXISTS regime_detections (
                    detection_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    detected_regimes TEXT NOT NULL,
                    primary_regime TEXT,
                    primary_probability REAL DEFAULT 0.0,
                    uncertainty REAL DEFAULT 0.0,
                    feature_snapshot TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(timestamp)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rd_timestamp
                ON regime_detections(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rd_primary
                ON regime_detections(primary_regime)
            """)
            
            conn.commit()
        
        logger.debug(f"RegimeStore initialized at {self.db_path}")
    
    def save(self, result: RegimeDetectionResult) -> bool:
        """탐지 결과 저장"""
        now = datetime.now().isoformat()
        
        try:
            # detected_regimes를 JSON으로 변환
            regimes_json = json.dumps({
                k.value if hasattr(k, 'value') else k: v
                for k, v in result.detected_regimes.items()
            })
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO regime_detections (
                        detection_id, timestamp, detected_regimes,
                        primary_regime, primary_probability, uncertainty,
                        feature_snapshot, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(timestamp) DO UPDATE SET
                        detected_regimes = excluded.detected_regimes,
                        primary_regime = excluded.primary_regime,
                        primary_probability = excluded.primary_probability,
                        uncertainty = excluded.uncertainty,
                        feature_snapshot = excluded.feature_snapshot
                """, (
                    result.detection_id,
                    result.timestamp.isoformat(),
                    regimes_json,
                    result.primary_regime.value if result.primary_regime else None,
                    result.primary_probability,
                    result.uncertainty,
                    json.dumps(result.feature_snapshot),
                    now,
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save regime detection: {e}")
            return False
    
    def get_at(
        self,
        timestamp: datetime,
    ) -> Optional[RegimeDetectionResult]:
        """특정 시점 결과 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM regime_detections
                WHERE timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (timestamp.isoformat(),))
            row = cursor.fetchone()
        
        return self._row_to_result(row) if row else None
    
    def get_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[RegimeDetectionResult]:
        """범위 조회"""
        query = "SELECT * FROM regime_detections WHERE 1=1"
        params = []
        
        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        return [self._row_to_result(row) for row in rows]
    
    def get_latest(self) -> Optional[RegimeDetectionResult]:
        """최신 결과"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM regime_detections
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
        
        return self._row_to_result(row) if row else None
    
    def get_by_regime(
        self,
        regime_type: RegimeType,
        limit: int = 100,
    ) -> List[RegimeDetectionResult]:
        """특정 레짐인 결과만 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM regime_detections
                WHERE primary_regime = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (regime_type.value, limit))
            rows = cursor.fetchall()
        
        return [self._row_to_result(row) for row in rows]
    
    def get_transitions(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        레짐 전환 시점 조회
        
        Returns:
            [{"timestamp": ..., "from": ..., "to": ...}, ...]
        """
        results = self.get_range(start, end, limit=10000)
        transitions = []
        
        prev_regime = None
        for result in reversed(results):  # 시간순 정렬
            if result.primary_regime and result.primary_probability > 0.5:
                if prev_regime and result.primary_regime != prev_regime:
                    transitions.append({
                        "timestamp": result.timestamp,
                        "from": prev_regime.value,
                        "to": result.primary_regime.value,
                    })
                prev_regime = result.primary_regime
        
        return transitions
    
    def count(self, regime_type: Optional[RegimeType] = None) -> int:
        """레코드 수"""
        with sqlite3.connect(self.db_path) as conn:
            if regime_type:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM regime_detections WHERE primary_regime = ?",
                    (regime_type.value,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM regime_detections")
            return cursor.fetchone()[0]
    
    def get_stats(self) -> Dict:
        """통계"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(primary_probability) as avg_prob,
                    AVG(uncertainty) as avg_uncertainty
                FROM regime_detections
            """)
            row = cursor.fetchone()
            
            # 레짐별 분포
            cursor = conn.execute("""
                SELECT primary_regime, COUNT(*) as cnt
                FROM regime_detections
                WHERE primary_regime IS NOT NULL
                GROUP BY primary_regime
            """)
            by_regime = {r[0]: r[1] for r in cursor.fetchall()}
        
        return {
            "total_detections": row[0] or 0,
            "avg_probability": row[1] or 0,
            "avg_uncertainty": row[2] or 0,
            "by_regime": by_regime,
        }
    
    def _row_to_result(self, row) -> RegimeDetectionResult:
        """Row를 RegimeDetectionResult로 변환"""
        # detected_regimes JSON 파싱
        regimes_json = json.loads(row['detected_regimes'])
        detected_regimes = {}
        for k, v in regimes_json.items():
            try:
                detected_regimes[RegimeType(k)] = v
            except:
                pass
        
        # primary_regime 파싱
        primary_regime = None
        if row['primary_regime']:
            try:
                primary_regime = RegimeType(row['primary_regime'])
            except:
                pass
        
        return RegimeDetectionResult(
            detection_id=row['detection_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            detected_regimes=detected_regimes,
            primary_regime=primary_regime,
            primary_probability=row['primary_probability'],
            uncertainty=row['uncertainty'],
            feature_snapshot=json.loads(row['feature_snapshot']) if row['feature_snapshot'] else {},
        )
    
    def clear(self) -> int:
        """모든 데이터 삭제 (테스트용)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM regime_detections")
            conn.commit()
            return cursor.rowcount
