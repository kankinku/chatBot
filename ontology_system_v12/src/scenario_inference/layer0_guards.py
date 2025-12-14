"""
Layer 0: 공통 계약 (필수 안전장치)

AsOfContextGuard: 미래 데이터 누수 차단
SnapshotResolver: 시점별 스냅샷 고정
"""
import logging
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from functools import wraps
import threading

logger = logging.getLogger(__name__)


# ============================================================
# AsOfContextGuard
# ============================================================

@dataclass
class QueryAuditLog:
    """조회 감사 로그"""
    timestamp: datetime
    as_of_time: datetime
    query_type: str  # timeseries, kg_edge, evidence, feature
    target_id: str
    requested_time: Optional[datetime]
    was_allowed: bool
    violation_type: Optional[str] = None  # future_leak, no_context


class AsOfContextGuard:
    """
    As-Of 컨텍스트 가드
    
    책임:
    - 모든 추론/검증에 `as_of_time` 강제
    - 조회는 `<= as_of_time`만 허용 (누수 차단)
    - 감사 로그 기록
    
    실패모드:
    - 우회 조회가 1곳이라도 생기면 학습은 "가짜 고성능"으로 망함
    """
    
    # Thread-local context
    _context = threading.local()
    
    def __init__(self, enable_strict_mode: bool = True):
        """
        Args:
            enable_strict_mode: True면 as_of 없이 조회 시 예외 발생
        """
        self.enable_strict_mode = enable_strict_mode
        self._audit_logs: List[QueryAuditLog] = []
        self._violation_count = 0
    
    @classmethod
    def set_context(cls, as_of_time: datetime) -> None:
        """현재 스레드의 as_of_time 설정"""
        cls._context.as_of_time = as_of_time
        cls._context.is_set = True
    
    @classmethod
    def get_context(cls) -> Optional[datetime]:
        """현재 스레드의 as_of_time 조회"""
        if hasattr(cls._context, 'as_of_time') and getattr(cls._context, 'is_set', False):
            return cls._context.as_of_time
        return None
    
    @classmethod
    def clear_context(cls) -> None:
        """컨텍스트 초기화"""
        cls._context.as_of_time = None
        cls._context.is_set = False
    
    def validate_query(
        self,
        query_type: str,
        target_id: str,
        requested_time: Optional[datetime] = None,
    ) -> bool:
        """
        조회 유효성 검증
        
        Args:
            query_type: 조회 유형 (timeseries, kg_edge, evidence, feature)
            target_id: 대상 ID
            requested_time: 요청된 시점 (None이면 컨텍스트의 as_of_time 사용)
        
        Returns:
            True: 허용, False: 차단
        
        Raises:
            ValueError: strict_mode에서 컨텍스트 없거나 누수 시
        """
        as_of_time = self.get_context()
        
        # 컨텍스트 검증
        if as_of_time is None:
            self._log_audit(
                query_type, target_id, requested_time,
                was_allowed=False, violation_type="no_context"
            )
            self._violation_count += 1
            
            if self.enable_strict_mode:
                raise ValueError(
                    f"AsOfContextGuard: as_of_time 컨텍스트가 설정되지 않음. "
                    f"query_type={query_type}, target={target_id}"
                )
            return False
        
        # 미래 데이터 누수 검증
        check_time = requested_time or as_of_time
        if check_time > as_of_time:
            self._log_audit(
                query_type, target_id, requested_time,
                was_allowed=False, violation_type="future_leak"
            )
            self._violation_count += 1
            
            if self.enable_strict_mode:
                raise ValueError(
                    f"AsOfContextGuard: 미래 데이터 접근 시도! "
                    f"as_of={as_of_time}, requested={check_time}, "
                    f"query_type={query_type}, target={target_id}"
                )
            return False
        
        # 허용
        self._log_audit(query_type, target_id, requested_time, was_allowed=True)
        return True
    
    def _log_audit(
        self,
        query_type: str,
        target_id: str,
        requested_time: Optional[datetime],
        was_allowed: bool,
        violation_type: Optional[str] = None,
    ) -> None:
        """감사 로그 기록"""
        log = QueryAuditLog(
            timestamp=datetime.now(),
            as_of_time=self.get_context() or datetime.min,
            query_type=query_type,
            target_id=target_id,
            requested_time=requested_time,
            was_allowed=was_allowed,
            violation_type=violation_type,
        )
        self._audit_logs.append(log)
        
        if violation_type:
            logger.warning(
                f"[AsOfGuard VIOLATION] {violation_type}: "
                f"{query_type}/{target_id}, as_of={log.as_of_time}"
            )
    
    def get_audit_logs(self, limit: int = 100) -> List[QueryAuditLog]:
        """감사 로그 조회"""
        return self._audit_logs[-limit:]
    
    def get_violation_count(self) -> int:
        """위반 횟수"""
        return self._violation_count
    
    def generate_leak_report(self) -> Dict:
        """
        누수 감사 리포트 생성
        
        배포 전 자동 검증에 사용
        """
        violations = [log for log in self._audit_logs if not log.was_allowed]
        
        by_type = {}
        for v in violations:
            vtype = v.violation_type or "unknown"
            by_type[vtype] = by_type.get(vtype, 0) + 1
        
        return {
            "total_queries": len(self._audit_logs),
            "total_violations": len(violations),
            "violation_rate": len(violations) / max(1, len(self._audit_logs)),
            "by_violation_type": by_type,
            "recent_violations": [
                {
                    "type": v.violation_type,
                    "query_type": v.query_type,
                    "target": v.target_id,
                    "as_of": v.as_of_time.isoformat() if v.as_of_time != datetime.min else None,
                }
                for v in violations[-10:]
            ],
            "is_clean": len(violations) == 0,
        }
    
    def reset(self) -> None:
        """상태 초기화"""
        self._audit_logs.clear()
        self._violation_count = 0
        self.clear_context()


def require_as_of_context(func: Callable) -> Callable:
    """
    as_of 컨텍스트 필수 데코레이터
    
    사용법:
    @require_as_of_context
    def my_query_function(...):
        ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        guard = AsOfContextGuard(enable_strict_mode=True)
        guard.validate_query("function_call", func.__name__)
        return func(*args, **kwargs)
    return wrapper


# ============================================================
# SnapshotResolver
# ============================================================

@dataclass
class SnapshotRef:
    """스냅샷 참조"""
    ref_id: str
    snapshot_type: str  # kg, timeseries, evidence
    as_of_time: datetime
    version: str
    content_hash: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class SnapshotResolver:
    """
    스냅샷 해석기
    
    책임:
    - KG/Evidence/TimeSeries를 "그 시점 버전"으로 고정
    - 리비전/스키마 변경 포함
    
    실패모드:
    - 버전 미고정 → 재현 불가, 학습 데이터 오염
    """
    
    def __init__(
        self,
        kg_store: Optional[Any] = None,
        ts_store: Optional[Any] = None,
        evidence_store: Optional[Any] = None,
    ):
        """
        Args:
            kg_store: KG 저장소
            ts_store: 시계열 저장소
            evidence_store: 증거 저장소
        """
        self.kg_store = kg_store
        self.ts_store = ts_store
        self.evidence_store = evidence_store
        
        self._snapshot_cache: Dict[str, SnapshotRef] = {}
    
    def resolve_kg_snapshot(self, as_of_time: datetime) -> SnapshotRef:
        """
        KG 스냅샷 해석
        
        해당 시점에서 사용 가능했던 KG 상태 참조
        """
        cache_key = f"kg_{as_of_time.isoformat()}"
        
        if cache_key in self._snapshot_cache:
            return self._snapshot_cache[cache_key]
        
        # 실제 구현: kg_store.get_snapshot_at(as_of_time)
        # 여기서는 stub
        ref = SnapshotRef(
            ref_id=f"KG_SNAP_{as_of_time.strftime('%Y%m%d%H%M%S')}",
            snapshot_type="kg",
            as_of_time=as_of_time,
            version=self._get_kg_version(as_of_time),
        )
        
        self._snapshot_cache[cache_key] = ref
        return ref
    
    def resolve_ts_snapshot(self, as_of_time: datetime) -> SnapshotRef:
        """시계열 스냅샷 해석"""
        cache_key = f"ts_{as_of_time.isoformat()}"
        
        if cache_key in self._snapshot_cache:
            return self._snapshot_cache[cache_key]
        
        ref = SnapshotRef(
            ref_id=f"TS_SNAP_{as_of_time.strftime('%Y%m%d%H%M%S')}",
            snapshot_type="timeseries",
            as_of_time=as_of_time,
            version=self._get_ts_version(as_of_time),
        )
        
        self._snapshot_cache[cache_key] = ref
        return ref
    
    def resolve_evidence_snapshot(self, as_of_time: datetime) -> SnapshotRef:
        """증거 스냅샷 해석"""
        cache_key = f"ev_{as_of_time.isoformat()}"
        
        if cache_key in self._snapshot_cache:
            return self._snapshot_cache[cache_key]
        
        ref = SnapshotRef(
            ref_id=f"EV_SNAP_{as_of_time.strftime('%Y%m%d%H%M%S')}",
            snapshot_type="evidence",
            as_of_time=as_of_time,
            version=self._get_evidence_version(as_of_time),
        )
        
        self._snapshot_cache[cache_key] = ref
        return ref
    
    def resolve_all(self, as_of_time: datetime) -> Dict[str, SnapshotRef]:
        """모든 스냅샷 한번에 해석"""
        return {
            "kg": self.resolve_kg_snapshot(as_of_time),
            "timeseries": self.resolve_ts_snapshot(as_of_time),
            "evidence": self.resolve_evidence_snapshot(as_of_time),
        }
    
    def _get_kg_version(self, as_of_time: datetime) -> str:
        """KG 버전 조회 (stub)"""
        return f"kg_v{as_of_time.strftime('%Y%m%d')}"
    
    def _get_ts_version(self, as_of_time: datetime) -> str:
        """시계열 버전 조회 (stub)"""
        return f"ts_v{as_of_time.strftime('%Y%m%d')}"
    
    def _get_evidence_version(self, as_of_time: datetime) -> str:
        """증거 버전 조회 (stub)"""
        return f"ev_v{as_of_time.strftime('%Y%m%d')}"
    
    def verify_snapshot_consistency(
        self,
        ref: SnapshotRef,
        current_hash: Optional[str] = None,
    ) -> bool:
        """
        스냅샷 일관성 검증
        
        저장된 해시와 현재 해시 비교
        """
        if not current_hash or not ref.content_hash:
            return True  # 검증 불가 시 통과
        
        return ref.content_hash == current_hash
    
    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._snapshot_cache.clear()


# ============================================================
# Context Manager
# ============================================================

class AsOfContext:
    """
    as_of_time 컨텍스트 매니저
    
    사용법:
    with AsOfContext(datetime(2024, 12, 1)):
        # 이 블록 내에서 모든 조회는 2024-12-01 기준
        data = query_data(...)
    """
    
    def __init__(self, as_of_time: datetime):
        self.as_of_time = as_of_time
        self._previous_context: Optional[datetime] = None
    
    def __enter__(self):
        self._previous_context = AsOfContextGuard.get_context()
        AsOfContextGuard.set_context(self.as_of_time)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._previous_context:
            AsOfContextGuard.set_context(self._previous_context)
        else:
            AsOfContextGuard.clear_context()
        return False
