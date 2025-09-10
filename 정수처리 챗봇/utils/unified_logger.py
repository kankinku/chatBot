"""
통합 로깅 시스템

모든 로깅을 중앙에서 관리하는 통합 로거
디자인 패턴: Singleton + Observer + Strategy
"""

import logging
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from datetime import datetime
from collections import deque
import queue

# 에러 로거 import
try:
    from .error_logger import log_error, log_system_error
except ImportError:
    # 상대 import 실패 시 절대 import 시도
    try:
        from utils.error_logger import log_error, log_system_error
    except ImportError:
        # 폴백 함수들
        def log_error(error, context=None, additional_info=None):
            print(f"에러 로그 기록 실패: {error}", file=sys.stderr)
        
        def log_system_error(message, module="unknown", additional_info=None):
            print(f"시스템 에러: {message}", file=sys.stderr)

class LogLevel(Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """로그 카테고리"""
    SYSTEM = "system"
    SECURITY = "security"
    ROUTER = "router"
    QUERY = "query"
    MODEL = "model"
    CACHE = "cache"
    DATABASE = "database"
    PERFORMANCE = "performance"
    LEGAL = "legal"

@dataclass
class LogEntry:
    """통합 로그 엔트리"""
    timestamp: str
    level: LogLevel
    category: LogCategory
    module: str
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    execution_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None

class LogObserver:
    """로그 옵저버 인터페이스"""
    
    def on_log(self, entry: LogEntry):
        """로그 이벤트 처리"""
        raise NotImplementedError

# 파일 로그 옵저버들은 제거됨 (에러 로그만 유지)

class ConsoleLogObserver(LogObserver):
    """콘솔 로그 옵저버"""
    
    def __init__(self, min_level: LogLevel = LogLevel.INFO):
        self.min_level = min_level
        self._level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4
        }
    
    def on_log(self, entry: LogEntry):
        """콘솔에 로그 출력"""
        if self._level_order[entry.level] >= self._level_order[self.min_level]:
            formatted = self._format_for_console(entry)
            print(formatted)
    
    def _format_for_console(self, entry: LogEntry) -> str:
        """콘솔용 로그 포맷팅"""
        level_colors = {
            LogLevel.DEBUG: '\033[36m',    # 청록색
            LogLevel.INFO: '\033[32m',     # 녹색
            LogLevel.WARNING: '\033[33m',  # 노란색
            LogLevel.ERROR: '\033[31m',    # 빨간색
            LogLevel.CRITICAL: '\033[35m'  # 자주색
        }
        reset_color = '\033[0m'
        
        color = level_colors.get(entry.level, '')
        
        base = f"{color}[{entry.timestamp}] {entry.level.value} | {entry.category.value} | {entry.message}{reset_color}"
        
        if entry.execution_time_ms:
            base += f" ({entry.execution_time_ms:.2f}ms)"
        
        return base

class UnifiedLogger:
    """
    통합 로깅 시스템
    
    특징:
    - 싱글톤 패턴으로 전역 접근
    - 옵저버 패턴으로 다양한 출력 지원
    - 비동기 로깅으로 성능 최적화
    - 카테고리별 로그 분류
    - 성능 측정 내장
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._observers: List[LogObserver] = []
        self._log_queue = queue.Queue()
        self._worker_thread = None
        self._shutdown_event = threading.Event()
        self._session_id = None
        
        # 기본 옵저버 설정
        self._setup_default_observers()
        
        # 워커 스레드 시작
        self._start_worker()
        
        self._initialized = True
        
        # 초기화 로그
        self.info("UnifiedLogger 초기화 완료", LogCategory.SYSTEM, "unified_logger")
    
    def _setup_default_observers(self):
        """기본 옵저버 설정"""
        # 콘솔 출력만 활성화 (WARNING 이상)
        console_observer = ConsoleLogObserver(LogLevel.WARNING)
        self.add_observer(console_observer)
    
    def add_observer(self, observer: LogObserver):
        """옵저버 추가"""
        self._observers.append(observer)
    
    def remove_observer(self, observer: LogObserver):
        """옵저버 제거"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _start_worker(self):
        """워커 스레드 시작"""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def _worker_loop(self):
        """워커 스레드 루프"""
        while not self._shutdown_event.is_set():
            try:
                # 타임아웃으로 주기적으로 체크
                entry = self._log_queue.get(timeout=1.0)
                
                # 모든 옵저버에게 전달
                for observer in self._observers:
                    try:
                        observer.on_log(entry)
                    except Exception as e:
                        # 옵저버 실패는 다른 옵저버에 영향 주지 않음
                        print(f"로그 옵저버 오류: {e}", file=sys.stderr)
                        # 에러 로그에 기록
                        log_error(e, "UnifiedLogger._worker_loop", {"observer_type": type(observer).__name__})
                
                self._log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"로그 워커 오류: {e}", file=sys.stderr)
                # 에러 로그에 기록
                log_error(e, "UnifiedLogger._worker_loop")
    
    def set_session_id(self, session_id: str):
        """세션 ID 설정"""
        self._session_id = session_id
    
    def log(self, level: LogLevel, message: str, category: LogCategory, 
            module: str, execution_time_ms: Optional[float] = None,
            metadata: Optional[Dict[str, Any]] = None):
        """로그 기록"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            category=category,
            module=module,
            message=message,
            session_id=self._session_id,
            execution_time_ms=execution_time_ms,
            metadata=metadata
        )
        
        # 비동기 큐에 추가
        try:
            self._log_queue.put_nowait(entry)
        except queue.Full:
            # 큐가 가득 찬 경우 직접 처리
            for observer in self._observers:
                try:
                    observer.on_log(entry)
                except Exception:
                    pass
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
              module: str = "unknown", **kwargs):
        """디버그 로그"""
        self.log(LogLevel.DEBUG, message, category, module, **kwargs)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
             module: str = "unknown", **kwargs):
        """정보 로그"""
        self.log(LogLevel.INFO, message, category, module, **kwargs)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
                module: str = "unknown", **kwargs):
        """경고 로그"""
        self.log(LogLevel.WARNING, message, category, module, **kwargs)
    
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
              module: str = "unknown", **kwargs):
        """에러 로그"""
        self.log(LogLevel.ERROR, message, category, module, **kwargs)
    
    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
                 module: str = "unknown", **kwargs):
        """중요 로그"""
        self.log(LogLevel.CRITICAL, message, category, module, **kwargs)
    
    def performance(self, operation: str, execution_time_ms: float, 
                   module: str = "unknown", **kwargs):
        """성능 로그"""
        message = f"{operation} 완료"
        self.log(LogLevel.INFO, message, LogCategory.PERFORMANCE, module, 
                execution_time_ms=execution_time_ms, **kwargs)
    
    def shutdown(self):
        """로거 종료"""
        self._shutdown_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

# 전역 로거 인스턴스
unified_logger = UnifiedLogger()

# 편의 함수들
def log_debug(message: str, category: LogCategory = LogCategory.SYSTEM, 
              module: str = "unknown", **kwargs):
    unified_logger.debug(message, category, module, **kwargs)

def log_info(message: str, category: LogCategory = LogCategory.SYSTEM, 
             module: str = "unknown", **kwargs):
    unified_logger.info(message, category, module, **kwargs)

def log_warning(message: str, category: LogCategory = LogCategory.SYSTEM, 
                module: str = "unknown", **kwargs):
    unified_logger.warning(message, category, module, **kwargs)

def log_error(message: str, category: LogCategory = LogCategory.SYSTEM, 
              module: str = "unknown", **kwargs):
    unified_logger.error(message, category, module, **kwargs)

def log_performance(operation: str, execution_time_ms: float, 
                   module: str = "unknown", **kwargs):
    unified_logger.performance(operation, execution_time_ms, module, **kwargs)

# 성능 측정 데코레이터
def log_execution_time(category: LogCategory = LogCategory.PERFORMANCE, 
                      operation_name: Optional[str] = None):
    """실행 시간 로깅 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                log_performance(op_name, execution_time, func.__module__)
                
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                log_error(f"{op_name} 실행 실패: {e}", LogCategory.SYSTEM, 
                         func.__module__, execution_time_ms=execution_time)
                raise
        
        return wrapper
    return decorator

import sys
