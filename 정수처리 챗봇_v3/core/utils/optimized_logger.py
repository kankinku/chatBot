"""
최적화된 로깅 시스템

성능을 고려한 로깅 모듈로 다음 기능을 제공:
- 비동기 로깅
- 로그 레벨별 최적화
- 메모리 효율적인 로그 버퍼링
- 자동 로그 로테이션
"""

import logging
import logging.handlers
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import threading
from queue import Queue, Empty
import json

class OptimizedLogger:
    """
    성능 최적화된 로거 클래스
    
    특징:
    - 비동기 로깅으로 메인 스레드 블로킹 방지
    - 구조화된 로깅 (JSON 형태)
    - 자동 로그 로테이션
    - 메모리 효율적인 버퍼링
    """
    
    def __init__(self, name: str, log_dir: str = "./logs", max_size: int = 10*1024*1024):
        """
        OptimizedLogger 초기화
        
        Args:
            name: 로거 이름
            log_dir: 로그 디렉토리
            max_size: 최대 로그 파일 크기 (바이트)
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 로그 큐 및 워커 스레드
        self.log_queue = Queue(maxsize=1000)
        self.worker_thread = None
        self.shutdown_flag = threading.Event()
        
        # 로거 설정
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 핸들러 설정
        self._setup_handlers(max_size)
        
        # 워커 스레드 시작
        self._start_worker()
        
    def _setup_handlers(self, max_size: int):
        """로그 핸들러 설정"""
        # 파일 핸들러 (로테이션 지원)
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=max_size,
            backupCount=5,
            encoding='utf-8'
        )
        
        # JSON 포맷터
        json_formatter = JsonFormatter()
        file_handler.setFormatter(json_formatter)
        
        # 콘솔 핸들러 (개발용)
        if os.getenv("DEBUG", "false").lower() == "true":
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.addHandler(file_handler)
    
    def _start_worker(self):
        """비동기 로깅 워커 스레드 시작"""
        self.worker_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.worker_thread.start()
    
    def _log_worker(self):
        """로그 워커 스레드 메인 루프"""
        while not self.shutdown_flag.is_set():
            try:
                # 큐에서 로그 메시지 가져오기
                log_record = self.log_queue.get(timeout=1.0)
                if log_record is None:  # 종료 신호
                    break
                
                # 실제 로깅 수행
                self.logger.handle(log_record)
                self.log_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                # 워커 스레드에서 오류 발생시 콘솔에 출력
                print(f"로그 워커 오류: {e}")
    
    def _log_async(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """비동기 로깅 메서드"""
        try:
            # 로그 레코드 생성
            record = self.logger.makeRecord(
                name=self.name,
                level=level,
                fn="",
                lno=0,
                msg=message,
                args=(),
                exc_info=None,
                extra=extra or {}
            )
            
            # 큐에 추가 (논블로킹)
            if not self.log_queue.full():
                self.log_queue.put_nowait(record)
            else:
                # 큐가 가득 찬 경우 가장 오래된 메시지 제거
                try:
                    self.log_queue.get_nowait()
                    self.log_queue.put_nowait(record)
                except Empty:
                    pass
                    
        except Exception as e:
            # 비동기 로깅 실패시 동기 로깅으로 폴백
            print(f"비동기 로깅 실패, 동기 로깅 사용: {e}")
            self.logger.log(level, message, extra=extra or {})
    
    def info(self, message: str, **kwargs):
        """정보 레벨 로깅"""
        self._log_async(logging.INFO, message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """경고 레벨 로깅"""
        self._log_async(logging.WARNING, message, kwargs)
    
    def error(self, message: str, **kwargs):
        """오류 레벨 로깅"""
        self._log_async(logging.ERROR, message, kwargs)
    
    def debug(self, message: str, **kwargs):
        """디버그 레벨 로깅"""
        if self.logger.isEnabledFor(logging.DEBUG):
            self._log_async(logging.DEBUG, message, kwargs)
    
    def performance(self, operation: str, duration: float, **kwargs):
        """성능 측정 로깅"""
        self._log_async(logging.INFO, f"Performance: {operation}", {
            'operation': operation,
            'duration_ms': round(duration * 1000, 2),
            'category': 'performance',
            **kwargs
        })
    
    def shutdown(self):
        """로거 종료"""
        self.shutdown_flag.set()
        
        # 남은 로그 처리
        if self.worker_thread and self.worker_thread.is_alive():
            self.log_queue.put(None)  # 종료 신호
            self.worker_thread.join(timeout=5.0)
        
        # 핸들러 정리
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

class JsonFormatter(logging.Formatter):
    """JSON 형태의 로그 포맷터"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 추가 필드 포함
        if hasattr(record, 'operation'):
            log_data['operation'] = record.operation
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        if hasattr(record, 'category'):
            log_data['category'] = record.category
        
        # 예외 정보 포함
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)

class PerformanceTimer:
    """성능 측정을 위한 컨텍스트 매니저"""
    
    def __init__(self, logger: OptimizedLogger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.performance(self.operation, duration, **self.kwargs)
        else:
            self.logger.error(
                f"Operation failed: {self.operation}",
                operation=self.operation,
                duration_ms=round(duration * 1000, 2),
                error=str(exc_val),
                **self.kwargs
            )

# 전역 로거 인스턴스들
_loggers: Dict[str, OptimizedLogger] = {}

def get_optimized_logger(name: str) -> OptimizedLogger:
    """최적화된 로거 인스턴스 반환"""
    if name not in _loggers:
        _loggers[name] = OptimizedLogger(name)
    return _loggers[name]

def shutdown_all_loggers():
    """모든 로거 종료"""
    for logger in _loggers.values():
        logger.shutdown()
    _loggers.clear()

# 편의 함수들
def log_performance(operation: str):
    """성능 측정 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_optimized_logger(func.__module__)
            with PerformanceTimer(logger, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # 테스트 코드
    logger = get_optimized_logger("test")
    
    # 기본 로깅 테스트
    logger.info("테스트 메시지", user_id=123, action="login")
    logger.warning("경고 메시지", error_code="W001")
    logger.error("오류 메시지", error_code="E001", details="상세 정보")
    
    # 성능 측정 테스트
    with PerformanceTimer(logger, "test_operation", component="test"):
        time.sleep(0.1)
    
    # 종료
    logger.shutdown()
    print("로거 테스트 완료")
