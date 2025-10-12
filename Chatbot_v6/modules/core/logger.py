"""
Structured Logging

구조화된 JSON 로깅 시스템을 제공합니다.
Console.log 사용 금지 - 모든 로그는 이 시스템을 통해 기록됩니다.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler

from config.constants import StatusCode


class JSONFormatter(logging.Formatter):
    """JSON 형식 로그 포매터"""
    
    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON 형식으로 변환"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # 추가 컨텍스트 정보
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # 예외 정보
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }
        
        # 모듈 정보
        log_data["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """텍스트 형식 로그 포매터 (개발용)"""
    
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


class StructuredLogger:
    """
    구조화된 로거 래퍼
    
    모든 로그에 컨텍스트 정보를 자동으로 추가합니다.
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """로거 컨텍스트 설정"""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """로거 컨텍스트 초기화"""
        self._context.clear()
    
    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """내부 로그 메서드"""
        log_extra = {**self._context}
        if extra:
            log_extra.update(extra)
        
        self._logger.log(
            level,
            message,
            extra={"extra": log_extra},
            exc_info=exc_info,
        )
    
    def debug(self, message: str, **kwargs) -> None:
        """디버그 로그"""
        self._log(logging.DEBUG, message, extra=kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """정보 로그"""
        self._log(logging.INFO, message, extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """경고 로그"""
        self._log(logging.WARNING, message, extra=kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """에러 로그"""
        self._log(logging.ERROR, message, extra=kwargs, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """크리티컬 로그"""
        self._log(logging.CRITICAL, message, extra=kwargs, exc_info=exc_info)
    
    def success(self, message: str, **kwargs) -> None:
        """성공 로그 (info 레벨)"""
        kwargs["status"] = StatusCode.SUCCESS
        self.info(message, **kwargs)
    
    def failure(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """실패 로그 (error 레벨)"""
        kwargs["status"] = StatusCode.FAILURE
        self.error(message, exc_info=exc_info, **kwargs)
    
    def log_exception(self, exception: Exception, **kwargs) -> None:
        """예외 로깅"""
        from modules.core.exceptions import ChatbotException
        
        if isinstance(exception, ChatbotException):
            # 구조화된 예외
            self.error(
                exception.message,
                exc_info=True,
                error_code=exception.error_code,
                error_type=exception.__class__.__name__,
                **exception.details,
                **kwargs,
            )
        else:
            # 일반 예외
            self.error(
                str(exception),
                exc_info=True,
                error_type=exception.__class__.__name__,
                **kwargs,
            )


# Logger cache
_loggers: Dict[str, StructuredLogger] = {}


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_format: str = "json",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 3,
) -> None:
    """
    로깅 시스템 초기화
    
    Args:
        log_dir: 로그 디렉토리
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 로그 형식 ("json" or "text")
        max_bytes: 로그 파일 최대 크기
        backup_count: 백업 파일 개수
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Root logger 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 포매터 선택
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    # 파일 핸들러 (일반 로그)
    file_handler = RotatingFileHandler(
        log_path / "chatbot.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 파일 핸들러 (에러 로그)
    error_handler = RotatingFileHandler(
        log_path / "error.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # 콘솔 핸들러 (개발용)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if log_level == "DEBUG" else logging.INFO)
    console_handler.setFormatter(formatter if log_format == "json" else TextFormatter())
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> StructuredLogger:
    """
    구조화된 로거 가져오기
    
    Args:
        name: 로거 이름 (일반적으로 __name__ 사용)
    
    Returns:
        StructuredLogger 인스턴스
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", question="고산 정수장 URL은?", status=0)
        >>> logger.error("LLM timeout", error_code="E302", exc_info=True)
    """
    if name not in _loggers:
        python_logger = logging.getLogger(name)
        _loggers[name] = StructuredLogger(python_logger)
    return _loggers[name]

