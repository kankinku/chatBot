"""
Utils 패키지

유틸리티 함수들과 통합 시스템을 제공하는 패키지
"""

from .unified_logger import (
    unified_logger, 
    log_info, log_debug, log_warning, log_error, log_performance,
    LogCategory, LogLevel, log_execution_time
)
from .chatbot_logger import chatbot_logger, QuestionType, ProcessingStep
from .performance_monitor import PerformanceMonitor
from .log_viewer import ChatbotLogViewer
# FileManager는 현재 사용하지 않음 (import 오류 방지)

__version__ = "2.0.0"

__all__ = [
    # 통합 로깅
    'unified_logger',
    'log_info',
    'log_debug', 
    'log_warning',
    'log_error',
    'log_performance',
    'LogCategory',
    'LogLevel',
    'log_execution_time',
    
    # 기존 유틸리티
    'chatbot_logger',
    'QuestionType',
    'ProcessingStep',
    'PerformanceMonitor',
    'ChatbotLogViewer',
    
    # 메타
    '__version__'
]

# 초기화 로그
import logging
logging.getLogger(__name__).info("Utils 패키지 초기화 완료 (통합 로깅 시스템 적용)")