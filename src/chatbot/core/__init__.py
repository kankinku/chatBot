"""
Core module - 핵심 비즈니스 로직

예외 처리, 타입 정의, 로깅 등 핵심 기능을 제공합니다.
"""

from .exceptions import (
    ChatbotException,
    ConfigurationError,
    EmbeddingError,
    RetrievalError,
    GenerationError,
    PreprocessingError,
    ChunkingError,
    PipelineError,
)
from .types import (
    Chunk,
    RetrievedSpan,
    Answer,
    QuestionAnalysis,
)
from .logger import get_logger, setup_logging

__all__ = [
    # Exceptions
    "ChatbotException",
    "ConfigurationError",
    "EmbeddingError",
    "RetrievalError",
    "GenerationError",
    "PreprocessingError",
    "ChunkingError",
    "PipelineError",
    # Types
    "Chunk",
    "RetrievedSpan",
    "Answer",
    "QuestionAnalysis",
    # Logger
    "get_logger",
    "setup_logging",
]

