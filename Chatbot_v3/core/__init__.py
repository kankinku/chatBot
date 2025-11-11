"""
Core 모듈 패키지

모든 핵심 기능을 중앙에서 관리하는 패키지
디자인 패턴 적용: Singleton, Factory, Observer
"""

# 통합 시스템 import
from .config.unified_config import config, get_config, set_config
from .utils.singleton_manager import singleton_manager, get_embedding_model, get_cache

# 기존 모듈들
from .document.pdf_processor import PDFProcessor, TextChunk
from .document.vector_store import HybridVectorStore
from .query.query_router import QueryRouter, QueryRoute
from .query.question_analyzer import QuestionAnalyzer, AnalyzedQuestion
from .llm.answer_generator import AnswerGenerator, Answer
from .database.sql_generator import SQLGenerator, DatabaseSchema, SQLQuery
from .cache.fast_cache import FastCache
from .utils.memory_optimizer import MemoryOptimizer, model_memory_manager

# 법률 모듈
from .legal import (
    LegalRouter, LegalMode, LegalResponse,
    create_default_router, create_optimized_router
)

__version__ = "2.0.0"

__all__ = [
    # 통합 시스템
    'config',
    'get_config', 
    'set_config',
    'singleton_manager',
    'get_embedding_model',
    'get_cache',
    
    # 핵심 모듈
    'PDFProcessor',
    'TextChunk',
    'HybridVectorStore',
    'QueryRouter',
    'QueryRoute',
    'QuestionAnalyzer',
    'AnalyzedQuestion',
    'AnswerGenerator',
    'Answer',
    'SQLGenerator',
    'DatabaseSchema',
    'SQLQuery',
    'FastCache',
    'MemoryOptimizer',
    'model_memory_manager',
    
    # 법률 모듈
    'LegalRouter',
    'LegalMode',
    'LegalResponse',
    'create_default_router',
    'create_optimized_router',
    
    # 메타
    '__version__'
]

# 초기화 로그
import logging
logging.getLogger(__name__).info("Core 패키지 초기화 완료 (통합 시스템 적용)")