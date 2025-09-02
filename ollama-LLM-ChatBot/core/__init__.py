"""
경량화된 챗봇 코어 모듈

기능별로 분류된 경량화된 모듈들을 제공합니다.
"""

# LLM 관련 모듈
from .llm import AnswerGenerator, LocalLLMInterface, DataAnalysisGenerator, TrafficAnalysisHandler

# 데이터베이스 관련 모듈
from .database import RealDatabaseExecutor, SQLGenerator, SQLElementExtractor, SQLSlotExtractor

# 문서 처리 관련 모듈
from .document import PDFPreprocessor, PDFProcessor, VectorStore

# 쿼리 처리 관련 모듈
from .query import QueryRouter, QuestionAnalyzer, TimeBasedQueryHandler

# 캐시 관련 모듈
from .cache import FastCache

# 이동 분석 관련 모듈
from .movement import MovementHandler

__all__ = [
    # LLM
    'AnswerGenerator',
    'LocalLLMInterface',
    'DataAnalysisGenerator', 
    'TrafficAnalysisHandler',
    
    # Database
    'RealDatabaseExecutor',
    'SQLGenerator',
    'SQLElementExtractor',
    'SQLSlotExtractor',
    
    # Document
    'PDFPreprocessor',
    'PDFProcessor',
    'VectorStore',
    
    # Query
    'QueryRouter',
    'QuestionAnalyzer',
    'TimeBasedQueryHandler',
    
    # Cache
    'FastCache',
    
    # Movement
    'MovementHandler'
]
