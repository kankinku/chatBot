"""
데이터베이스 관련 모듈

이 패키지는 SQL 생성, 데이터베이스 연결, 쿼리 실행 등을 담당합니다.
Enhanced SQL System을 통해 더 나은 성능과 정확도를 제공합니다.
"""

# 기존 모듈들 (하위 호환성)
from .real_database_executor import RealDatabaseExecutor
from .sql_generator import SQLGenerator
from .sql_element_extractor import SQLElementExtractor
from .sql_slot_extractor import SQLSlotExtractor

# 새로운 Enhanced SQL System
try:
    from .enhanced_sql_system import EnhancedSQLSystem, EnhancedSQLResult
    from .hybrid_sql_generator import HybridSQLGenerator, SQLGenerationResult
    from .dynamic_schema_manager import DynamicSchemaManager
    from .enhanced_information_extractor import EnhancedInformationExtractor
    from .performance_monitor import PerformanceMonitor, get_performance_monitor
    ENHANCED_SQL_AVAILABLE = True
except ImportError as e:
    ENHANCED_SQL_AVAILABLE = False
    import logging
    logging.warning(f"Enhanced SQL System을 로드할 수 없습니다: {e}")

__all__ = [
    # 기존 모듈들
    'RealDatabaseExecutor',
    'SQLGenerator',
    'SQLElementExtractor',
    'SQLSlotExtractor'
]

# Enhanced SQL System이 사용 가능한 경우 추가
if ENHANCED_SQL_AVAILABLE:
    __all__.extend([
        'EnhancedSQLSystem',
        'EnhancedSQLResult',
        'HybridSQLGenerator',
        'SQLGenerationResult',
        'DynamicSchemaManager',
        'EnhancedInformationExtractor',
        'PerformanceMonitor',
        'get_performance_monitor'
    ])
