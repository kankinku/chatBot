"""
강화된 SQL 시스템 - 통합 모듈

기존 SQL 생성기를 대체하는 새로운 하이브리드 시스템의 통합 인터페이스
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time

# 새로운 모듈들 import
from core.database.enhanced_information_extractor import EnhancedInformationExtractor, ExtractionResult
from core.database.hybrid_sql_generator import HybridSQLGenerator, SQLGenerationResult, GenerationMethod
from core.database.dynamic_schema_manager import DynamicSchemaManager, DatabaseSchema as NewDatabaseSchema
from core.database.performance_monitor import PerformanceMonitor, get_performance_monitor

# 기존 모듈과의 호환성을 위한 import
try:
    from core.database.sql_generator import SQLQuery, DatabaseSchema as OldDatabaseSchema
    from core.database.real_database_executor import RealDatabaseExecutor
    LEGACY_MODULES_AVAILABLE = True
except ImportError:
    LEGACY_MODULES_AVAILABLE = False
    logging.warning("기존 SQL 모듈들을 찾을 수 없습니다.")

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSQLResult:
    """강화된 SQL 결과"""
    # 기본 결과
    query: str
    is_valid: bool
    confidence_score: float
    execution_time: float
    
    # 강화된 정보
    generation_method: str
    extraction_confidence: float
    entity_count: int
    intent: str
    
    # 호환성을 위한 필드들
    query_type: str = "SELECT"
    model_name: str = "enhanced_hybrid"
    validation_passed: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EnhancedSQLSystem:
    """
    강화된 SQL 시스템
    
    기존 SQLGenerator를 대체하는 새로운 시스템:
    1. 강화된 정보 추출
    2. 하이브리드 SQL 생성
    3. 동적 스키마 관리
    4. 성능 모니터링
    5. 기존 API와의 호환성
    """
    
    def __init__(self, 
                 enable_legacy_fallback: bool = True,
                 enable_performance_monitoring: bool = True):
        """강화된 SQL 시스템 초기화"""
        
        self.enable_legacy_fallback = enable_legacy_fallback
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # 새로운 컴포넌트들
        self.information_extractor = EnhancedInformationExtractor()
        self.hybrid_generator = HybridSQLGenerator(use_legacy_llm=enable_legacy_fallback)
        
        # 동적 스키마 관리자 (옵션)
        self.schema_manager = None
        try:
            self.schema_manager = DynamicSchemaManager()
            logger.info("동적 스키마 관리자 활성화")
        except Exception as e:
            logger.warning(f"동적 스키마 관리자 초기화 실패: {e}")
        
        # 성능 모니터링
        self.performance_monitor = None
        if enable_performance_monitoring:
            self.performance_monitor = get_performance_monitor()
            logger.info("성능 모니터링 활성화")
        
        # 기존 모듈들 (폴백용)
        self.legacy_executor = None
        if LEGACY_MODULES_AVAILABLE:
            try:
                self.legacy_executor = RealDatabaseExecutor()
                logger.info("기존 데이터베이스 실행기 로드")
            except Exception as e:
                logger.warning(f"기존 데이터베이스 실행기 로드 실패: {e}")
        
        logger.info("강화된 SQL 시스템 초기화 완료")
    
    def generate_sql(self, 
                    question: str, 
                    schema: Optional[Union[OldDatabaseSchema, NewDatabaseSchema]] = None,
                    few_shot_examples: List[Dict[str, str]] = None) -> Union[SQLQuery, EnhancedSQLResult]:
        """
        SQL 생성 (기존 API 호환)
        
        Args:
            question: 자연어 질문
            schema: 데이터베이스 스키마 (옵션)
            few_shot_examples: Few-shot 예시들 (현재 미사용)
            
        Returns:
            생성된 SQL 쿼리 결과
        """
        start_time = time.time()
        
        logger.info(f"강화된 SQL 생성 시작: {question}")
        
        try:
            # 1. 정보 추출
            extraction_result = self.information_extractor.extract_information(question)
            
            # 2. 하이브리드 SQL 생성
            generation_result = self.hybrid_generator.generate_sql(question, schema)
            
            # 3. 결과 생성
            enhanced_result = EnhancedSQLResult(
                query=generation_result.query,
                is_valid=generation_result.is_valid,
                confidence_score=generation_result.confidence,
                execution_time=generation_result.execution_time,
                generation_method=generation_result.method.value,
                extraction_confidence=extraction_result.confidence,
                entity_count=len(extraction_result.entities),
                intent=extraction_result.intent.value,
                query_type=self._infer_query_type(generation_result.query),
                error_message=generation_result.error_message,
                metadata={
                    'extraction_result': {
                        'intent': extraction_result.intent.value,
                        'entities': [
                            {
                                'type': e.entity_type.value,
                                'value': str(e.value),
                                'confidence': e.confidence
                            } for e in extraction_result.entities
                        ],
                        'reasoning': extraction_result.reasoning
                    },
                    'generation_metadata': generation_result.metadata
                }
            )
            
            # 4. 성능 모니터링
            if self.performance_monitor:
                self.performance_monitor.log_sql_generation(
                    question=question,
                    generated_sql=generation_result.query,
                    method=generation_result.method.value,
                    execution_time=generation_result.execution_time,
                    confidence=generation_result.confidence,
                    success=generation_result.is_valid,
                    error_message=generation_result.error_message,
                    entity_count=len(extraction_result.entities),
                    cache_hit=generation_result.method == GenerationMethod.RULE_BASED,
                    metadata={
                        'intent': extraction_result.intent.value,
                        'extraction_confidence': extraction_result.confidence
                    }
                )
            
            total_time = time.time() - start_time
            logger.info(f"강화된 SQL 생성 완료: {generation_result.method.value}, {total_time:.3f}초")
            
            return enhanced_result
            
        except Exception as e:
            error_msg = f"강화된 SQL 생성 실패: {e}"
            logger.error(error_msg)
            
            # 성능 모니터링 (오류)
            if self.performance_monitor:
                self.performance_monitor.log_sql_generation(
                    question=question,
                    generated_sql="-- SQL 생성 실패",
                    method="error",
                    execution_time=time.time() - start_time,
                    confidence=0.0,
                    success=False,
                    error_message=str(e),
                    entity_count=0
                )
            
            return EnhancedSQLResult(
                query="-- SQL 생성 실패",
                is_valid=False,
                confidence_score=0.0,
                execution_time=time.time() - start_time,
                generation_method="error",
                extraction_confidence=0.0,
                entity_count=0,
                intent="unknown",
                error_message=error_msg
            )
    
    def execute_sql(self, sql_query: Union[SQLQuery, EnhancedSQLResult, str]) -> Dict[str, Any]:
        """
        SQL 실행 (기존 API 호환)
        
        Args:
            sql_query: SQL 쿼리 (문자열 또는 쿼리 객체)
            
        Returns:
            실행 결과
        """
        if not self.legacy_executor:
            return {
                'success': False,
                'error': '데이터베이스 실행기를 사용할 수 없습니다.',
                'data': None
            }
        
        # SQL 문자열 추출
        if isinstance(sql_query, str):
            sql_string = sql_query
        elif hasattr(sql_query, 'query'):
            sql_string = sql_query.query
        else:
            return {
                'success': False,
                'error': '유효하지 않은 SQL 쿼리 형식입니다.',
                'data': None
            }
        
        # 유효성 검사
        if not sql_string or sql_string.strip().startswith('--'):
            return {
                'success': False,
                'error': 'SQL 쿼리가 유효하지 않습니다.',
                'data': None
            }
        
        try:
            result = self.legacy_executor.execute_sql(sql_string)
            return result
        except Exception as e:
            logger.error(f"SQL 실행 실패: {e}")
            return {
                'success': False,
                'error': f'SQL 실행 중 오류 발생: {str(e)}',
                'data': None
            }
    
    def test_database_connection(self) -> Dict[str, Any]:
        """데이터베이스 연결 테스트"""
        if not self.legacy_executor:
            return {
                'success': False,
                'error': '데이터베이스 실행기를 사용할 수 없습니다.'
            }
        
        try:
            return self.legacy_executor.test_connection()
        except Exception as e:
            logger.error(f"데이터베이스 연결 테스트 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_database_schema(self) -> List[Dict[str, Any]]:
        """데이터베이스 스키마 조회"""
        if self.schema_manager:
            try:
                schema = self.schema_manager.get_schema()
                return self._convert_schema_to_legacy_format(schema)
            except Exception as e:
                logger.warning(f"동적 스키마 조회 실패: {e}")
        
        # 폴백: 기존 방식
        if self.legacy_executor:
            try:
                if hasattr(self.legacy_executor, 'get_database_schema'):
                    return self.legacy_executor.get_database_schema()
            except Exception as e:
                logger.error(f"기존 스키마 조회 실패: {e}")
        
        return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 조회"""
        if not self.performance_monitor:
            return {'monitoring_enabled': False}
        
        try:
            current_stats = self.performance_monitor.get_current_stats()
            real_time_metrics = self.performance_monitor.get_real_time_metrics()
            method_comparison = self.performance_monitor.get_method_comparison()
            
            return {
                'monitoring_enabled': True,
                'current_stats': {
                    'total_requests': current_stats.total_requests,
                    'successful_requests': current_stats.successful_requests,
                    'failed_requests': current_stats.failed_requests,
                    'avg_response_time': current_stats.avg_response_time,
                    'avg_confidence': current_stats.avg_confidence,
                    'cache_hit_count': current_stats.cache_hit_count,
                    'method_counts': current_stats.method_counts,
                    'error_types': current_stats.error_types
                },
                'real_time_metrics': real_time_metrics,
                'method_comparison': method_comparison
            }
        except Exception as e:
            logger.error(f"성능 통계 조회 실패: {e}")
            return {
                'monitoring_enabled': True,
                'error': str(e)
            }
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """성능 요약 조회"""
        if not self.performance_monitor:
            return {'monitoring_enabled': False}
        
        try:
            return self.performance_monitor.get_performance_summary(hours)
        except Exception as e:
            logger.error(f"성능 요약 조회 실패: {e}")
            return {
                'monitoring_enabled': True,
                'error': str(e)
            }
    
    def refresh_schema(self):
        """스키마 갱신"""
        if self.schema_manager:
            try:
                self.schema_manager.refresh_schema()
                logger.info("스키마 갱신 완료")
            except Exception as e:
                logger.error(f"스키마 갱신 실패: {e}")
        else:
            logger.warning("동적 스키마 관리자가 비활성화되어 있습니다.")
    
    def clear_cache(self):
        """캐시 초기화"""
        if self.schema_manager:
            self.schema_manager.clear_cache()
        
        if self.performance_monitor:
            self.performance_monitor.reset_stats()
        
        logger.info("캐시 초기화 완료")
    
    def _infer_query_type(self, sql: str) -> str:
        """SQL 쿼리 타입 추론"""
        sql_upper = sql.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            if 'COUNT(' in sql_upper:
                return 'COUNT'
            elif any(func in sql_upper for func in ['SUM(', 'AVG(', 'MAX(', 'MIN(']):
                return 'AGGREGATE'
            else:
                return 'SELECT'
        elif sql_upper.startswith('INSERT'):
            return 'INSERT'
        elif sql_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif sql_upper.startswith('DELETE'):
            return 'DELETE'
        else:
            return 'UNKNOWN'
    
    def _convert_schema_to_legacy_format(self, schema: NewDatabaseSchema) -> List[Dict[str, Any]]:
        """새로운 스키마 형식을 기존 형식으로 변환"""
        legacy_schema = []
        
        for table_name, table_info in schema.tables.items():
            columns = []
            for col_name, col_info in table_info.columns.items():
                columns.append({
                    'name': col_info.name,
                    'type': col_info.type,
                    'null': 'YES' if col_info.is_nullable else 'NO',
                    'key': 'PRI' if col_info.is_primary_key else ('MUL' if col_info.is_foreign_key else ''),
                    'default': col_info.default_value,
                    'extra': ''
                })
            
            # 샘플 데이터 생성 (각 컬럼의 샘플 값들을 조합)
            sample_data = []
            if table_info.columns:
                max_samples = 3
                for i in range(max_samples):
                    sample_row = {}
                    for col_name, col_info in table_info.columns.items():
                        if col_info.sample_values and i < len(col_info.sample_values):
                            sample_row[col_name] = col_info.sample_values[i]
                        else:
                            sample_row[col_name] = None
                    if any(v is not None for v in sample_row.values()):
                        sample_data.append(sample_row)
            
            legacy_schema.append({
                'table_name': table_name,
                'columns': columns,
                'sample_data': sample_data
            })
        
        return legacy_schema

# 기존 SQLGenerator와의 호환성을 위한 별칭
class SQLGenerator(EnhancedSQLSystem):
    """기존 SQLGenerator와의 호환성을 위한 별칭"""
    pass

# 전역 인스턴스 생성 함수
def create_enhanced_sql_system(**kwargs) -> EnhancedSQLSystem:
    """강화된 SQL 시스템 인스턴스 생성"""
    return EnhancedSQLSystem(**kwargs)

if __name__ == "__main__":
    # 테스트 코드
    system = EnhancedSQLSystem()
    
    test_questions = [
        "조치원읍 교차로가 몇 개인가요?",
        "지난주 한솔동 교통량 평균은?",
        "상위 10개 지역의 교통사고를 보여줘",
        "어제 가장 많은 교통량이 발생한 곳은?",
        "이번달 새롬동 교통량을 알려주세요"
    ]
    
    print("=== 강화된 SQL 시스템 테스트 ===\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. 질문: {question}")
        
        result = system.generate_sql(question)
        
        print(f"   방법: {result.generation_method}")
        print(f"   실행 시간: {result.execution_time:.3f}초")
        print(f"   신뢰도: {result.confidence_score:.3f}")
        print(f"   의도: {result.intent}")
        print(f"   엔티티 수: {result.entity_count}")
        print(f"   SQL: {result.query}")
        
        if result.error_message:
            print(f"   오류: {result.error_message}")
        
        print()
    
    # 성능 통계 출력
    print("=== 성능 통계 ===")
    stats = system.get_performance_stats()
    if stats.get('monitoring_enabled'):
        current = stats['current_stats']
        print(f"총 요청: {current['total_requests']}")
        print(f"성공 요청: {current['successful_requests']}")
        print(f"평균 응답 시간: {current['avg_response_time']:.3f}초")
        print(f"평균 신뢰도: {current['avg_confidence']:.3f}")
        print(f"방법별 사용: {current['method_counts']}")
        
        if stats['method_comparison']:
            print(f"\n방법별 성능 비교:")
            for method, metrics in stats['method_comparison'].items():
                print(f"  {method}: 성공률 {metrics['success_rate']:.2%}, 응답시간 {metrics['avg_response_time']:.3f}초")
    else:
        print("성능 모니터링이 비활성화되어 있습니다.")
    
    # 데이터베이스 연결 테스트
    print(f"\n=== 데이터베이스 연결 테스트 ===")
    connection_result = system.test_database_connection()
    if connection_result['success']:
        print("데이터베이스 연결 성공")
        if 'details' in connection_result:
            details = connection_result['details']
            print(f"사용 가능한 테이블: {details.get('available_main_tables', [])}")
    else:
        print(f"데이터베이스 연결 실패: {connection_result.get('error', 'Unknown error')}")
