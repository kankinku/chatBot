"""
하이브리드 SQL 생성기 - 규칙 기반 + LLM 백업

빠른 규칙 기반 처리를 우선하고, 복잡한 경우에만 LLM을 사용하는 하이브리드 접근법
"""

import os
import re
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# 강화된 정보 추출기 import
from core.database.enhanced_information_extractor import (
    EnhancedInformationExtractor, 
    ExtractionResult, 
    EntityType, 
    IntentType
)

# 기존 SQL 생성기 import (백업용)
try:
    from core.database.sql_generator import SQLGenerator, SQLQuery, DatabaseSchema
    LEGACY_SQL_AVAILABLE = True
except ImportError:
    LEGACY_SQL_AVAILABLE = False
    logging.warning("기존 SQL 생성기를 찾을 수 없습니다.")

# 동적 스키마 관리자 import
try:
    from core.database.dynamic_schema_manager import DynamicSchemaManager
    DYNAMIC_SCHEMA_AVAILABLE = True
except ImportError:
    DYNAMIC_SCHEMA_AVAILABLE = False
    logging.warning("동적 스키마 관리자를 찾을 수 없습니다.")

logger = logging.getLogger(__name__)

class GenerationMethod(Enum):
    """SQL 생성 방법"""
    RULE_BASED = "rule_based"       # 규칙 기반
    TEMPLATE_BASED = "template_based"  # 템플릿 기반
    LLM_BASED = "llm_based"         # LLM 기반
    HYBRID = "hybrid"               # 하이브리드

@dataclass
class SQLGenerationResult:
    """SQL 생성 결과"""
    query: str
    method: GenerationMethod
    confidence: float
    execution_time: float
    is_valid: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class RuleBasedSQLEngine:
    """규칙 기반 SQL 엔진"""
    
    def __init__(self):
        """규칙 기반 SQL 엔진 초기화"""
        # 기본 테이블 정보
        self.default_tables = {
            'traffic_intersection': {
                'columns': ['id', 'name', 'latitude', 'longitude', 'created_at', 'updated_at'],
                'primary_key': 'id'
            },
            'traffic_trafficvolume': {
                'columns': ['id', 'intersection_id', 'datetime', 'direction', 'volume', 'is_simulated', 'created_at', 'updated_at'],
                'primary_key': 'id',
                'foreign_keys': {'intersection_id': 'traffic_intersection.id'}
            },
            'traffic_incident': {
                'columns': ['incident_id', 'incident_type', 'intersection_id', 'district', 'intersection_name', 'status', 'registered_at', 'created_at', 'updated_at'],
                'primary_key': 'incident_id'
            }
        }
        
        # SQL 템플릿
        self.sql_templates = {
            # 개수 조회 템플릿
            'count_basic': "SELECT COUNT(*) FROM {table}",
            'count_with_condition': "SELECT COUNT(*) FROM {table} WHERE {condition}",
            'count_with_location': "SELECT COUNT(*) FROM {table} WHERE name LIKE '%{location}%'",
            
            # 집계 조회 템플릿
            'aggregate_basic': "SELECT {function}({column}) FROM {table}",
            'aggregate_with_condition': "SELECT {function}({column}) FROM {table} WHERE {condition}",
            'aggregate_with_location': "SELECT {function}({column}) FROM {table} WHERE name LIKE '%{location}%'",
            
            # 목록 조회 템플릿
            'list_basic': "SELECT {columns} FROM {table}",
            'list_with_condition': "SELECT {columns} FROM {table} WHERE {condition}",
            'list_with_location': "SELECT {columns} FROM {table} WHERE name LIKE '%{location}%'",
            'list_with_limit': "SELECT {columns} FROM {table} ORDER BY {order_column} {order_direction} LIMIT {limit}",
            
            # 순위 조회 템플릿
            'ranking_basic': "SELECT {columns} FROM {table} ORDER BY {order_column} {order_direction} LIMIT {limit}",
            'ranking_with_condition': "SELECT {columns} FROM {table} WHERE {condition} ORDER BY {order_column} {order_direction} LIMIT {limit}",
            
            # 조인 템플릿
            'join_basic': "SELECT {columns} FROM {table1} t1 JOIN {table2} t2 ON t1.{join_column1} = t2.{join_column2}",
            'join_with_condition': "SELECT {columns} FROM {table1} t1 JOIN {table2} t2 ON t1.{join_column1} = t2.{join_column2} WHERE {condition}",
        }
        
        logger.info("규칙 기반 SQL 엔진 초기화 완료")
    
    def generate_sql(self, extraction_result: ExtractionResult) -> Optional[SQLGenerationResult]:
        """
        규칙 기반 SQL 생성
        
        Args:
            extraction_result: 정보 추출 결과
            
        Returns:
            SQL 생성 결과 (생성 불가능하면 None)
        """
        start_time = time.time()
        
        try:
            # 1. 테이블 결정
            table_name = self._determine_table(extraction_result)
            if not table_name:
                return None
            
            # 2. 의도별 SQL 생성
            if extraction_result.intent == IntentType.COUNT:
                sql = self._generate_count_sql(extraction_result, table_name)
            elif extraction_result.intent == IntentType.AGGREGATE:
                sql = self._generate_aggregate_sql(extraction_result, table_name)
            elif extraction_result.intent == IntentType.LIST:
                sql = self._generate_list_sql(extraction_result, table_name)
            elif extraction_result.intent == IntentType.RANKING:
                sql = self._generate_ranking_sql(extraction_result, table_name)
            else:
                return None
            
            if not sql:
                return None
            
            execution_time = time.time() - start_time
            
            return SQLGenerationResult(
                query=sql,
                method=GenerationMethod.RULE_BASED,
                confidence=0.85,
                execution_time=execution_time,
                is_valid=True,
                metadata={
                    'table_name': table_name,
                    'intent': extraction_result.intent.value,
                    'entity_count': len(extraction_result.entities)
                }
            )
            
        except Exception as e:
            logger.warning(f"규칙 기반 SQL 생성 실패: {e}")
            return None
    
    def _determine_table(self, extraction_result: ExtractionResult) -> Optional[str]:
        """테이블 결정"""
        # 테이블 힌트가 있으면 우선 사용
        for entity in extraction_result.entities:
            if entity.entity_type == EntityType.TABLE_HINT:
                return entity.value
        
        # 키워드 기반 테이블 결정
        question = extraction_result.processed_question.lower()
        
        if any(keyword in question for keyword in ['교차로', 'intersection']):
            return 'traffic_intersection'
        elif any(keyword in question for keyword in ['교통량', '통행량', 'volume', 'traffic']):
            return 'traffic_trafficvolume'
        elif any(keyword in question for keyword in ['사고', 'incident', '사건']):
            return 'traffic_incident'
        
        return 'traffic_intersection'  # 기본값
    
    def _generate_count_sql(self, extraction_result: ExtractionResult, table_name: str) -> Optional[str]:
        """개수 조회 SQL 생성"""
        # 위치 조건 확인
        location_condition = self._build_location_condition(extraction_result, table_name)
        time_condition = self._build_time_condition(extraction_result, table_name)
        
        conditions = []
        if location_condition:
            conditions.append(location_condition)
        if time_condition:
            conditions.append(time_condition)
        
        if conditions:
            condition = " AND ".join(conditions)
            return self.sql_templates['count_with_condition'].format(
                table=table_name,
                condition=condition
            )
        else:
            return self.sql_templates['count_basic'].format(table=table_name)
    
    def _generate_aggregate_sql(self, extraction_result: ExtractionResult, table_name: str) -> Optional[str]:
        """집계 조회 SQL 생성"""
        # 집계 함수 찾기
        agg_function = None
        for entity in extraction_result.entities:
            if entity.entity_type == EntityType.AGGREGATION:
                agg_function = entity.value
                break
        
        if not agg_function:
            return None
        
        # 집계 대상 컬럼 결정
        if table_name == 'traffic_trafficvolume':
            column = 'volume'
        elif table_name == 'traffic_incident':
            column = '*'
        else:
            column = '*'
        
        # 조건 생성
        location_condition = self._build_location_condition(extraction_result, table_name)
        time_condition = self._build_time_condition(extraction_result, table_name)
        
        conditions = []
        if location_condition:
            conditions.append(location_condition)
        if time_condition:
            conditions.append(time_condition)
        
        if conditions:
            condition = " AND ".join(conditions)
            return self.sql_templates['aggregate_with_condition'].format(
                function=agg_function,
                column=column,
                table=table_name,
                condition=condition
            )
        else:
            return self.sql_templates['aggregate_basic'].format(
                function=agg_function,
                column=column,
                table=table_name
            )
    
    def _generate_list_sql(self, extraction_result: ExtractionResult, table_name: str) -> Optional[str]:
        """목록 조회 SQL 생성"""
        # 컬럼 결정
        columns = self._determine_columns(extraction_result, table_name)
        
        # 조건 생성
        location_condition = self._build_location_condition(extraction_result, table_name)
        time_condition = self._build_time_condition(extraction_result, table_name)
        
        conditions = []
        if location_condition:
            conditions.append(location_condition)
        if time_condition:
            conditions.append(time_condition)
        
        # 제한 조건 확인
        limit = self._get_limit(extraction_result)
        
        if conditions:
            condition = " AND ".join(conditions)
            sql = self.sql_templates['list_with_condition'].format(
                columns=columns,
                table=table_name,
                condition=condition
            )
        else:
            sql = self.sql_templates['list_basic'].format(
                columns=columns,
                table=table_name
            )
        
        # LIMIT 추가
        if limit:
            sql += f" LIMIT {limit}"
        
        return sql
    
    def _generate_ranking_sql(self, extraction_result: ExtractionResult, table_name: str) -> Optional[str]:
        """순위 조회 SQL 생성"""
        # 컬럼 결정
        columns = self._determine_columns(extraction_result, table_name)
        
        # 정렬 기준 결정
        order_column, order_direction = self._determine_order(extraction_result, table_name)
        
        # 제한 조건 확인
        limit = self._get_limit(extraction_result)
        if not limit:
            limit = 10  # 기본값
        
        # 조건 생성
        location_condition = self._build_location_condition(extraction_result, table_name)
        time_condition = self._build_time_condition(extraction_result, table_name)
        
        conditions = []
        if location_condition:
            conditions.append(location_condition)
        if time_condition:
            conditions.append(time_condition)
        
        if conditions:
            condition = " AND ".join(conditions)
            return self.sql_templates['ranking_with_condition'].format(
                columns=columns,
                table=table_name,
                condition=condition,
                order_column=order_column,
                order_direction=order_direction,
                limit=limit
            )
        else:
            return self.sql_templates['ranking_basic'].format(
                columns=columns,
                table=table_name,
                order_column=order_column,
                order_direction=order_direction,
                limit=limit
            )
    
    def _determine_columns(self, extraction_result: ExtractionResult, table_name: str) -> str:
        """컬럼 결정"""
        # 컬럼 힌트가 있으면 사용
        columns = []
        for entity in extraction_result.entities:
            if entity.entity_type == EntityType.COLUMN_HINT:
                columns.append(entity.value)
        
        if columns:
            return ", ".join(columns)
        
        # 기본 컬럼 선택
        if table_name == 'traffic_intersection':
            return "id, name, latitude, longitude"
        elif table_name == 'traffic_trafficvolume':
            return "id, intersection_id, datetime, direction, volume"
        elif table_name == 'traffic_incident':
            return "incident_id, incident_type, district, intersection_name, status"
        else:
            return "*"
    
    def _build_location_condition(self, extraction_result: ExtractionResult, table_name: str) -> Optional[str]:
        """위치 조건 생성"""
        for entity in extraction_result.entities:
            if entity.entity_type == EntityType.LOCATION:
                location = entity.value
                if table_name == 'traffic_intersection':
                    return f"name LIKE '%{location}%'"
                elif table_name == 'traffic_incident':
                    return f"(district LIKE '%{location}%' OR intersection_name LIKE '%{location}%')"
        
        return None
    
    def _build_time_condition(self, extraction_result: ExtractionResult, table_name: str) -> Optional[str]:
        """시간 조건 생성"""
        for entity in extraction_result.entities:
            if entity.entity_type == EntityType.TIME_RANGE:
                time_range = entity.value
                if table_name == 'traffic_trafficvolume':
                    return f"DATE(datetime) BETWEEN '{time_range['start']}' AND '{time_range['end']}'"
                elif table_name == 'traffic_incident':
                    return f"DATE(registered_at) BETWEEN '{time_range['start']}' AND '{time_range['end']}'"
        
        return None
    
    def _determine_order(self, extraction_result: ExtractionResult, table_name: str) -> Tuple[str, str]:
        """정렬 기준 결정"""
        # 정렬 순서 엔티티 확인
        for entity in extraction_result.entities:
            if entity.entity_type == EntityType.SORT_ORDER:
                direction = entity.value
                # 기본 정렬 컬럼
                if table_name == 'traffic_trafficvolume':
                    return 'volume', direction
                elif table_name == 'traffic_incident':
                    return 'registered_at', direction
                else:
                    return 'id', direction
        
        # 기본값
        if table_name == 'traffic_trafficvolume':
            return 'volume', 'DESC'
        elif table_name == 'traffic_incident':
            return 'registered_at', 'DESC'
        else:
            return 'id', 'ASC'
    
    def _get_limit(self, extraction_result: ExtractionResult) -> Optional[int]:
        """제한 조건 추출"""
        for entity in extraction_result.entities:
            if entity.entity_type == EntityType.LIMIT:
                return entity.value
        return None

class HybridSQLGenerator:
    """
    하이브리드 SQL 생성기
    
    1. 규칙 기반 빠른 처리 (0.1초 이내)
    2. 템플릿 기반 중간 처리 (0.5초 이내)
    3. LLM 기반 백업 처리 (2-5초)
    """
    
    def __init__(self, 
                 use_legacy_llm: bool = True,
                 confidence_threshold: float = 0.8):
        """하이브리드 SQL 생성기 초기화"""
        
        self.confidence_threshold = confidence_threshold
        
        # 1. 강화된 정보 추출기
        self.information_extractor = EnhancedInformationExtractor()
        
        # 2. 규칙 기반 엔진
        self.rule_engine = RuleBasedSQLEngine()
        
        # 3. 동적 스키마 관리자 (옵션)
        self.schema_manager = None
        if DYNAMIC_SCHEMA_AVAILABLE:
            try:
                self.schema_manager = DynamicSchemaManager()
                logger.info("동적 스키마 관리자 활성화")
            except Exception as e:
                logger.warning(f"동적 스키마 관리자 초기화 실패: {e}")
        
        # 4. 기존 LLM 생성기 (백업용)
        self.legacy_generator = None
        if use_legacy_llm and LEGACY_SQL_AVAILABLE:
            try:
                self.legacy_generator = SQLGenerator()
                logger.info("기존 LLM SQL 생성기 백업 활성화")
            except Exception as e:
                logger.warning(f"기존 LLM SQL 생성기 초기화 실패: {e}")
        
        # 성능 통계
        self.performance_stats = {
            'rule_based_success': 0,
            'llm_fallback_count': 0,
            'total_requests': 0,
            'avg_response_time': 0.0
        }
        
        logger.info("하이브리드 SQL 생성기 초기화 완료")
    
    def generate_sql(self, question: str, schema: Optional[DatabaseSchema] = None) -> SQLGenerationResult:
        """
        하이브리드 SQL 생성
        
        Args:
            question: 사용자 질문
            schema: 데이터베이스 스키마 (옵션)
            
        Returns:
            SQL 생성 결과
        """
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        logger.info(f"하이브리드 SQL 생성 시작: {question}")
        
        # 1. 정보 추출
        extraction_start = time.time()
        extraction_result = self.information_extractor.extract_information(question)
        extraction_time = time.time() - extraction_start
        
        logger.info(f"정보 추출 완료: {extraction_result.intent.value}, 신뢰도 {extraction_result.confidence:.3f}")
        
        # 2. 규칙 기반 시도 (빠른 처리)
        if extraction_result.confidence >= self.confidence_threshold:
            rule_start = time.time()
            rule_result = self.rule_engine.generate_sql(extraction_result)
            rule_time = time.time() - rule_start
            
            if rule_result:
                self.performance_stats['rule_based_success'] += 1
                total_time = time.time() - start_time
                
                logger.info(f"규칙 기반 SQL 생성 성공: {total_time:.3f}초")
                
                return SQLGenerationResult(
                    query=rule_result.query,
                    method=GenerationMethod.RULE_BASED,
                    confidence=rule_result.confidence,
                    execution_time=total_time,
                    is_valid=True,
                    metadata={
                        'extraction_time': extraction_time,
                        'rule_generation_time': rule_time,
                        'extraction_confidence': extraction_result.confidence,
                        'entities': len(extraction_result.entities)
                    }
                )
        
        # 3. LLM 백업 처리 (복잡한 경우)
        if self.legacy_generator:
            self.performance_stats['llm_fallback_count'] += 1
            
            logger.info("LLM 백업 처리로 전환")
            
            # 기본 스키마 생성
            if not schema:
                schema = self._create_default_schema()
            
            try:
                llm_start = time.time()
                legacy_result = self.legacy_generator.generate_sql(question, schema)
                llm_time = time.time() - llm_start
                
                total_time = time.time() - start_time
                
                logger.info(f"LLM 백업 SQL 생성 완료: {total_time:.3f}초")
                
                return SQLGenerationResult(
                    query=legacy_result.query,
                    method=GenerationMethod.LLM_BASED,
                    confidence=legacy_result.confidence_score,
                    execution_time=total_time,
                    is_valid=legacy_result.is_valid,
                    error_message=legacy_result.error_message,
                    metadata={
                        'extraction_time': extraction_time,
                        'llm_generation_time': llm_time,
                        'extraction_confidence': extraction_result.confidence,
                        'fallback_reason': 'rule_based_failed'
                    }
                )
                
            except Exception as e:
                logger.error(f"LLM 백업 처리 실패: {e}")
        
        # 4. 모든 방법 실패
        total_time = time.time() - start_time
        
        return SQLGenerationResult(
            query="-- SQL 생성 실패",
            method=GenerationMethod.RULE_BASED,
            confidence=0.0,
            execution_time=total_time,
            is_valid=False,
            error_message="모든 SQL 생성 방법 실패",
            metadata={
                'extraction_time': extraction_time,
                'extraction_confidence': extraction_result.confidence,
                'failure_reason': 'all_methods_failed'
            }
        )
    
    def _create_default_schema(self) -> DatabaseSchema:
        """기본 스키마 생성"""
        return DatabaseSchema(
            table_name="traffic_intersection",
            columns=[
                {"name": "id", "type": "INT", "description": "고유 식별자"},
                {"name": "name", "type": "VARCHAR(255)", "description": "교차로 이름"},
                {"name": "latitude", "type": "DECIMAL(10,8)", "description": "위도"},
                {"name": "longitude", "type": "DECIMAL(11,8)", "description": "경도"},
                {"name": "created_at", "type": "DATETIME", "description": "생성일시"},
                {"name": "updated_at", "type": "DATETIME", "description": "수정일시"}
            ],
            primary_key="id"
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        if self.performance_stats['total_requests'] > 0:
            rule_success_rate = self.performance_stats['rule_based_success'] / self.performance_stats['total_requests']
            llm_fallback_rate = self.performance_stats['llm_fallback_count'] / self.performance_stats['total_requests']
        else:
            rule_success_rate = 0.0
            llm_fallback_rate = 0.0
        
        return {
            'total_requests': self.performance_stats['total_requests'],
            'rule_based_success_count': self.performance_stats['rule_based_success'],
            'llm_fallback_count': self.performance_stats['llm_fallback_count'],
            'rule_success_rate': rule_success_rate,
            'llm_fallback_rate': llm_fallback_rate,
            'avg_response_time': self.performance_stats['avg_response_time']
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self.performance_stats = {
            'rule_based_success': 0,
            'llm_fallback_count': 0,
            'total_requests': 0,
            'avg_response_time': 0.0
        }
        logger.info("성능 통계 초기화")

if __name__ == "__main__":
    # 테스트 코드
    generator = HybridSQLGenerator()
    
    test_questions = [
        "조치원읍 교차로가 몇 개인가요?",
        "지난주 한솔동 교통량 평균은?",
        "상위 10개 지역의 교통사고를 보여줘",
        "어제 가장 많은 교통량이 발생한 곳은?",
        "이번달 새롬동 교통량을 알려주세요"
    ]
    
    for question in test_questions:
        print(f"\n질문: {question}")
        result = generator.generate_sql(question)
        print(f"생성 방법: {result.method.value}")
        print(f"실행 시간: {result.execution_time:.3f}초")
        print(f"신뢰도: {result.confidence:.3f}")
        print(f"SQL: {result.query}")
        if result.error_message:
            print(f"오류: {result.error_message}")
    
    print(f"\n성능 통계:")
    stats = generator.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
