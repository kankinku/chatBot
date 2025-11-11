"""
SQL Slot Extractor - 질문에서 SQL 쿼리 요소 추출

규칙 기반 + NER 모델을 사용하여 질문에서 SQL 쿼리 생성에 필요한 요소들을 추출
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class SlotType(Enum):
    """슬롯 유형"""
    TARGET = "target"           # 대상 (컬럼)
    LOCATION = "location"       # 위치
    TIME_RANGE = "time_range"   # 시간 범위
    AGGREGATION = "aggregation" # 집계 함수
    CONDITION = "condition"     # 조건
    LIMIT = "limit"            # 제한

@dataclass
class ExtractedSlot:
    """추출된 슬롯"""
    slot_type: SlotType
    value: Any
    confidence: float
    original_text: str
    metadata: Optional[Dict] = None

@dataclass
class SlotExtractionResult:
    """슬롯 추출 결과"""
    slots: Dict[SlotType, ExtractedSlot]
    confidence: float
    reasoning: str
    original_question: str

class SQLSlotExtractor:
    """
    SQL 슬롯 추출기
    
    질문에서 SQL 쿼리 생성에 필요한 요소들을 추출
    """
    
    def __init__(self):
        """초기화"""
        # 대상 매핑 (한국어 → DB 컬럼)
        self.target_mapping = {
            '교통량': 'traffic_volume',
            '통행량': 'traffic_volume', 
            '차량': 'vehicle_count',
            '이용객': 'passenger_count',
            '사고': 'accident_count',
            '교통사고': 'accident_count',
            '사고건수': 'accident_count',
            '교차로': 'intersection_count',
            '신호등': 'traffic_light_count',
            '혼잡도': 'congestion_level',
            '평균속도': 'average_speed',
            '정체': 'traffic_jam_count'
        }
        
        # 집계 함수 매핑
        self.aggregation_mapping = {
            '총': 'SUM',
            '총합': 'SUM',
            '합계': 'SUM',
            '평균': 'AVG',
            '평균값': 'AVG',
            '최대': 'MAX',
            '최대값': 'MAX',
            '최소': 'MIN',
            '최소값': 'MIN',
            '개수': 'COUNT',
            '건수': 'COUNT',
            '수': 'COUNT'
        }
        
        # 시간 표현 패턴
        self.time_patterns = {
            '오늘': {'start': 'today', 'end': 'today'},
            '어제': {'start': 'yesterday', 'end': 'yesterday'},
            '이번주': {'start': 'this_week', 'end': 'this_week'},
            '지난주': {'start': 'last_week', 'end': 'last_week'},
            '이번달': {'start': 'this_month', 'end': 'this_month'},
            '지난달': {'start': 'last_month', 'end': 'last_month'},
            '올해': {'start': 'this_year', 'end': 'this_year'},
            '작년': {'start': 'last_year', 'end': 'last_year'},
            '최근': {'start': 'recent_7_days', 'end': 'today'},
            '지난': {'start': 'last_30_days', 'end': 'yesterday'}
        }
        
        # 위치 패턴
        self.location_patterns = {
            '구': r'([가-힣]+구)',
            '역': r'([가-힣]+역)',
            '대로': r'([가-힣]+대로)',
            '동': r'([가-힣]+동)',
            '로': r'([가-힣]+로)'
        }
        
        # 조건 패턴
        self.condition_patterns = {
            '많은': '>',
            '적은': '<',
            '높은': '>',
            '낮은': '<',
            '상위': 'TOP',
            '하위': 'BOTTOM'
        }
        
        logger.info("SQL 슬롯 추출기 초기화 완료")
    
    def extract_slots(self, question: str) -> SlotExtractionResult:
        """
        질문에서 슬롯 추출
        
        Args:
            question: 사용자 질문
            
        Returns:
            추출된 슬롯들
        """
        question_lower = question.lower()
        slots = {}
        confidence_scores = []
        reasoning_parts = []
        
        # 1. 대상 추출
        target_slot = self._extract_target(question_lower)
        if target_slot:
            slots[SlotType.TARGET] = target_slot
            confidence_scores.append(target_slot.confidence)
            reasoning_parts.append(f"대상: {target_slot.original_text} → {target_slot.value}")
        
        # 2. 위치 추출
        location_slot = self._extract_location(question)
        if location_slot:
            slots[SlotType.LOCATION] = location_slot
            confidence_scores.append(location_slot.confidence)
            reasoning_parts.append(f"위치: {location_slot.original_text}")
        
        # 3. 시간 범위 추출
        time_slot = self._extract_time_range(question)
        if time_slot:
            slots[SlotType.TIME_RANGE] = time_slot
            confidence_scores.append(time_slot.confidence)
            reasoning_parts.append(f"시간: {time_slot.original_text}")
        
        # 4. 집계 함수 추출
        aggregation_slot = self._extract_aggregation(question_lower)
        if aggregation_slot:
            slots[SlotType.AGGREGATION] = aggregation_slot
            confidence_scores.append(aggregation_slot.confidence)
            reasoning_parts.append(f"집계: {aggregation_slot.original_text} → {aggregation_slot.value}")
        
        # 5. 조건 추출
        condition_slot = self._extract_condition(question_lower)
        if condition_slot:
            slots[SlotType.CONDITION] = condition_slot
            confidence_scores.append(condition_slot.confidence)
            reasoning_parts.append(f"조건: {condition_slot.original_text}")
        
        # 6. 제한 추출
        limit_slot = self._extract_limit(question)
        if limit_slot:
            slots[SlotType.LIMIT] = limit_slot
            confidence_scores.append(limit_slot.confidence)
            reasoning_parts.append(f"제한: {limit_slot.original_text}")
        
        # 전체 신뢰도 계산
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        return SlotExtractionResult(
            slots=slots,
            confidence=overall_confidence,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "슬롯 추출 없음",
            original_question=question
        )
    
    def _extract_target(self, question: str) -> Optional[ExtractedSlot]:
        """대상(컬럼) 추출"""
        for korean, column in self.target_mapping.items():
            if korean in question:
                return ExtractedSlot(
                    slot_type=SlotType.TARGET,
                    value=column,
                    confidence=0.9,
                    original_text=korean,
                    metadata={'column_name': column}
                )
        
        return None
    
    def _extract_location(self, question: str) -> Optional[ExtractedSlot]:
        """위치 추출"""
        for location_type, pattern in self.location_patterns.items():
            match = re.search(pattern, question)
            if match:
                location = match.group(1)
                return ExtractedSlot(
                    slot_type=SlotType.LOCATION,
                    value=location,
                    confidence=0.8,
                    original_text=location,
                    metadata={'location_type': location_type}
                )
        
        return None
    
    def _extract_time_range(self, question: str) -> Optional[ExtractedSlot]:
        """시간 범위 추출"""
        for pattern, time_range in self.time_patterns.items():
            if pattern in question:
                resolved_range = self._resolve_time_range(time_range)
                return ExtractedSlot(
                    slot_type=SlotType.TIME_RANGE,
                    value=resolved_range,
                    confidence=0.9,
                    original_text=pattern,
                    metadata={'time_pattern': pattern}
                )
        
        return None
    
    def _resolve_time_range(self, time_range: Dict) -> Dict:
        """시간 범위를 실제 날짜로 변환"""
        now = datetime.now()
        
        if time_range['start'] == 'today':
            return {
                'start': now.strftime('%Y-%m-%d'),
                'end': now.strftime('%Y-%m-%d')
            }
        elif time_range['start'] == 'yesterday':
            yesterday = now - timedelta(days=1)
            return {
                'start': yesterday.strftime('%Y-%m-%d'),
                'end': yesterday.strftime('%Y-%m-%d')
            }
        elif time_range['start'] == 'this_week':
            start_of_week = now - timedelta(days=now.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            return {
                'start': start_of_week.strftime('%Y-%m-%d'),
                'end': end_of_week.strftime('%Y-%m-%d')
            }
        elif time_range['start'] == 'last_week':
            start_of_last_week = now - timedelta(days=now.weekday() + 7)
            end_of_last_week = start_of_last_week + timedelta(days=6)
            return {
                'start': start_of_last_week.strftime('%Y-%m-%d'),
                'end': end_of_last_week.strftime('%Y-%m-%d')
            }
        elif time_range['start'] == 'this_month':
            start_of_month = now.replace(day=1)
            return {
                'start': start_of_month.strftime('%Y-%m-%d'),
                'end': now.strftime('%Y-%m-%d')
            }
        elif time_range['start'] == 'last_month':
            if now.month == 1:
                last_month = now.replace(year=now.year-1, month=12)
            else:
                last_month = now.replace(month=now.month-1)
            start_of_last_month = last_month.replace(day=1)
            return {
                'start': start_of_last_month.strftime('%Y-%m-%d'),
                'end': last_month.strftime('%Y-%m-%d')
            }
        elif time_range['start'] == 'recent_7_days':
            start_date = now - timedelta(days=7)
            return {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': now.strftime('%Y-%m-%d')
            }
        elif time_range['start'] == 'last_30_days':
            start_date = now - timedelta(days=30)
            return {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': now.strftime('%Y-%m-%d')
            }
        
        return time_range
    
    def _extract_aggregation(self, question: str) -> Optional[ExtractedSlot]:
        """집계 함수 추출"""
        for korean, func in self.aggregation_mapping.items():
            if korean in question:
                return ExtractedSlot(
                    slot_type=SlotType.AGGREGATION,
                    value=func,
                    confidence=0.9,
                    original_text=korean,
                    metadata={'function': func}
                )
        
        return None
    
    def _extract_condition(self, question: str) -> Optional[ExtractedSlot]:
        """조건 추출"""
        for condition, operator in self.condition_patterns.items():
            if condition in question:
                return ExtractedSlot(
                    slot_type=SlotType.CONDITION,
                    value=operator,
                    confidence=0.7,
                    original_text=condition,
                    metadata={'operator': operator}
                )
        
        return None
    
    def _extract_limit(self, question: str) -> Optional[ExtractedSlot]:
        """제한 추출 (상위 N개, 하위 N개 등)"""
        # 상위/하위 N개 패턴
        limit_patterns = [
            r'상위\s*(\d+)개',
            r'하위\s*(\d+)개',
            r'최고\s*(\d+)개',
            r'최저\s*(\d+)개'
        ]
        
        for pattern in limit_patterns:
            match = re.search(pattern, question)
            if match:
                limit_num = int(match.group(1))
                return ExtractedSlot(
                    slot_type=SlotType.LIMIT,
                    value=limit_num,
                    confidence=0.8,
                    original_text=match.group(0),
                    metadata={'limit_type': 'top' if '상위' in match.group(0) or '최고' in match.group(0) else 'bottom'}
                )
        
        return None
    
    def generate_sql_from_slots(self, slots: Dict[SlotType, ExtractedSlot], table_name: str = "traffic_data") -> str:
        """
        추출된 슬롯으로부터 SQL 쿼리 생성
        
        Args:
            slots: 추출된 슬롯들
            table_name: 테이블 이름
            
        Returns:
            생성된 SQL 쿼리
        """
        query_parts = []
        
        # SELECT 절
        select_parts = []
        if SlotType.AGGREGATION in slots and SlotType.TARGET in slots:
            agg_func = slots[SlotType.AGGREGATION].value
            target_col = slots[SlotType.TARGET].value
            select_parts.append(f"{agg_func}({target_col})")
        elif SlotType.TARGET in slots:
            select_parts.append(slots[SlotType.TARGET].value)
        else:
            select_parts.append("*")
        
        query_parts.append(f"SELECT {', '.join(select_parts)}")
        query_parts.append(f"FROM {table_name}")
        
        # WHERE 절
        conditions = []
        
        if SlotType.LOCATION in slots:
            location = slots[SlotType.LOCATION].value
            conditions.append(f"location LIKE '%{location}%'")
        
        if SlotType.TIME_RANGE in slots:
            time_range = slots[SlotType.TIME_RANGE].value
            conditions.append(f"date BETWEEN '{time_range['start']}' AND '{time_range['end']}'")
        
        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))
        
        # ORDER BY 절
        if SlotType.CONDITION in slots:
            condition = slots[SlotType.CONDITION].value
            if condition in ['>', '<'] and SlotType.TARGET in slots:
                target_col = slots[SlotType.TARGET].value
                order_direction = "DESC" if condition == '>' else "ASC"
                query_parts.append(f"ORDER BY {target_col} {order_direction}")
        
        # LIMIT 절
        if SlotType.LIMIT in slots:
            limit_num = slots[SlotType.LIMIT].value
            query_parts.append(f"LIMIT {limit_num}")
        
        return " ".join(query_parts)
    
    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """
        SQL 쿼리 검증
        
        Args:
            sql: 검증할 SQL 쿼리
            
        Returns:
            (유효성, 오류 메시지)
        """
        # 위험한 키워드 체크
        dangerous_keywords = ['DELETE', 'DROP', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        sql_upper = sql.upper()
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"위험한 키워드 '{keyword}'가 포함되어 있습니다."
        
        # 기본 SQL 구조 체크
        if not sql_upper.startswith('SELECT'):
            return False, "SELECT 문으로 시작해야 합니다."
        
        # FROM 절 체크
        if 'FROM' not in sql_upper:
            return False, "FROM 절이 없습니다."
        
        return True, "유효한 SQL 쿼리입니다."

if __name__ == "__main__":
    # 테스트 코드
    extractor = SQLSlotExtractor()
    
    test_questions = [
        "지난주 강남구 교통량 평균은?",
        "서울역 이용객 수를 알려줘",
        "이번달 교통사고 총 건수",
        "상위 10개 지역의 혼잡도",
        "어제 강남대로 평균속도"
    ]
    
    for question in test_questions:
        print(f"\n질문: {question}")
        result = extractor.extract_slots(question)
        print(f"추출된 슬롯: {result.reasoning}")
        print(f"신뢰도: {result.confidence:.3f}")
        
        if result.slots:
            sql = extractor.generate_sql_from_slots(result.slots)
            is_valid, message = extractor.validate_sql(sql)
            print(f"생성된 SQL: {sql}")
            print(f"검증 결과: {message}")
