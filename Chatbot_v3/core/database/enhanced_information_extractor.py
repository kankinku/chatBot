"""
강화된 정보 추출기 - 통합된 NER, 의도 분석, 엔티티 매핑

질문에서 필요한 정보를 정확하게 추출하여 SQL 생성 정확도를 높이는 모듈
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logging.warning("sentence-transformers 라이브러리를 찾을 수 없습니다.")

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """엔티티 타입"""
    LOCATION = "location"           # 위치/지역
    TIME_RANGE = "time_range"       # 시간 범위
    METRIC = "metric"               # 측정값/지표
    AGGREGATION = "aggregation"     # 집계 함수
    CONDITION = "condition"         # 조건
    COMPARISON = "comparison"       # 비교 연산자
    SORT_ORDER = "sort_order"       # 정렬 순서
    LIMIT = "limit"                 # 제한 개수
    TABLE_HINT = "table_hint"       # 테이블 힌트
    COLUMN_HINT = "column_hint"     # 컬럼 힌트

class IntentType(Enum):
    """의도 타입"""
    COUNT = "count"                 # 개수 조회
    AGGREGATE = "aggregate"         # 집계 조회
    LIST = "list"                   # 목록 조회
    COMPARISON = "comparison"       # 비교 조회
    RANKING = "ranking"             # 순위 조회
    ANALYSIS = "analysis"           # 분석 조회
    SEARCH = "search"               # 검색 조회

@dataclass
class ExtractedEntity:
    """추출된 엔티티"""
    entity_type: EntityType
    value: Any
    original_text: str
    confidence: float
    start_pos: int = -1
    end_pos: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractionResult:
    """정보 추출 결과"""
    original_question: str
    processed_question: str
    intent: IntentType
    entities: List[ExtractedEntity]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedInformationExtractor:
    """
    강화된 정보 추출기
    
    기능:
    1. 다층 NER (Named Entity Recognition)
    2. 의도 분석 (Intent Analysis)
    3. 컨텍스트 기반 엔티티 매핑
    4. 시간 표현 정규화
    5. 지역 정보 표준화
    """
    
    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask"):
        """강화된 정보 추출기 초기화"""
        
        # 임베딩 모델 초기화
        self.embedding_model = None
        if SBERT_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"정보 추출용 임베딩 모델 로드: {embedding_model}")
            except Exception as e:
                logger.warning(f"임베딩 모델 로드 실패: {e}")
        
        # 세종시 특화 지역 매핑
        self.location_mapping = {
            # 행정동
            "한솔동": ["한솔", "한솔동", "hansol"],
            "새롬동": ["새롬", "새롬동", "saerom"],
            "도담동": ["도담", "도담동", "dodam"],
            "아름동": ["아름", "아름동", "areum"],
            "종촌동": ["종촌", "종촌동", "jongchon"],
            "고운동": ["고운", "고운동", "goun"],
            "보람동": ["보람", "보람동", "boram"],
            "대평동": ["대평", "대평동", "daepyeong"],
            "소담동": ["소담", "소담동", "sodam"],
            "반곡동": ["반곡", "반곡동", "bangok"],
            "다정동": ["다정", "다정동", "dajeong"],
            "어진동": ["어진", "어진동", "eojin"],
            "나성동": ["나성", "나성동", "naseong"],
            "새뜸동": ["새뜸", "새뜸동", "saettum"],
            "다솜동": ["다솜", "다솜동", "dasom"],
            "한별동": ["한별", "한별동", "hanbyeol"],
            "가람동": ["가람", "가람동", "garam"],
            "도움동": ["도움", "도움동", "doum"],
            "비전동": ["비전", "비전동", "bijeon"],
            "새움동": ["새움", "새움동", "saeeum"],
            
            # 읍면
            "조치원읍": ["조치원", "조치원읍", "jochiwon"],
            "연기면": ["연기", "연기면", "yeongi"],
            "연동면": ["연동", "연동면", "yeondong"],
            "부강면": ["부강", "부강면", "bugang"],
            "금남면": ["금남", "금남면", "geumnam"],
            "장군면": ["장군", "장군면", "janggun"],
            "연서면": ["연서", "연서면", "yeonseo"],
            "전동면": ["전동", "전동면", "jeondong"],
            "전의면": ["전의", "전의면", "jeonui"],
            "소정면": ["소정", "소정면", "sojeong"],
        }
        
        # 시간 표현 패턴 (정규화된)
        self.time_patterns = {
            # 절대 시간
            r'(\d{4})년': self._parse_year,
            r'(\d{1,2})월': self._parse_month,
            r'(\d{1,2})일': self._parse_day,
            r'(\d{1,2})시': self._parse_hour,
            
            # 상대 시간
            '오늘': self._get_today,
            '어제': self._get_yesterday,
            '내일': self._get_tomorrow,
            '이번주': self._get_this_week,
            '지난주': self._get_last_week,
            '다음주': self._get_next_week,
            '이번달': self._get_this_month,
            '지난달': self._get_last_month,
            '다음달': self._get_next_month,
            '올해': self._get_this_year,
            '작년': self._get_last_year,
            '내년': self._get_next_year,
            
            # 기간 표현
            '최근': self._get_recent,
            '지난': self._get_past,
        }
        
        # 집계 함수 매핑
        self.aggregation_mapping = {
            # 개수
            '개수': 'COUNT', '수': 'COUNT', '건수': 'COUNT', '몇개': 'COUNT', '몇': 'COUNT',
            
            # 합계
            '총': 'SUM', '총합': 'SUM', '합계': 'SUM', '전체': 'SUM',
            
            # 평균
            '평균': 'AVG', '평균값': 'AVG', '평균적으로': 'AVG',
            
            # 최대/최소
            '최대': 'MAX', '최고': 'MAX', '가장많은': 'MAX', '가장큰': 'MAX',
            '최소': 'MIN', '최저': 'MIN', '가장적은': 'MIN', '가장작은': 'MIN',
        }
        
        # 비교 연산자 매핑
        self.comparison_mapping = {
            '많은': '>', '높은': '>', '큰': '>', '이상': '>=', '초과': '>',
            '적은': '<', '낮은': '<', '작은': '<', '이하': '<=', '미만': '<',
            '같은': '=', '동일한': '=', '똑같은': '=',
        }
        
        # 정렬 순서 매핑
        self.sort_mapping = {
            '높은순': 'DESC', '많은순': 'DESC', '큰순': 'DESC', '내림차순': 'DESC',
            '낮은순': 'ASC', '적은순': 'ASC', '작은순': 'ASC', '오름차순': 'ASC',
            '상위': 'DESC', '하위': 'ASC', '최신': 'DESC', '과거': 'ASC',
        }
        
        # 의도 분석용 키워드 패턴
        self.intent_patterns = {
            IntentType.COUNT: [
                '몇개', '개수', '수', '건수', '몇', '얼마나많은', '총몇개'
            ],
            IntentType.AGGREGATE: [
                '총', '합계', '평균', '최대', '최소', '총합', '전체'
            ],
            IntentType.LIST: [
                '목록', '리스트', '보여줘', '알려줘', '나열', '전체보기'
            ],
            IntentType.COMPARISON: [
                '비교', '대비', '차이', '비율', '대조', '상대적'
            ],
            IntentType.RANKING: [
                '순위', '랭킹', '상위', '하위', '1위', '2위', '톱', 'top'
            ],
            IntentType.ANALYSIS: [
                '분석', '통계', '추세', '패턴', '경향', '변화', '증감'
            ],
            IntentType.SEARCH: [
                '찾아줘', '검색', '조회', '확인', '어디', '언제', '무엇'
            ]
        }
        
        # 테이블/컬럼 힌트 매핑
        self.table_hints = {
            '교차로': 'traffic_intersection',
            '교통량': 'traffic_trafficvolume',
            '통행량': 'traffic_trafficvolume',
            '사고': 'traffic_incident',
            '교통사고': 'traffic_incident',
        }
        
        self.column_hints = {
            # traffic_intersection
            '교차로명': 'name',
            '이름': 'name',
            '위치': 'name',
            '위도': 'latitude',
            '경도': 'longitude',
            
            # traffic_trafficvolume
            '교통량': 'volume',
            '통행량': 'volume',
            '차량수': 'volume',
            '방향': 'direction',
            '시간': 'datetime',
            '날짜': 'datetime',
            
            # traffic_incident
            '사고유형': 'incident_type',
            '상태': 'status',
            '지역': 'district',
            '신고일': 'registered_at',
        }
        
        logger.info("강화된 정보 추출기 초기화 완료")
    
    def extract_information(self, question: str) -> ExtractionResult:
        """
        질문에서 정보 추출
        
        Args:
            question: 사용자 질문
            
        Returns:
            추출된 정보
        """
        start_time = datetime.now()
        
        # 1. 질문 전처리
        processed_question = self._preprocess_question(question)
        
        # 2. 의도 분석
        intent = self._analyze_intent(processed_question)
        
        # 3. 엔티티 추출
        entities = []
        
        # 3.1 위치 엔티티 추출
        location_entities = self._extract_location_entities(processed_question)
        entities.extend(location_entities)
        
        # 3.2 시간 엔티티 추출
        time_entities = self._extract_time_entities(processed_question)
        entities.extend(time_entities)
        
        # 3.3 집계 함수 추출
        aggregation_entities = self._extract_aggregation_entities(processed_question)
        entities.extend(aggregation_entities)
        
        # 3.4 비교 조건 추출
        comparison_entities = self._extract_comparison_entities(processed_question)
        entities.extend(comparison_entities)
        
        # 3.5 정렬 조건 추출
        sort_entities = self._extract_sort_entities(processed_question)
        entities.extend(sort_entities)
        
        # 3.6 제한 조건 추출
        limit_entities = self._extract_limit_entities(processed_question)
        entities.extend(limit_entities)
        
        # 3.7 테이블/컬럼 힌트 추출
        table_entities = self._extract_table_hints(processed_question)
        entities.extend(table_entities)
        
        column_entities = self._extract_column_hints(processed_question)
        entities.extend(column_entities)
        
        # 4. 신뢰도 계산
        confidence = self._calculate_confidence(intent, entities)
        
        # 5. 추론 과정 생성
        reasoning = self._generate_reasoning(intent, entities)
        
        # 6. 메타데이터 생성
        processing_time = (datetime.now() - start_time).total_seconds()
        metadata = {
            'processing_time': processing_time,
            'entity_count': len(entities),
            'entity_types': list(set(e.entity_type.value for e in entities)),
            'confidence_scores': [e.confidence for e in entities],
        }
        
        logger.info(f"정보 추출 완료: {intent.value}, 엔티티 {len(entities)}개, 신뢰도 {confidence:.3f}")
        
        return ExtractionResult(
            original_question=question,
            processed_question=processed_question,
            intent=intent,
            entities=entities,
            confidence=confidence,
            reasoning=reasoning,
            metadata=metadata
        )
    
    def _preprocess_question(self, question: str) -> str:
        """질문 전처리"""
        # 소문자 변환
        question = question.lower()
        
        # 불필요한 문자 제거
        question = re.sub(r'[^\w\s가-힣]', ' ', question)
        
        # 연속된 공백 정리
        question = re.sub(r'\s+', ' ', question)
        
        return question.strip()
    
    def _analyze_intent(self, question: str) -> IntentType:
        """의도 분석"""
        intent_scores = {}
        
        for intent_type, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in question)
            if score > 0:
                intent_scores[intent_type] = score
        
        if not intent_scores:
            return IntentType.SEARCH  # 기본값
        
        # 최고 점수 의도 반환
        return max(intent_scores, key=intent_scores.get)
    
    def _extract_location_entities(self, question: str) -> List[ExtractedEntity]:
        """위치 엔티티 추출"""
        entities = []
        
        for location, aliases in self.location_mapping.items():
            for alias in aliases:
                if alias in question:
                    start_pos = question.find(alias)
                    entities.append(ExtractedEntity(
                        entity_type=EntityType.LOCATION,
                        value=location,
                        original_text=alias,
                        confidence=0.9,
                        start_pos=start_pos,
                        end_pos=start_pos + len(alias),
                        metadata={'normalized_name': location}
                    ))
        
        return entities
    
    def _extract_time_entities(self, question: str) -> List[ExtractedEntity]:
        """시간 엔티티 추출"""
        entities = []
        
        for pattern, parser in self.time_patterns.items():
            if isinstance(pattern, str):
                # 문자열 패턴
                if pattern in question:
                    start_pos = question.find(pattern)
                    time_range = parser()
                    entities.append(ExtractedEntity(
                        entity_type=EntityType.TIME_RANGE,
                        value=time_range,
                        original_text=pattern,
                        confidence=0.95,
                        start_pos=start_pos,
                        end_pos=start_pos + len(pattern),
                        metadata={'time_type': 'relative'}
                    ))
            else:
                # 정규식 패턴
                matches = re.finditer(pattern, question)
                for match in matches:
                    time_value = parser(match.group(1))
                    entities.append(ExtractedEntity(
                        entity_type=EntityType.TIME_RANGE,
                        value=time_value,
                        original_text=match.group(0),
                        confidence=0.85,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        metadata={'time_type': 'absolute'}
                    ))
        
        return entities
    
    def _extract_aggregation_entities(self, question: str) -> List[ExtractedEntity]:
        """집계 함수 엔티티 추출"""
        entities = []
        
        for keyword, function in self.aggregation_mapping.items():
            if keyword in question:
                start_pos = question.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.AGGREGATION,
                    value=function,
                    original_text=keyword,
                    confidence=0.9,
                    start_pos=start_pos,
                    end_pos=start_pos + len(keyword),
                    metadata={'sql_function': function}
                ))
        
        return entities
    
    def _extract_comparison_entities(self, question: str) -> List[ExtractedEntity]:
        """비교 조건 엔티티 추출"""
        entities = []
        
        for keyword, operator in self.comparison_mapping.items():
            if keyword in question:
                start_pos = question.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.COMPARISON,
                    value=operator,
                    original_text=keyword,
                    confidence=0.85,
                    start_pos=start_pos,
                    end_pos=start_pos + len(keyword),
                    metadata={'sql_operator': operator}
                ))
        
        return entities
    
    def _extract_sort_entities(self, question: str) -> List[ExtractedEntity]:
        """정렬 조건 엔티티 추출"""
        entities = []
        
        for keyword, order in self.sort_mapping.items():
            if keyword in question:
                start_pos = question.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.SORT_ORDER,
                    value=order,
                    original_text=keyword,
                    confidence=0.8,
                    start_pos=start_pos,
                    end_pos=start_pos + len(keyword),
                    metadata={'sql_order': order}
                ))
        
        return entities
    
    def _extract_limit_entities(self, question: str) -> List[ExtractedEntity]:
        """제한 조건 엔티티 추출"""
        entities = []
        
        # 숫자 + 개/건 패턴
        limit_patterns = [
            r'(\d+)개',
            r'(\d+)건',
            r'상위\s*(\d+)',
            r'하위\s*(\d+)',
            r'톱\s*(\d+)',
            r'top\s*(\d+)',
        ]
        
        for pattern in limit_patterns:
            matches = re.finditer(pattern, question)
            for match in matches:
                limit_value = int(match.group(1))
                entities.append(ExtractedEntity(
                    entity_type=EntityType.LIMIT,
                    value=limit_value,
                    original_text=match.group(0),
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={'limit_value': limit_value}
                ))
        
        return entities
    
    def _extract_table_hints(self, question: str) -> List[ExtractedEntity]:
        """테이블 힌트 추출"""
        entities = []
        
        for keyword, table in self.table_hints.items():
            if keyword in question:
                start_pos = question.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.TABLE_HINT,
                    value=table,
                    original_text=keyword,
                    confidence=0.8,
                    start_pos=start_pos,
                    end_pos=start_pos + len(keyword),
                    metadata={'table_name': table}
                ))
        
        return entities
    
    def _extract_column_hints(self, question: str) -> List[ExtractedEntity]:
        """컬럼 힌트 추출"""
        entities = []
        
        for keyword, column in self.column_hints.items():
            if keyword in question:
                start_pos = question.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.COLUMN_HINT,
                    value=column,
                    original_text=keyword,
                    confidence=0.75,
                    start_pos=start_pos,
                    end_pos=start_pos + len(keyword),
                    metadata={'column_name': column}
                ))
        
        return entities
    
    def _calculate_confidence(self, intent: IntentType, entities: List[ExtractedEntity]) -> float:
        """전체 신뢰도 계산"""
        if not entities:
            return 0.5
        
        # 엔티티 신뢰도 평균
        entity_confidence = sum(e.confidence for e in entities) / len(entities)
        
        # 의도와 엔티티 일치도 보너스
        intent_bonus = 0.0
        if intent == IntentType.COUNT and any(e.entity_type == EntityType.AGGREGATION and e.value == 'COUNT' for e in entities):
            intent_bonus = 0.1
        elif intent == IntentType.AGGREGATE and any(e.entity_type == EntityType.AGGREGATION for e in entities):
            intent_bonus = 0.1
        elif intent == IntentType.RANKING and any(e.entity_type == EntityType.SORT_ORDER for e in entities):
            intent_bonus = 0.1
        
        # 엔티티 다양성 보너스
        entity_types = set(e.entity_type for e in entities)
        diversity_bonus = min(len(entity_types) * 0.05, 0.2)
        
        total_confidence = min(entity_confidence + intent_bonus + diversity_bonus, 1.0)
        return total_confidence
    
    def _generate_reasoning(self, intent: IntentType, entities: List[ExtractedEntity]) -> str:
        """추론 과정 생성"""
        reasoning_parts = [f"의도: {intent.value}"]
        
        entity_summary = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            if entity_type not in entity_summary:
                entity_summary[entity_type] = []
            entity_summary[entity_type].append(entity.original_text)
        
        for entity_type, texts in entity_summary.items():
            reasoning_parts.append(f"{entity_type}: {', '.join(texts)}")
        
        return " | ".join(reasoning_parts)
    
    # 시간 처리 함수들
    def _get_today(self) -> Dict[str, str]:
        """오늘 날짜 반환"""
        today = datetime.now()
        return {
            'start': today.strftime('%Y-%m-%d'),
            'end': today.strftime('%Y-%m-%d')
        }
    
    def _get_yesterday(self) -> Dict[str, str]:
        """어제 날짜 반환"""
        yesterday = datetime.now() - timedelta(days=1)
        return {
            'start': yesterday.strftime('%Y-%m-%d'),
            'end': yesterday.strftime('%Y-%m-%d')
        }
    
    def _get_tomorrow(self) -> Dict[str, str]:
        """내일 날짜 반환"""
        tomorrow = datetime.now() + timedelta(days=1)
        return {
            'start': tomorrow.strftime('%Y-%m-%d'),
            'end': tomorrow.strftime('%Y-%m-%d')
        }
    
    def _get_this_week(self) -> Dict[str, str]:
        """이번주 날짜 범위 반환"""
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        return {
            'start': start_of_week.strftime('%Y-%m-%d'),
            'end': end_of_week.strftime('%Y-%m-%d')
        }
    
    def _get_last_week(self) -> Dict[str, str]:
        """지난주 날짜 범위 반환"""
        today = datetime.now()
        start_of_last_week = today - timedelta(days=today.weekday() + 7)
        end_of_last_week = start_of_last_week + timedelta(days=6)
        return {
            'start': start_of_last_week.strftime('%Y-%m-%d'),
            'end': end_of_last_week.strftime('%Y-%m-%d')
        }
    
    def _get_next_week(self) -> Dict[str, str]:
        """다음주 날짜 범위 반환"""
        today = datetime.now()
        start_of_next_week = today + timedelta(days=7 - today.weekday())
        end_of_next_week = start_of_next_week + timedelta(days=6)
        return {
            'start': start_of_next_week.strftime('%Y-%m-%d'),
            'end': end_of_next_week.strftime('%Y-%m-%d')
        }
    
    def _get_this_month(self) -> Dict[str, str]:
        """이번달 날짜 범위 반환"""
        today = datetime.now()
        start_of_month = today.replace(day=1)
        if today.month == 12:
            end_of_month = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end_of_month = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
        return {
            'start': start_of_month.strftime('%Y-%m-%d'),
            'end': end_of_month.strftime('%Y-%m-%d')
        }
    
    def _get_last_month(self) -> Dict[str, str]:
        """지난달 날짜 범위 반환"""
        today = datetime.now()
        if today.month == 1:
            start_of_last_month = today.replace(year=today.year - 1, month=12, day=1)
        else:
            start_of_last_month = today.replace(month=today.month - 1, day=1)
        end_of_last_month = today.replace(day=1) - timedelta(days=1)
        return {
            'start': start_of_last_month.strftime('%Y-%m-%d'),
            'end': end_of_last_month.strftime('%Y-%m-%d')
        }
    
    def _get_next_month(self) -> Dict[str, str]:
        """다음달 날짜 범위 반환"""
        today = datetime.now()
        if today.month == 12:
            start_of_next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            start_of_next_month = today.replace(month=today.month + 1, day=1)
        if start_of_next_month.month == 12:
            end_of_next_month = start_of_next_month.replace(year=start_of_next_month.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end_of_next_month = start_of_next_month.replace(month=start_of_next_month.month + 1, day=1) - timedelta(days=1)
        return {
            'start': start_of_next_month.strftime('%Y-%m-%d'),
            'end': end_of_next_month.strftime('%Y-%m-%d')
        }
    
    def _get_this_year(self) -> Dict[str, str]:
        """올해 날짜 범위 반환"""
        today = datetime.now()
        start_of_year = today.replace(month=1, day=1)
        end_of_year = today.replace(month=12, day=31)
        return {
            'start': start_of_year.strftime('%Y-%m-%d'),
            'end': end_of_year.strftime('%Y-%m-%d')
        }
    
    def _get_last_year(self) -> Dict[str, str]:
        """작년 날짜 범위 반환"""
        today = datetime.now()
        start_of_last_year = today.replace(year=today.year - 1, month=1, day=1)
        end_of_last_year = today.replace(year=today.year - 1, month=12, day=31)
        return {
            'start': start_of_last_year.strftime('%Y-%m-%d'),
            'end': end_of_last_year.strftime('%Y-%m-%d')
        }
    
    def _get_next_year(self) -> Dict[str, str]:
        """내년 날짜 범위 반환"""
        today = datetime.now()
        start_of_next_year = today.replace(year=today.year + 1, month=1, day=1)
        end_of_next_year = today.replace(year=today.year + 1, month=12, day=31)
        return {
            'start': start_of_next_year.strftime('%Y-%m-%d'),
            'end': end_of_next_year.strftime('%Y-%m-%d')
        }
    
    def _get_recent(self) -> Dict[str, str]:
        """최근 (7일) 날짜 범위 반환"""
        today = datetime.now()
        start_date = today - timedelta(days=7)
        return {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': today.strftime('%Y-%m-%d')
        }
    
    def _get_past(self) -> Dict[str, str]:
        """지난 (30일) 날짜 범위 반환"""
        today = datetime.now()
        start_date = today - timedelta(days=30)
        return {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': today.strftime('%Y-%m-%d')
        }
    
    # 절대 시간 파서들
    def _parse_year(self, year_str: str) -> Dict[str, str]:
        """연도 파싱"""
        year = int(year_str)
        return {
            'start': f'{year}-01-01',
            'end': f'{year}-12-31'
        }
    
    def _parse_month(self, month_str: str) -> Dict[str, str]:
        """월 파싱"""
        month = int(month_str)
        today = datetime.now()
        year = today.year
        
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        return {
            'start': f'{year}-{month:02d}-01',
            'end': end_date.strftime('%Y-%m-%d')
        }
    
    def _parse_day(self, day_str: str) -> Dict[str, str]:
        """일 파싱"""
        day = int(day_str)
        today = datetime.now()
        target_date = today.replace(day=day)
        
        return {
            'start': target_date.strftime('%Y-%m-%d'),
            'end': target_date.strftime('%Y-%m-%d')
        }
    
    def _parse_hour(self, hour_str: str) -> Dict[str, str]:
        """시간 파싱"""
        hour = int(hour_str)
        today = datetime.now()
        target_datetime = today.replace(hour=hour, minute=0, second=0)
        
        return {
            'start': target_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'end': target_datetime.replace(hour=hour, minute=59, second=59).strftime('%Y-%m-%d %H:%M:%S')
        }

if __name__ == "__main__":
    # 테스트 코드
    extractor = EnhancedInformationExtractor()
    
    test_questions = [
        "조치원읍 교차로가 몇 개인가요?",
        "지난주 한솔동 교통량 평균은?",
        "상위 10개 지역의 교통사고 건수를 보여줘",
        "어제 가장 많은 교통량이 발생한 곳은?",
        "이번달 새롬동과 도담동 교통량을 비교해줘"
    ]
    
    for question in test_questions:
        print(f"\n질문: {question}")
        result = extractor.extract_information(question)
        print(f"의도: {result.intent.value}")
        print(f"엔티티 수: {len(result.entities)}")
        print(f"신뢰도: {result.confidence:.3f}")
        print(f"추론: {result.reasoning}")
        
        for entity in result.entities:
            print(f"  - {entity.entity_type.value}: {entity.original_text} → {entity.value}")
