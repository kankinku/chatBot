"""
SQL 요소 추출기 - 규칙 기반/NER/슬롯 채우기 방식

실제 데이터베이스 스키마를 기반으로 SQL 요소를 추출하는 규칙 기반 방식
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """쿼리 타입"""
    SELECT = "SELECT"
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MAX = "MAX"
    MIN = "MIN"
    GROUP_BY = "GROUP_BY"
    ORDER_BY = "ORDER_BY"

class ComparisonOperator(Enum):
    """비교 연산자"""
    EQUAL = "="
    GREATER = ">"
    LESS = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    NOT_EQUAL = "!="
    LIKE = "LIKE"
    IN = "IN"
    BETWEEN = "BETWEEN"

@dataclass
class SQLSlot:
    """SQL 슬롯 (요소)"""
    slot_type: str  # table, column, value, condition 등
    value: str
    confidence: float = 1.0
    original_text: str = ""

@dataclass
class ExtractedSQLElements:
    """추출된 SQL 요소들"""
    query_type: QueryType
    table_name: str
    columns: List[str] = None
    conditions: List[Dict[str, Any]] = None
    group_by: List[str] = None
    order_by: List[str] = None
    limit: Optional[int] = None
    slots: List[SQLSlot] = None
    confidence: float = 0.0

class SQLElementExtractor:
    """
    SQL 요소 추출기 - 규칙 기반 방식
    
    기능:
    1. 한국어 질문에서 SQL 요소 추출
    2. 실제 테이블/컬럼 매핑
    3. 조건절 추출
    4. 집계 함수 식별
    """
    
    def __init__(self):
        """SQL 요소 추출기 초기화"""
        
        # 실제 데이터베이스 스키마 정의
        self.schema = {
            "traffic_intersection": {
                "columns": {
                    "id": ["ID", "식별자", "번호", "교차로ID"],
                    "name": ["이름", "명칭", "교차로명", "교차로이름", "위치명"],
                    "latitude": ["위도", "위도좌표", "lat"],
                    "longitude": ["경도", "경도좌표", "lng", "lon"],
                    "created_at": ["생성일", "생성시간", "등록일"],
                    "updated_at": ["수정일", "수정시간", "업데이트일"]
                },
                "aliases": ["교차로", "교차점", "신호등", "신호", "사거리", "삼거리", "intersection"]
            },
            "traffic_trafficvolume": {
                "columns": {
                    "id": ["ID", "식별자", "번호", "교통량ID"],
                    "intersection_id": ["교차로ID", "교차로번호", "intersection_id"],
                    "datetime": ["시간", "일시", "날짜", "시간대", "timestamp"],
                    "direction": ["방향", "진행방향", "dir"],
                    "volume": ["교통량", "통행량", "차량수", "대수", "traffic_volume"],
                    "is_simulated": ["시뮬레이션", "가상", "실제여부", "simulated"],
                    "created_at": ["생성일", "생성시간", "등록일"],
                    "updated_at": ["수정일", "수정시간", "업데이트일"]
                },
                "aliases": ["교통량", "통행량", "차량수", "traffic", "volume"]
            },
            "traffic_incident": {
                "columns": {
                    "incident_id": ["사고ID", "사건ID", "incident_id"],
                    "incident_type": ["사고유형", "사고종류", "유형", "사건유형"],
                    "intersection_id": ["교차로ID", "교차로번호", "intersection_id"],
                    "district": ["구", "지역", "행정구역", "district"],
                    "intersection_name": ["교차로명", "교차로이름", "intersection_name"],
                    "status": ["상태", "처리상태", "진행상태"],
                    "registered_at": ["등록일", "신고일", "발생일", "registered_at"],
                    "created_at": ["생성일", "생성시간"],
                    "updated_at": ["수정일", "수정시간"]
                },
                "aliases": ["사고", "접촉사고", "교통사고", "사건", "incident", "accident"]
            }
        }
        
        # 지역 키워드 매핑
        self.region_keywords = {
            "세종": ["세종", "세종특별자치시", "sejong"],
            "조치원": ["조치원", "조치원읍", "jochiwon"],
            "부강": ["부강", "부강면", "bugang"],
            "금남": ["금남", "금남면", "geumnam"],
            "전의": ["전의", "전의면", "jeonui"],
            "전동": ["전동", "전동면", "jeondong"],
            "연동": ["연동", "연동면", "yeondong"],
            "연서": ["연서", "연서면", "yeonseo"],
            "장군": ["장군", "장군면", "janggun"],
            "소정": ["소정", "소정면", "sojeong"],
            "한솔": ["한솔", "한솔동", "hansol"],
            "새롬": ["새롬", "새롬동", "saerom"],
            "도담": ["도담", "도담동", "dodam"],
            "아름": ["아름", "아름동", "areum"],
            "종촌": ["종촌", "종촌동", "jongchon"],
            "고운": ["고운", "고운동", "goun"],
            "보람": ["보람", "보람동", "boram"],
            "대평": ["대평", "대평동", "daepyeong"],
            "소담": ["소담", "소담동", "sodam"],
            "반곡": ["반곡", "반곡동", "bangok"],
            "다정": ["다정", "다정동", "dajeong"],
            "어진": ["어진", "어진동", "eojin"],
            "나성": ["나성", "나성동", "naseong"],
            "새뜸": ["새뜸", "새뜸동", "saettum"],
            "다솜": ["다솜", "다솜동", "dasom"],
            "한별": ["한별", "한별동", "hanbyeol"],
            "가람": ["가람", "가람동", "garam"],
            "도움": ["도움", "도움동", "doum"],
            "비전": ["비전", "비전동", "bijeon"],
            "새움": ["새움", "새움동", "saeeum"]
        }
        
        # 시간 관련 키워드
        self.time_keywords = {
            "오늘": "CURDATE()",
            "어제": "DATE_SUB(CURDATE(), INTERVAL 1 DAY)",
            "내일": "DATE_ADD(CURDATE(), INTERVAL 1 DAY)",
            "이번주": "YEARWEEK(CURDATE())",
            "지난주": "YEARWEEK(DATE_SUB(CURDATE(), INTERVAL 1 WEEK))",
            "이번달": "MONTH(CURDATE())",
            "지난달": "MONTH(DATE_SUB(CURDATE(), INTERVAL 1 MONTH))",
            "올해": "YEAR(CURDATE())",
            "작년": "YEAR(DATE_SUB(CURDATE(), INTERVAL 1 YEAR))"
        }
        
        # 집계 함수 키워드
        self.aggregate_keywords = {
            "개수": "COUNT",
            "수": "COUNT", 
            "건수": "COUNT",
            "총합": "SUM",
            "합계": "SUM",
            "평균": "AVG",
            "최대": "MAX",
            "최소": "MIN",
            "최고": "MAX",
            "최저": "MIN"
        }
        
        # 방향 키워드
        self.direction_keywords = {
            "북쪽": "N",
            "남쪽": "S", 
            "동쪽": "E",
            "서쪽": "W",
            "북": "N",
            "남": "S",
            "동": "E", 
            "서": "W"
        }
        
        logger.info("SQL 요소 추출기 초기화 완료")
    
    def extract_elements(self, question: str) -> ExtractedSQLElements:
        """
        질문에서 SQL 요소 추출
        
        Args:
            question: 자연어 질문
            
        Returns:
            추출된 SQL 요소들
        """
        question_lower = question.lower()
        
        # 1. 테이블 선택
        table_name = self._select_table(question_lower)
        
        # 2. 쿼리 타입 결정
        query_type = self._determine_query_type(question_lower)
        
        # 3. 컬럼 추출
        columns = self._extract_columns(question_lower, table_name)
        
        # 4. 조건 추출
        conditions = self._extract_conditions(question_lower, table_name)
        
        # 5. 정렬 조건 추출
        order_by = self._extract_order_by(question_lower, table_name)
        
        # 6. 그룹핑 조건 추출
        group_by = self._extract_group_by(question_lower, table_name)
        
        # 7. 제한 조건 추출
        limit = self._extract_limit(question_lower)
        
        # 8. 신뢰도 계산
        confidence = self._calculate_confidence(question_lower, table_name, columns, conditions)
        
        return ExtractedSQLElements(
            query_type=query_type,
            table_name=table_name,
            columns=columns,
            conditions=conditions,
            order_by=order_by,
            group_by=group_by,
            limit=limit,
            confidence=confidence
        )
    
    def _select_table(self, question: str) -> str:
        """질문에 적합한 테이블 선택"""
        # 키워드 기반 테이블 선택
        if any(keyword in question for keyword in ["교차로", "intersection", "위치", "좌표"]):
            return "traffic_intersection"
        elif any(keyword in question for keyword in ["교통량", "volume", "traffic", "시간", "방향"]):
            return "traffic_trafficvolume"
        elif any(keyword in question for keyword in ["사고", "incident", "사건", "고장", "정체"]):
            return "traffic_incident"
        else:
            # 기본적으로 교통량 테이블 반환
            return "traffic_trafficvolume"
    
    def _determine_query_type(self, question: str) -> QueryType:
        """쿼리 타입 결정"""
        if any(keyword in question for keyword in ["개수", "수", "건수", "몇개"]):
            return QueryType.COUNT
        elif any(keyword in question for keyword in ["총합", "합계", "총"]):
            return QueryType.SUM
        elif any(keyword in question for keyword in ["평균", "평균값"]):
            return QueryType.AVG
        elif any(keyword in question for keyword in ["최대", "최고", "가장많은"]):
            return QueryType.MAX
        elif any(keyword in question for keyword in ["최소", "최저", "가장적은"]):
            return QueryType.MIN
        else:
            return QueryType.SELECT
    
    def _extract_columns(self, question: str, table_name: str) -> List[str]:
        """컬럼 추출"""
        columns = []
        table_schema = self.schema.get(table_name, {})
        
        # 기본 컬럼들
        if table_name == "traffic_intersection":
            columns = ["id", "name", "latitude", "longitude"]
        elif table_name == "traffic_trafficvolume":
            columns = ["intersection_id", "datetime", "direction", "volume"]
        elif table_name == "traffic_incident":
            columns = ["incident_id", "incident_type", "district", "intersection_name", "status"]
        
        # 질문에서 특정 컬럼 요청 확인
        for col_name, keywords in table_schema.get("columns", {}).items():
            if any(keyword in question for keyword in keywords):
                if col_name not in columns:
                    columns.append(col_name)
        
        return columns if columns else ["*"]
    
    def _extract_conditions(self, question: str, table_name: str) -> List[Dict[str, Any]]:
        """조건 추출"""
        conditions = []
        
        # 지역 조건
        for region, keywords in self.region_keywords.items():
            if any(keyword in question for keyword in keywords):
                if table_name == "traffic_intersection":
                    conditions.append({
                        "column": "name",
                        "operator": "LIKE",
                        "value": f"%{region}%"
                    })
                elif table_name == "traffic_incident":
                    conditions.append({
                        "column": "district",
                        "operator": "LIKE", 
                        "value": f"%{region}%"
                    })
                break
        
        # 시간 조건
        for time_keyword, sql_function in self.time_keywords.items():
            if time_keyword in question:
                if table_name == "traffic_trafficvolume":
                    conditions.append({
                        "column": "datetime",
                        "operator": ">=",
                        "value": sql_function
                    })
                elif table_name == "traffic_incident":
                    conditions.append({
                        "column": "registered_at",
                        "operator": ">=",
                        "value": sql_function
                    })
                break
        
        # 방향 조건
        for direction_keyword, direction_code in self.direction_keywords.items():
            if direction_keyword in question and table_name == "traffic_trafficvolume":
                conditions.append({
                    "column": "direction",
                    "operator": "=",
                    "value": f"'{direction_code}'"
                })
                break
        
        return conditions
    
    def _extract_order_by(self, question: str, table_name: str) -> List[str]:
        """정렬 조건 추출"""
        order_by = []
        
        if "최신" in question or "최근" in question:
            if table_name == "traffic_trafficvolume":
                order_by.append("datetime DESC")
            elif table_name == "traffic_incident":
                order_by.append("registered_at DESC")
            elif table_name == "traffic_intersection":
                order_by.append("created_at DESC")
        
        if "오래된" in question or "과거" in question:
            if table_name == "traffic_trafficvolume":
                order_by.append("datetime ASC")
            elif table_name == "traffic_incident":
                order_by.append("registered_at ASC")
            elif table_name == "traffic_intersection":
                order_by.append("created_at ASC")
        
        if "많은" in question or "높은" in question:
            if table_name == "traffic_trafficvolume":
                order_by.append("volume DESC")
        
        if "적은" in question or "낮은" in question:
            if table_name == "traffic_trafficvolume":
                order_by.append("volume ASC")
        
        return order_by
    
    def _extract_group_by(self, question: str, table_name: str) -> List[str]:
        """그룹핑 조건 추출"""
        group_by = []
        
        if "별로" in question or "구분" in question:
            if table_name == "traffic_trafficvolume":
                group_by.append("intersection_id")
            elif table_name == "traffic_incident":
                group_by.append("incident_type")
        
        return group_by
    
    def _extract_limit(self, question: str) -> Optional[int]:
        """제한 조건 추출"""
        # 숫자 패턴 찾기
        number_pattern = r'(\d+)개|(\d+)건|(\d+)개씩|(\d+)개만'
        match = re.search(number_pattern, question)
        if match:
            for group in match.groups():
                if group:
                    return int(group)
        
        # 상위/하위 키워드
        if "상위" in question or "top" in question.lower():
            return 10
        elif "최근" in question or "최신" in question:
            return 5
        
        return None
    
    def _calculate_confidence(self, question: str, table_name: str, columns: List[str], conditions: List[Dict]) -> float:
        """신뢰도 계산"""
        confidence = 0.5  # 기본 신뢰도
        
        # 테이블 매칭 점수
        table_keywords = self.schema.get(table_name, {}).get("aliases", [])
        if any(keyword in question for keyword in table_keywords):
            confidence += 0.2
        
        # 컬럼 매칭 점수
        if columns and columns != ["*"]:
            confidence += 0.1
        
        # 조건 매칭 점수
        if conditions:
            confidence += 0.1
        
        # 지역 키워드 매칭
        for region_keywords in self.region_keywords.values():
            if any(keyword in question for keyword in region_keywords):
                confidence += 0.1
                break
        
        return min(confidence, 1.0)
    
    def generate_sql(self, elements: ExtractedSQLElements) -> str:
        """SQL 생성"""
        sql_parts = []
        
        # SELECT 절
        if elements.query_type == QueryType.COUNT:
            sql_parts.append("SELECT COUNT(*)")
        elif elements.query_type == QueryType.SUM:
            sql_parts.append(f"SELECT SUM({elements.columns[0] if elements.columns and elements.columns != ['*'] else 'volume'})")
        elif elements.query_type == QueryType.AVG:
            sql_parts.append(f"SELECT AVG({elements.columns[0] if elements.columns and elements.columns != ['*'] else 'volume'})")
        elif elements.query_type == QueryType.MAX:
            sql_parts.append(f"SELECT MAX({elements.columns[0] if elements.columns and elements.columns != ['*'] else 'volume'})")
        elif elements.query_type == QueryType.MIN:
            sql_parts.append(f"SELECT MIN({elements.columns[0] if elements.columns and elements.columns != ['*'] else 'volume'})")
        else:
            columns_str = ", ".join(elements.columns) if elements.columns else "*"
            sql_parts.append(f"SELECT {columns_str}")
        
        # FROM 절
        sql_parts.append(f"FROM {elements.table_name}")
        
        # WHERE 절
        if elements.conditions:
            where_conditions = []
            for condition in elements.conditions:
                # 문자열 값에 따옴표 추가
                value = condition['value']
                if condition['operator'] == 'LIKE' and not value.startswith("'"):
                    value = f"'{value}'"
                elif condition['operator'] == '=' and not value.startswith("'"):
                    value = f"'{value}'"
                
                where_conditions.append(f"{condition['column']} {condition['operator']} {value}")
            sql_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        # GROUP BY 절
        if elements.group_by:
            sql_parts.append(f"GROUP BY {', '.join(elements.group_by)}")
        
        # ORDER BY 절
        if elements.order_by:
            sql_parts.append(f"ORDER BY {', '.join(elements.order_by)}")
        
        # LIMIT 절
        if elements.limit:
            sql_parts.append(f"LIMIT {elements.limit}")
        
        return " ".join(sql_parts)
