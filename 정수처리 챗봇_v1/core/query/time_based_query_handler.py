"""
시간 기반 질문 처리 모듈 (비활성화됨)

SQL 관련 기능이 제거되어 현재 비활성화된 상태입니다.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TimeExpression(Enum):
    """시간 표현 유형"""
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"
    CUSTOM_RANGE = "custom_range"

@dataclass
class TimeRange:
    """시간 범위 정보"""
    start_date: str
    end_date: str
    time_expression: TimeExpression
    description: str

@dataclass
class LocationInfo:
    """위치 정보"""
    region: str  # 구/읍/면
    intersection: Optional[str] = None  # 교차로명
    district: Optional[str] = None  # 동/리

@dataclass
class TimeBasedQuery:
    """시간 기반 질문 분석 결과"""
    original_question: str
    time_range: TimeRange
    location: Optional[LocationInfo]
    metrics: List[str]  # 요청된 지표들
    aggregation_type: str  # 요약, 상세, 비교 등
    sql_template: str  # SQL 템플릿

class TimeBasedQueryHandler:
    """시간 기반 질문 처리기"""
    
    def __init__(self):
        """초기화"""
        # 한국어 시간 표현 패턴
        self.time_patterns = {
            '오늘': TimeExpression.TODAY,
            '어제': TimeExpression.YESTERDAY,
            '이번주': TimeExpression.THIS_WEEK,
            '지난주': TimeExpression.LAST_WEEK,
            '1주전': TimeExpression.LAST_WEEK,
            '이번달': TimeExpression.THIS_MONTH,
            '지난달': TimeExpression.LAST_MONTH,
            '올해': TimeExpression.THIS_YEAR,
            '작년': TimeExpression.LAST_YEAR,
            '금주': TimeExpression.THIS_WEEK,
            '전주': TimeExpression.LAST_WEEK,
            '금월': TimeExpression.THIS_MONTH,
            '전월': TimeExpression.LAST_MONTH,
            '금년': TimeExpression.THIS_YEAR,
            '전년': TimeExpression.LAST_YEAR
        }
        
        # 세종시 지역 패턴
        self.location_patterns = {
            '읍': [
                r'조치원읍', r'연서면', r'연동면', r'부강면', r'금남면', 
                r'장군면', r'연기면', r'전의면', r'전동면', r'소정면'
            ],
            '동': [
                r'한솔동', r'새롬동', r'도담동', r'아름동', r'종촌동',
                r'고운동', r'소담동', r'보람동', r'대평동', r'다정동'
            ]
        }
        
        # 교통 관련 지표
        self.traffic_metrics = {
            '통행량': ['traffic_volume', 'vehicle_count'],
            '교통량': ['traffic_volume', 'vehicle_count'],
            '사고': ['accident_count', 'incident_count'],
            '접촉사고': ['contact_accident_count'],
            '신호': ['signal_count', 'intersection_count'],
            '평균': ['avg_traffic_volume', 'avg_vehicle_count'],
            '최대': ['max_traffic_volume', 'max_vehicle_count'],
            '최소': ['min_traffic_volume', 'min_vehicle_count'],
            '총': ['total_traffic_volume', 'total_vehicle_count']
        }
        
        # 집계 유형
        self.aggregation_types = {
            '요약': 'summary',
            '상세': 'detailed',
            '비교': 'comparison',
            '분석': 'analysis',
            '통계': 'statistics',
            '보고': 'report'
        }
        
        logger.info("시간 기반 질문 처리기 초기화 완료")
    
    def parse_time_based_question(self, question: str) -> TimeBasedQuery:
        """
        시간 기반 질문을 파싱하고 분석
        
        Args:
            question: 사용자 질문
            
        Returns:
            분석된 시간 기반 질문 정보
        """
        logger.info(f"시간 기반 질문 파싱: {question}")
        
        # 1. 시간 범위 추출
        time_range = self._extract_time_range(question)
        
        # 2. 위치 정보 추출
        location = self._extract_location(question)
        
        # 3. 지표 추출
        metrics = self._extract_metrics(question)
        
        # 4. 집계 유형 추출
        aggregation_type = self._extract_aggregation_type(question)
        
        # 5. SQL 템플릿 생성
        sql_template = self._generate_sql_template(time_range, location, metrics, aggregation_type)
        
        return TimeBasedQuery(
            original_question=question,
            time_range=time_range,
            location=location,
            metrics=metrics,
            aggregation_type=aggregation_type,
            sql_template=sql_template
        )
    
    def _extract_time_range(self, question: str) -> TimeRange:
        """시간 범위 추출"""
        for pattern, time_expr in self.time_patterns.items():
            if pattern in question:
                start_date, end_date = self._resolve_time_range(time_expr)
                description = f"{pattern} ({start_date} ~ {end_date})"
                
                return TimeRange(
                    start_date=start_date,
                    end_date=end_date,
                    time_expression=time_expr,
                    description=description
                )
        
        # 기본값: 최근 7일
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        return TimeRange(
            start_date=start_date,
            end_date=end_date,
            time_expression=TimeExpression.CUSTOM_RANGE,
            description=f"최근 7일 ({start_date} ~ {end_date})"
        )
    
    def _resolve_time_range(self, time_expr: TimeExpression) -> Tuple[str, str]:
        """시간 표현을 실제 날짜 범위로 변환"""
        now = datetime.now()
        
        if time_expr == TimeExpression.TODAY:
            date_str = now.strftime('%Y-%m-%d')
            return date_str, date_str
        
        elif time_expr == TimeExpression.YESTERDAY:
            yesterday = now - timedelta(days=1)
            date_str = yesterday.strftime('%Y-%m-%d')
            return date_str, date_str
        
        elif time_expr == TimeExpression.THIS_WEEK:
            start_of_week = now - timedelta(days=now.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            return start_of_week.strftime('%Y-%m-%d'), end_of_week.strftime('%Y-%m-%d')
        
        elif time_expr == TimeExpression.LAST_WEEK:
            start_of_last_week = now - timedelta(days=now.weekday() + 7)
            end_of_last_week = start_of_last_week + timedelta(days=6)
            return start_of_last_week.strftime('%Y-%m-%d'), end_of_last_week.strftime('%Y-%m-%d')
        
        elif time_expr == TimeExpression.THIS_MONTH:
            start_of_month = now.replace(day=1)
            return start_of_month.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')
        
        elif time_expr == TimeExpression.LAST_MONTH:
            if now.month == 1:
                last_month = now.replace(year=now.year-1, month=12)
            else:
                last_month = now.replace(month=now.month-1)
            start_of_last_month = last_month.replace(day=1)
            return start_of_last_month.strftime('%Y-%m-%d'), last_month.strftime('%Y-%m-%d')
        
        elif time_expr == TimeExpression.THIS_YEAR:
            start_of_year = now.replace(month=1, day=1)
            return start_of_year.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')
        
        elif time_expr == TimeExpression.LAST_YEAR:
            last_year = now.replace(year=now.year-1)
            start_of_last_year = last_year.replace(month=1, day=1)
            end_of_last_year = last_year.replace(month=12, day=31)
            return start_of_last_year.strftime('%Y-%m-%d'), end_of_last_year.strftime('%Y-%m-%d')
        
        # 기본값: 최근 7일
        end_date = now.strftime('%Y-%m-%d')
        start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
        return start_date, end_date
    
    def _extract_location(self, question: str) -> Optional[LocationInfo]:
        """위치 정보 추출"""
        # 세종시 지역명 검색
        for region_type, patterns in self.location_patterns.items():
            for pattern in patterns:
                if pattern in question:
                    return LocationInfo(
                        region=pattern,
                        intersection=None,
                        district=None
                    )
        
        # 교차로명 검색
        intersection_match = re.search(r'([가-힣]+교차로)', question)
        if intersection_match:
            intersection_name = intersection_match.group(1)
            return LocationInfo(
                region=None,
                intersection=intersection_name,
                district=None
            )
        
        return None
    
    def _extract_metrics(self, question: str) -> List[str]:
        """요청된 지표 추출"""
        metrics = []
        
        for metric_name, metric_fields in self.traffic_metrics.items():
            if metric_name in question:
                metrics.extend(metric_fields)
        
        # 기본 지표 (지정되지 않은 경우)
        if not metrics:
            metrics = ['traffic_volume', 'vehicle_count']
        
        return list(set(metrics))  # 중복 제거
    
    def _extract_aggregation_type(self, question: str) -> str:
        """집계 유형 추출"""
        for agg_name, agg_type in self.aggregation_types.items():
            if agg_name in question:
                return agg_type
        
        return 'summary'  # 기본값
    
    def _generate_sql_template(self, 
                             time_range: TimeRange, 
                             location: Optional[LocationInfo],
                             metrics: List[str],
                             aggregation_type: str) -> str:
        """SQL 템플릿 생성"""
        
        # 기본 SELECT 절
        select_clause = self._build_select_clause(metrics, aggregation_type)
        
        # FROM 절
        from_clause = "FROM traffic_trafficvolume"
        
        # WHERE 절
        where_conditions = []
        
        # 시간 조건
        where_conditions.append(f"date BETWEEN '{time_range.start_date}' AND '{time_range.end_date}'")
        
        # 위치 조건
        if location:
            if location.region:
                where_conditions.append(f"region = '{location.region}'")
            elif location.intersection:
                where_conditions.append(f"intersection_name = '{location.intersection}'")
        
        where_clause = " AND ".join(where_conditions)
        
        # GROUP BY 절 (집계 유형에 따라)
        group_by_clause = ""
        if aggregation_type in ['summary', 'statistics']:
            group_by_clause = "GROUP BY date"
        elif aggregation_type == 'detailed':
            group_by_clause = "GROUP BY date, region, intersection_name"
        
        # ORDER BY 절
        order_by_clause = "ORDER BY date DESC"
        
        # LIMIT 절
        limit_clause = ""
        if aggregation_type == 'summary':
            limit_clause = "LIMIT 10"
        
        # SQL 조합
        sql = f"""
        {select_clause}
        {from_clause}
        WHERE {where_clause}
        {group_by_clause}
        {order_by_clause}
        {limit_clause}
        """.strip()
        
        return sql
    
    def _build_select_clause(self, metrics: List[str], aggregation_type: str) -> str:
        """SELECT 절 구성"""
        if aggregation_type == 'summary':
            # 요약: 기본 집계 함수 사용
            select_parts = ["date"]
            
            for metric in metrics:
                if 'traffic_volume' in metric or 'vehicle_count' in metric:
                    select_parts.extend([
                        f"SUM({metric}) as total_{metric}",
                        f"AVG({metric}) as avg_{metric}",
                        f"MAX({metric}) as max_{metric}",
                        f"MIN({metric}) as min_{metric}"
                    ])
                elif 'accident' in metric or 'incident' in metric:
                    select_parts.extend([
                        f"SUM({metric}) as total_{metric}",
                        f"COUNT({metric}) as count_{metric}"
                    ])
            
            return f"SELECT {', '.join(select_parts)}"
        
        elif aggregation_type == 'detailed':
            # 상세: 모든 필드 포함
            select_parts = [
                "date", "region", "intersection_name", "time_period"
            ]
            select_parts.extend(metrics)
            return f"SELECT {', '.join(select_parts)}"
        
        else:
            # 기본: 단순 집계
            select_parts = ["date"]
            for metric in metrics:
                select_parts.append(f"SUM({metric}) as total_{metric}")
            return f"SELECT {', '.join(select_parts)}"
    
    def generate_final_sql(self, query_info: TimeBasedQuery) -> str:
        """최종 SQL 쿼리 생성"""
        # SQL 템플릿을 실제 쿼리로 변환
        sql = query_info.sql_template
        
        # 쿼리 정리
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        logger.info(f"생성된 SQL: {sql}")
        return sql
    
    def analyze_and_generate_sql(self, question: str) -> Dict[str, Any]:
        """
        질문을 분석하고 SQL을 생성하는 통합 메서드
        
        Args:
            question: 사용자 질문
            
        Returns:
            분석 결과와 SQL 쿼리
        """
        try:
            # 1. 시간 기반 질문 파싱
            query_info = self.parse_time_based_question(question)
            
            # 2. SQL 생성
            sql_query = self.generate_final_sql(query_info)
            
            return {
                'success': True,
                'query_info': query_info,
                'sql_query': sql_query,
                'time_range': {
                    'start_date': query_info.time_range.start_date,
                    'end_date': query_info.time_range.end_date,
                    'description': query_info.time_range.description
                },
                'location': query_info.location.region if query_info.location else None,
                'metrics': query_info.metrics,
                'aggregation_type': query_info.aggregation_type
            }
            
        except Exception as e:
            logger.error(f"시간 기반 질문 처리 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'sql_query': None
            }
