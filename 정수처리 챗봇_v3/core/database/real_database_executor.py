"""
실제 데이터베이스 실행기

Docker MySQL 데이터베이스에 연결하여 SQL을 실행하고 결과를 반환
"""

import os
import pymysql
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class RealDatabaseExecutor:
    """실제 데이터베이스 실행기"""
    
    def __init__(self):
        """초기화"""
        # Docker 환경의 데이터베이스 연결 설정
        self.db_config = {
            'host': os.getenv('MYSQL_HOST', 'localhost'),  # Docker 컨테이너에서 접근할 때는 localhost
            'user': os.getenv('MYSQL_USER', 'root'),
            'password': os.getenv('MYSQL_PASSWORD', '1234'),
            'database': os.getenv('MYSQL_DATABASE', 'traffic'),
            'charset': 'utf8mb4',
            'port': int(os.getenv('MYSQL_PORT', 3307))  # Docker 포트 매핑: 3307
        }
        
        # 연결 상태
        self.connection = None
        self.is_connected = False
        
        logger.info(f"실제 데이터베이스 실행기 초기화: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
    
    def connect(self) -> bool:
        """데이터베이스 연결"""
        try:
            if self.connection is None or not self.is_connected:
                self.connection = pymysql.connect(**self.db_config)
                self.is_connected = True
                logger.info("데이터베이스 연결 성공")
                return True
            return True
        except Exception as e:
            logger.error(f"✗ 데이터베이스 연결 실패: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """데이터베이스 연결 해제"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self.is_connected = False
                logger.info("데이터베이스 연결 해제")
        except Exception as e:
            logger.warning(f"데이터베이스 연결 해제 중 오류: {e}")
    
    def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        SQL 쿼리를 실제 데이터베이스에서 실행
        
        Args:
            sql_query: 실행할 SQL 쿼리
            
        Returns:
            실행 결과 딕셔너리
        """
        if not self.connect():
            return {
                'success': False,
                'error': '데이터베이스 연결 실패',
                'data': None,
                'row_count': 0
            }
        
        try:
            cursor = self.connection.cursor(pymysql.cursors.DictCursor)
            
            logger.info(f"SQL 실행: {sql_query}")
            
            # 쿼리 실행
            cursor.execute(sql_query)
            
            # 결과 가져오기
            results = cursor.fetchall()
            
            # 결과를 JSON 직렬화 가능한 형태로 변환
            processed_results = []
            for row in results:
                processed_row = {}
                for key, value in row.items():
                    if isinstance(value, datetime):
                        processed_row[key] = value.isoformat()
                    elif isinstance(value, timedelta):
                        processed_row[key] = str(value)
                    else:
                        processed_row[key] = value
                processed_results.append(processed_row)
            
            cursor.close()
            
            return {
                'success': True,
                'data': processed_results,
                'row_count': len(processed_results),
                'sql_query': sql_query
            }
            
        except Exception as e:
            logger.error(f"SQL 실행 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'row_count': 0,
                'sql_query': sql_query
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """데이터베이스 연결 테스트"""
        try:
            if not self.connect():
                return {
                    'success': False,
                    'error': '데이터베이스 연결 실패',
                    'details': None
                }
            
            cursor = self.connection.cursor(pymysql.cursors.DictCursor)
            
            # 테이블 목록 확인
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            table_list = [list(table.values())[0] for table in tables]
            
            # 주요 테이블 존재 확인
            main_tables = ['traffic_intersection', 'traffic_trafficvolume', 'traffic_incident']
            available_tables = [table for table in main_tables if table in table_list]
            
            # 데이터 개수 확인
            table_counts = {}
            for table in available_tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                result = cursor.fetchone()
                table_counts[table] = result['count'] if result else 0
            
            # 세종 지역 데이터 확인
            sejong_intersection_count = 0
            sejong_incident_count = 0
            
            if 'traffic_intersection' in available_tables:
                cursor.execute("SELECT COUNT(*) as count FROM traffic_intersection WHERE name LIKE '%세종%'")
                result = cursor.fetchone()
                sejong_intersection_count = result['count'] if result else 0
            
            if 'traffic_incident' in available_tables:
                cursor.execute("SELECT COUNT(*) as count FROM traffic_incident WHERE district LIKE '%세종%' OR intersection_name LIKE '%세종%'")
                result = cursor.fetchone()
                sejong_incident_count = result['count'] if result else 0
            
            cursor.close()
            
            return {
                'success': True,
                'details': {
                    'tables': table_list,
                    'available_main_tables': available_tables,
                    'table_counts': table_counts,
                    'sejong_intersection_count': sejong_intersection_count,
                    'sejong_incident_count': sejong_incident_count
                }
            }
            
        except Exception as e:
            logger.error(f"연결 테스트 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'details': None
            }
    
    def get_intersection_data(self, region: str = '세종', limit: int = 10) -> Dict[str, Any]:
        """교차로 데이터 조회"""
        sql = f"""
        SELECT id, name, latitude, longitude, created_at, updated_at
        FROM traffic_intersection 
        WHERE name LIKE '%{region}%'
        ORDER BY name 
        LIMIT {limit}
        """
        
        return self.execute_sql(sql)
    
    def get_traffic_volume_data(self, intersection_id: Optional[int] = None, limit: int = 10) -> Dict[str, Any]:
        """교통량 데이터 조회"""
        if intersection_id:
            sql = f"""
            SELECT tv.*, ti.name as intersection_name
            FROM traffic_trafficvolume tv
            JOIN traffic_intersection ti ON tv.intersection_id = ti.id
            WHERE tv.intersection_id = {intersection_id}
            ORDER BY tv.datetime DESC 
            LIMIT {limit}
            """
        else:
            sql = f"""
            SELECT tv.*, ti.name as intersection_name
            FROM traffic_trafficvolume tv
            JOIN traffic_intersection ti ON tv.intersection_id = ti.id
            ORDER BY tv.datetime DESC 
            LIMIT {limit}
            """
        
        return self.execute_sql(sql)
    
    def get_incident_data(self, region: str = '세종', limit: int = 10) -> Dict[str, Any]:
        """사고 데이터 조회"""
        sql = f"""
        SELECT incident_id, incident_type, district, intersection_name, status, registered_at, created_at
        FROM traffic_incident 
        WHERE district LIKE '%{region}%' OR intersection_name LIKE '%{region}%'
        ORDER BY registered_at DESC 
        LIMIT {limit}
        """
        
        return self.execute_sql(sql)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """데이터 요약 정보 조회"""
        summary_queries = {
            'intersection_summary': """
                SELECT 
                    COUNT(*) as total_intersections,
                    COUNT(CASE WHEN name LIKE '%세종%' THEN 1 END) as sejong_intersections,
                    MIN(created_at) as earliest_intersection,
                    MAX(created_at) as latest_intersection
                FROM traffic_intersection
            """,
            'traffic_volume_summary': """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT intersection_id) as unique_intersections,
                    COUNT(CASE WHEN is_simulated = 1 THEN 1 END) as simulated_records,
                    COUNT(CASE WHEN is_simulated = 0 THEN 1 END) as real_records,
                    MIN(datetime) as earliest_record,
                    MAX(datetime) as latest_record,
                    AVG(volume) as avg_volume,
                    MAX(volume) as max_volume
                FROM traffic_trafficvolume
            """,
            'incident_summary': """
                SELECT 
                    COUNT(*) as total_incidents,
                    COUNT(DISTINCT incident_type) as unique_incident_types,
                    COUNT(DISTINCT district) as unique_districts,
                    COUNT(CASE WHEN status = 'OPEN' THEN 1 END) as open_incidents,
                    COUNT(CASE WHEN status = 'CLOSED' THEN 1 END) as closed_incidents,
                    COUNT(CASE WHEN status = 'IN_PROGRESS' THEN 1 END) as in_progress_incidents,
                    MIN(registered_at) as earliest_incident,
                    MAX(registered_at) as latest_incident
                FROM traffic_incident
            """
        }
        
        summary_results = {}
        for key, sql in summary_queries.items():
            result = self.execute_sql(sql)
            if result['success']:
                summary_results[key] = result['data'][0] if result['data'] else {}
            else:
                summary_results[key] = {'error': result['error']}
        
        return {
            'success': True,
            'data': summary_results
        }
    
    def execute_time_based_query(self, 
                               time_range: Dict[str, str], 
                               location: Optional[str] = None,
                               metrics: List[str] = None) -> Dict[str, Any]:
        """
        시간 기반 쿼리 실행
        
        Args:
            time_range: 시간 범위 {'start_date': '2025-07-10', 'end_date': '2025-07-11'}
            location: 위치 (예: '세종')
            metrics: 조회할 지표들
            
        Returns:
            쿼리 실행 결과
        """
        if not metrics:
            metrics = ['volume']
        
        # SELECT 절 구성
        select_parts = ["DATE(tv.datetime) as date", "HOUR(tv.datetime) as hour", "tv.direction"]
        for metric in metrics:
            select_parts.extend([
                f"SUM(tv.{metric}) as total_{metric}",
                f"AVG(tv.{metric}) as avg_{metric}",
                f"MAX(tv.{metric}) as max_{metric}",
                f"MIN(tv.{metric}) as min_{metric}"
            ])
        
        select_clause = ", ".join(select_parts)
        
        # WHERE 절 구성
        where_conditions = [f"tv.datetime BETWEEN '{time_range['start_date']}' AND '{time_range['end_date']}'"]
        
        if location:
            where_conditions.append(f"ti.name LIKE '%{location}%'")
        
        where_clause = " AND ".join(where_conditions)
        
        # SQL 쿼리 생성
        sql = f"""
        SELECT {select_clause}
        FROM traffic_trafficvolume tv
        JOIN traffic_intersection ti ON tv.intersection_id = ti.id
        WHERE {where_clause}
        GROUP BY DATE(tv.datetime), HOUR(tv.datetime), tv.direction
        ORDER BY date DESC, hour DESC, tv.direction
        """
        
        return self.execute_sql(sql)
    
    def get_intersection_traffic_stats(self, intersection_id: int) -> Dict[str, Any]:
        """특정 교차로의 교통량 통계"""
        sql = f"""
        SELECT 
            ti.name as intersection_name,
            COUNT(tv.id) as total_records,
            AVG(tv.volume) as avg_volume,
            MAX(tv.volume) as max_volume,
            MIN(tv.volume) as min_volume,
            COUNT(CASE WHEN tv.is_simulated = 1 THEN 1 END) as simulated_count,
            COUNT(CASE WHEN tv.is_simulated = 0 THEN 1 END) as real_count,
            MIN(tv.datetime) as first_record,
            MAX(tv.datetime) as last_record,
            COUNT(DISTINCT tv.direction) as direction_count,
            GROUP_CONCAT(DISTINCT tv.direction) as directions
        FROM traffic_intersection ti
        LEFT JOIN traffic_trafficvolume tv ON ti.id = tv.intersection_id
        WHERE ti.id = {intersection_id}
        GROUP BY ti.id, ti.name
        """
        
        return self.execute_sql(sql)
    
    def get_incident_stats_by_type(self) -> Dict[str, Any]:
        """사고 유형별 통계"""
        sql = """
        SELECT 
            incident_type,
            COUNT(*) as total_count,
            COUNT(DISTINCT district) as district_count,
            COUNT(CASE WHEN status = 'OPEN' THEN 1 END) as open_count,
            COUNT(CASE WHEN status = 'CLOSED' THEN 1 END) as closed_count,
            COUNT(CASE WHEN status = 'IN_PROGRESS' THEN 1 END) as in_progress_count,
            MIN(registered_at) as earliest_incident,
            MAX(registered_at) as latest_incident
        FROM traffic_incident
        GROUP BY incident_type
        ORDER BY total_count DESC
        """
        
        return self.execute_sql(sql)
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.disconnect()
