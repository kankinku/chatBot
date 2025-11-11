"""
동적 스키마 관리자

실시간으로 데이터베이스 스키마를 학습하고 캐싱하여 SQL 생성 정확도를 높이는 모듈
"""

import os
import json
import logging
import pymysql
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ColumnInfo:
    """컬럼 정보"""
    name: str
    type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    default_value: Optional[str] = None
    max_length: Optional[int] = None
    description: Optional[str] = None
    sample_values: List[Any] = field(default_factory=list)

@dataclass
class ForeignKeyInfo:
    """외래키 정보"""
    column_name: str
    referenced_table: str
    referenced_column: str
    constraint_name: str

@dataclass
class TableInfo:
    """테이블 정보"""
    name: str
    columns: Dict[str, ColumnInfo]
    primary_keys: List[str]
    foreign_keys: List[ForeignKeyInfo]
    indexes: List[str]
    row_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None

@dataclass
class DatabaseSchema:
    """데이터베이스 스키마"""
    database_name: str
    tables: Dict[str, TableInfo]
    relationships: Dict[str, List[str]]  # table_name -> [related_table_names]
    last_analyzed: datetime = field(default_factory=datetime.now)

class DynamicSchemaManager:
    """
    동적 스키마 관리자
    
    기능:
    1. 데이터베이스 스키마 자동 발견
    2. 테이블 관계 추론
    3. 컬럼 의미 분석
    4. 스키마 캐싱 및 업데이트
    5. 샘플 데이터 수집
    """
    
    def __init__(self, cache_duration_hours: int = 24):
        """동적 스키마 관리자 초기화"""
        
        # 데이터베이스 연결 설정
        self.db_config = {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'user': os.getenv('MYSQL_USER', 'root'),
            'password': os.getenv('MYSQL_PASSWORD', '1234'),
            'database': os.getenv('MYSQL_DATABASE', 'traffic'),
            'charset': 'utf8mb4',
            'port': int(os.getenv('MYSQL_PORT', 3307)),
            'connect_timeout': 10,
            'read_timeout': 30,
            'write_timeout': 30
        }
        
        # 캐시 설정
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_file = Path("cache/database_schema.json")
        self.cache_file.parent.mkdir(exist_ok=True)
        
        # 스키마 캐시
        self.cached_schema: Optional[DatabaseSchema] = None
        
        # 컬럼 의미 매핑 (한국어 -> 영어 컬럼명)
        self.column_semantic_mapping = {
            # 공통 컬럼
            '아이디': 'id',
            '식별자': 'id',
            '번호': 'id',
            '이름': 'name',
            '명칭': 'name',
            '제목': 'title',
            '내용': 'content',
            '설명': 'description',
            '생성일': 'created_at',
            '수정일': 'updated_at',
            '등록일': 'registered_at',
            '삭제일': 'deleted_at',
            
            # 교통 관련
            '교차로': 'intersection',
            '교통량': 'volume',
            '통행량': 'volume',
            '방향': 'direction',
            '위도': 'latitude',
            '경도': 'longitude',
            '사고': 'incident',
            '상태': 'status',
            '유형': 'type',
            '지역': 'district',
            '구역': 'district',
            
            # 시간 관련
            '시간': 'datetime',
            '날짜': 'date',
            '년도': 'year',
            '월': 'month',
            '일': 'day',
            '시': 'hour',
            '분': 'minute',
            '초': 'second',
        }
        
        # 테이블 카테고리 매핑
        self.table_categories = {
            'traffic_intersection': '교통 인프라',
            'traffic_trafficvolume': '교통량 데이터',
            'traffic_incident': '교통 사고',
            'user': '사용자 관리',
            'log': '시스템 로그',
            'config': '설정 관리'
        }
        
        logger.info("동적 스키마 관리자 초기화 완료")
    
    def get_schema(self, force_refresh: bool = False) -> DatabaseSchema:
        """
        스키마 조회 (캐시 우선)
        
        Args:
            force_refresh: 강제 갱신 여부
            
        Returns:
            데이터베이스 스키마
        """
        # 캐시된 스키마 확인
        if not force_refresh and self._is_cache_valid():
            if self.cached_schema:
                logger.info("캐시된 스키마 사용")
                return self.cached_schema
            
            # 파일 캐시 확인
            cached_schema = self._load_schema_from_cache()
            if cached_schema:
                self.cached_schema = cached_schema
                logger.info("파일 캐시에서 스키마 로드")
                return cached_schema
        
        # 스키마 새로 분석
        logger.info("데이터베이스 스키마 분석 시작")
        schema = self._analyze_database_schema()
        
        # 캐시에 저장
        self.cached_schema = schema
        self._save_schema_to_cache(schema)
        
        logger.info(f"스키마 분석 완료: {len(schema.tables)}개 테이블")
        return schema
    
    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        if not self.cache_file.exists():
            return False
        
        cache_time = datetime.fromtimestamp(self.cache_file.stat().st_mtime)
        return datetime.now() - cache_time < self.cache_duration
    
    def _load_schema_from_cache(self) -> Optional[DatabaseSchema]:
        """캐시에서 스키마 로드"""
        try:
            if not self.cache_file.exists():
                return None
            
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON에서 DatabaseSchema 객체로 변환
            schema = self._deserialize_schema(data)
            return schema
            
        except Exception as e:
            logger.error(f"스키마 캐시 로드 실패: {e}")
            return None
    
    def _save_schema_to_cache(self, schema: DatabaseSchema):
        """스키마를 캐시에 저장"""
        try:
            # DatabaseSchema 객체를 JSON으로 직렬화
            data = self._serialize_schema(schema)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("스키마 캐시 저장 완료")
            
        except Exception as e:
            logger.error(f"스키마 캐시 저장 실패: {e}")
    
    def _analyze_database_schema(self) -> DatabaseSchema:
        """데이터베이스 스키마 분석"""
        try:
            connection = pymysql.connect(**self.db_config)
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            
            # 데이터베이스 이름
            database_name = self.db_config['database']
            
            # 테이블 목록 조회
            cursor.execute("SHOW TABLES")
            table_rows = cursor.fetchall()
            table_names = [list(row.values())[0] for row in table_rows]
            
            tables = {}
            relationships = {}
            
            for table_name in table_names:
                logger.info(f"테이블 분석 중: {table_name}")
                
                # 테이블 정보 분석
                table_info = self._analyze_table(cursor, table_name)
                tables[table_name] = table_info
                
                # 관계 추론
                related_tables = self._infer_relationships(cursor, table_name, table_names)
                relationships[table_name] = related_tables
            
            cursor.close()
            connection.close()
            
            schema = DatabaseSchema(
                database_name=database_name,
                tables=tables,
                relationships=relationships,
                last_analyzed=datetime.now()
            )
            
            return schema
            
        except Exception as e:
            logger.error(f"데이터베이스 스키마 분석 실패: {e}")
            raise
    
    def _analyze_table(self, cursor, table_name: str) -> TableInfo:
        """테이블 분석"""
        # 컬럼 정보 조회
        cursor.execute(f"DESCRIBE {table_name}")
        column_rows = cursor.fetchall()
        
        columns = {}
        primary_keys = []
        
        for row in column_rows:
            column_name = row['Field']
            column_type = row['Type']
            is_nullable = row['Null'] == 'YES'
            is_primary_key = row['Key'] == 'PRI'
            default_value = row['Default']
            
            if is_primary_key:
                primary_keys.append(column_name)
            
            # 컬럼 타입에서 최대 길이 추출
            max_length = None
            if '(' in column_type:
                try:
                    length_part = column_type.split('(')[1].split(')')[0]
                    max_length = int(length_part.split(',')[0])  # VARCHAR(255,0) 형태 처리
                except:
                    pass
            
            # 샘플 데이터 수집
            sample_values = self._collect_sample_values(cursor, table_name, column_name)
            
            # 컬럼 의미 추론
            description = self._infer_column_meaning(column_name, column_type, sample_values)
            
            column_info = ColumnInfo(
                name=column_name,
                type=column_type,
                is_nullable=is_nullable,
                is_primary_key=is_primary_key,
                is_foreign_key=False,  # 나중에 업데이트
                default_value=default_value,
                max_length=max_length,
                description=description,
                sample_values=sample_values
            )
            
            columns[column_name] = column_info
        
        # 외래키 정보 조회
        foreign_keys = self._get_foreign_keys(cursor, table_name)
        
        # 외래키 정보를 컬럼 정보에 반영
        for fk in foreign_keys:
            if fk.column_name in columns:
                columns[fk.column_name].is_foreign_key = True
        
        # 인덱스 정보 조회
        indexes = self._get_indexes(cursor, table_name)
        
        # 행 개수 조회
        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        row_count = cursor.fetchone()['count']
        
        return TableInfo(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            indexes=indexes,
            row_count=row_count,
            last_updated=datetime.now(),
            description=self.table_categories.get(table_name, "사용자 정의 테이블")
        )
    
    def _collect_sample_values(self, cursor, table_name: str, column_name: str, limit: int = 5) -> List[Any]:
        """샘플 데이터 수집"""
        try:
            cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT {limit}")
            rows = cursor.fetchall()
            return [row[column_name] for row in rows]
        except Exception as e:
            logger.warning(f"샘플 데이터 수집 실패 ({table_name}.{column_name}): {e}")
            return []
    
    def _infer_column_meaning(self, column_name: str, column_type: str, sample_values: List[Any]) -> str:
        """컬럼 의미 추론"""
        # 컬럼 이름 기반 의미 추론
        column_lower = column_name.lower()
        
        if 'id' in column_lower:
            return "식별자"
        elif 'name' in column_lower:
            return "이름"
        elif 'date' in column_lower or 'time' in column_lower:
            return "날짜/시간"
        elif 'latitude' in column_lower or 'lat' in column_lower:
            return "위도"
        elif 'longitude' in column_lower or 'lng' in column_lower or 'lon' in column_lower:
            return "경도"
        elif 'volume' in column_lower:
            return "교통량/통행량"
        elif 'direction' in column_lower:
            return "방향"
        elif 'status' in column_lower:
            return "상태"
        elif 'type' in column_lower:
            return "유형"
        elif 'district' in column_lower:
            return "지역/구역"
        
        # 컬럼 타입 기반 추론
        if 'varchar' in column_type.lower():
            return "문자열"
        elif 'int' in column_type.lower():
            return "정수"
        elif 'decimal' in column_type.lower() or 'float' in column_type.lower():
            return "실수"
        elif 'datetime' in column_type.lower() or 'timestamp' in column_type.lower():
            return "날짜시간"
        elif 'date' in column_type.lower():
            return "날짜"
        elif 'time' in column_type.lower():
            return "시간"
        elif 'text' in column_type.lower():
            return "긴 텍스트"
        elif 'boolean' in column_type.lower() or 'tinyint(1)' in column_type.lower():
            return "참/거짓"
        
        return "일반 데이터"
    
    def _get_foreign_keys(self, cursor, table_name: str) -> List[ForeignKeyInfo]:
        """외래키 정보 조회"""
        try:
            query = """
            SELECT 
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME,
                CONSTRAINT_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
            WHERE TABLE_SCHEMA = %s 
              AND TABLE_NAME = %s 
              AND REFERENCED_TABLE_NAME IS NOT NULL
            """
            
            cursor.execute(query, (self.db_config['database'], table_name))
            rows = cursor.fetchall()
            
            foreign_keys = []
            for row in rows:
                fk = ForeignKeyInfo(
                    column_name=row['COLUMN_NAME'],
                    referenced_table=row['REFERENCED_TABLE_NAME'],
                    referenced_column=row['REFERENCED_COLUMN_NAME'],
                    constraint_name=row['CONSTRAINT_NAME']
                )
                foreign_keys.append(fk)
            
            return foreign_keys
            
        except Exception as e:
            logger.warning(f"외래키 정보 조회 실패 ({table_name}): {e}")
            return []
    
    def _get_indexes(self, cursor, table_name: str) -> List[str]:
        """인덱스 정보 조회"""
        try:
            cursor.execute(f"SHOW INDEX FROM {table_name}")
            rows = cursor.fetchall()
            
            indexes = list(set(row['Key_name'] for row in rows if row['Key_name'] != 'PRIMARY'))
            return indexes
            
        except Exception as e:
            logger.warning(f"인덱스 정보 조회 실패 ({table_name}): {e}")
            return []
    
    def _infer_relationships(self, cursor, table_name: str, all_tables: List[str]) -> List[str]:
        """테이블 관계 추론"""
        related_tables = []
        
        # 외래키 기반 관계
        foreign_keys = self._get_foreign_keys(cursor, table_name)
        for fk in foreign_keys:
            if fk.referenced_table in all_tables:
                related_tables.append(fk.referenced_table)
        
        # 이름 패턴 기반 관계 추론
        if table_name.startswith('traffic_'):
            # 교통 관련 테이블들은 서로 관련
            for other_table in all_tables:
                if other_table != table_name and other_table.startswith('traffic_'):
                    if other_table not in related_tables:
                        related_tables.append(other_table)
        
        return related_tables
    
    def _serialize_schema(self, schema: DatabaseSchema) -> Dict[str, Any]:
        """스키마 직렬화"""
        return {
            'database_name': schema.database_name,
            'tables': {
                table_name: {
                    'name': table_info.name,
                    'columns': {
                        col_name: {
                            'name': col_info.name,
                            'type': col_info.type,
                            'is_nullable': col_info.is_nullable,
                            'is_primary_key': col_info.is_primary_key,
                            'is_foreign_key': col_info.is_foreign_key,
                            'default_value': col_info.default_value,
                            'max_length': col_info.max_length,
                            'description': col_info.description,
                            'sample_values': col_info.sample_values
                        } for col_name, col_info in table_info.columns.items()
                    },
                    'primary_keys': table_info.primary_keys,
                    'foreign_keys': [
                        {
                            'column_name': fk.column_name,
                            'referenced_table': fk.referenced_table,
                            'referenced_column': fk.referenced_column,
                            'constraint_name': fk.constraint_name
                        } for fk in table_info.foreign_keys
                    ],
                    'indexes': table_info.indexes,
                    'row_count': table_info.row_count,
                    'last_updated': table_info.last_updated.isoformat(),
                    'description': table_info.description
                } for table_name, table_info in schema.tables.items()
            },
            'relationships': schema.relationships,
            'last_analyzed': schema.last_analyzed.isoformat()
        }
    
    def _deserialize_schema(self, data: Dict[str, Any]) -> DatabaseSchema:
        """스키마 역직렬화"""
        tables = {}
        
        for table_name, table_data in data['tables'].items():
            columns = {}
            for col_name, col_data in table_data['columns'].items():
                columns[col_name] = ColumnInfo(
                    name=col_data['name'],
                    type=col_data['type'],
                    is_nullable=col_data['is_nullable'],
                    is_primary_key=col_data['is_primary_key'],
                    is_foreign_key=col_data['is_foreign_key'],
                    default_value=col_data['default_value'],
                    max_length=col_data['max_length'],
                    description=col_data['description'],
                    sample_values=col_data['sample_values']
                )
            
            foreign_keys = []
            for fk_data in table_data['foreign_keys']:
                foreign_keys.append(ForeignKeyInfo(
                    column_name=fk_data['column_name'],
                    referenced_table=fk_data['referenced_table'],
                    referenced_column=fk_data['referenced_column'],
                    constraint_name=fk_data['constraint_name']
                ))
            
            tables[table_name] = TableInfo(
                name=table_data['name'],
                columns=columns,
                primary_keys=table_data['primary_keys'],
                foreign_keys=foreign_keys,
                indexes=table_data['indexes'],
                row_count=table_data['row_count'],
                last_updated=datetime.fromisoformat(table_data['last_updated']),
                description=table_data['description']
            )
        
        return DatabaseSchema(
            database_name=data['database_name'],
            tables=tables,
            relationships=data['relationships'],
            last_analyzed=datetime.fromisoformat(data['last_analyzed'])
        )
    
    def get_table_suggestions(self, keywords: List[str]) -> List[Tuple[str, float]]:
        """키워드 기반 테이블 추천"""
        schema = self.get_schema()
        suggestions = []
        
        for table_name, table_info in schema.tables.items():
            score = 0.0
            
            # 테이블 이름 매칭
            for keyword in keywords:
                if keyword.lower() in table_name.lower():
                    score += 1.0
                
                # 테이블 설명 매칭
                if table_info.description and keyword.lower() in table_info.description.lower():
                    score += 0.5
                
                # 컬럼 이름 매칭
                for col_name in table_info.columns:
                    if keyword.lower() in col_name.lower():
                        score += 0.3
            
            if score > 0:
                suggestions.append((table_name, score))
        
        # 점수 순으로 정렬
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions
    
    def get_column_suggestions(self, table_name: str, keywords: List[str]) -> List[Tuple[str, float]]:
        """키워드 기반 컬럼 추천"""
        schema = self.get_schema()
        
        if table_name not in schema.tables:
            return []
        
        table_info = schema.tables[table_name]
        suggestions = []
        
        for col_name, col_info in table_info.columns.items():
            score = 0.0
            
            # 컬럼 이름 매칭
            for keyword in keywords:
                if keyword.lower() in col_name.lower():
                    score += 1.0
                
                # 컬럼 설명 매칭
                if col_info.description and keyword.lower() in col_info.description.lower():
                    score += 0.8
                
                # 의미 매핑 매칭
                if keyword in self.column_semantic_mapping:
                    mapped_name = self.column_semantic_mapping[keyword]
                    if mapped_name.lower() in col_name.lower():
                        score += 0.9
            
            if score > 0:
                suggestions.append((col_name, score))
        
        # 점수 순으로 정렬
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions
    
    def refresh_schema(self):
        """스키마 강제 갱신"""
        logger.info("스키마 강제 갱신 시작")
        self.get_schema(force_refresh=True)
        logger.info("스키마 강제 갱신 완료")
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cached_schema = None
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("스키마 캐시 초기화 완료")

if __name__ == "__main__":
    # 테스트 코드
    manager = DynamicSchemaManager()
    
    # 스키마 분석
    schema = manager.get_schema()
    
    print(f"데이터베이스: {schema.database_name}")
    print(f"테이블 수: {len(schema.tables)}")
    
    for table_name, table_info in schema.tables.items():
        print(f"\n테이블: {table_name} ({table_info.row_count}행)")
        print(f"  설명: {table_info.description}")
        print(f"  컬럼 수: {len(table_info.columns)}")
        print(f"  기본키: {table_info.primary_keys}")
        print(f"  외래키: {len(table_info.foreign_keys)}개")
        print(f"  관련 테이블: {schema.relationships.get(table_name, [])}")
    
    # 테이블 추천 테스트
    print(f"\n'교차로' 키워드 테이블 추천:")
    suggestions = manager.get_table_suggestions(['교차로'])
    for table, score in suggestions:
        print(f"  {table}: {score:.2f}")
    
    # 컬럼 추천 테스트
    print(f"\n'traffic_intersection' 테이블의 '이름' 키워드 컬럼 추천:")
    suggestions = manager.get_column_suggestions('traffic_intersection', ['이름'])
    for column, score in suggestions:
        print(f"  {column}: {score:.2f}")
