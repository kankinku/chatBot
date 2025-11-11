"""
DatabaseSchema 어댑터 패턴

기존 SQL Generator의 DatabaseSchema와 새로운 Dynamic Schema Manager의 
DatabaseSchema 간 호환성을 제공하는 어댑터
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .sql_generator import DatabaseSchema as LegacyDatabaseSchema
from .dynamic_schema_manager import DatabaseSchema as NewDatabaseSchema, TableInfo, ColumnInfo

@dataclass
class SchemaAdapter:
    """스키마 어댑터 클래스"""
    
    @staticmethod
    def legacy_to_new(legacy_schema: LegacyDatabaseSchema, database_name: str = "default") -> NewDatabaseSchema:
        """기존 스키마를 새로운 스키마로 변환"""
        
        # 컬럼 정보 변환
        columns = {}
        for col in legacy_schema.columns:
            col_info = ColumnInfo(
                name=col["name"],
                type=col["type"],
                is_nullable=True,  # 기본값
                is_primary_key=col["name"] == legacy_schema.primary_key,
                is_foreign_key=False,  # 기본값
                description=col.get("description", "")
            )
            columns[col["name"]] = col_info
        
        # 외래키 정보 처리
        foreign_keys = []
        if legacy_schema.foreign_keys:
            for fk in legacy_schema.foreign_keys:
                # 외래키 정보가 있으면 해당 컬럼을 외래키로 표시
                if fk["column"] in columns:
                    columns[fk["column"]].is_foreign_key = True
        
        # 테이블 정보 생성
        table_info = TableInfo(
            name=legacy_schema.table_name,
            columns=columns,
            primary_keys=[legacy_schema.primary_key] if legacy_schema.primary_key else [],
            foreign_keys=foreign_keys,
            indexes=[],
            row_count=len(legacy_schema.sample_data) if legacy_schema.sample_data else 0,
            description=f"Converted from legacy schema for {legacy_schema.table_name}"
        )
        
        # 새로운 스키마 생성
        new_schema = NewDatabaseSchema(
            database_name=database_name,
            tables={legacy_schema.table_name: table_info},
            relationships={}
        )
        
        return new_schema
    
    @staticmethod
    def new_to_legacy(new_schema: NewDatabaseSchema, table_name: str) -> LegacyDatabaseSchema:
        """새로운 스키마를 기존 스키마로 변환"""
        
        if table_name not in new_schema.tables:
            raise ValueError(f"Table {table_name} not found in schema")
        
        table_info = new_schema.tables[table_name]
        
        # 컬럼 정보 변환
        columns = []
        for col_name, col_info in table_info.columns.items():
            columns.append({
                "name": col_info.name,
                "type": col_info.type,
                "description": col_info.description or ""
            })
        
        # 기본키 찾기
        primary_key = table_info.primary_keys[0] if table_info.primary_keys else None
        
        # 외래키 정보 변환
        foreign_keys = []
        for fk in table_info.foreign_keys:
            foreign_keys.append({
                "column": fk.column_name,
                "references": f"{fk.referenced_table}.{fk.referenced_column}"
            })
        
        # 샘플 데이터 (빈 리스트로 초기화)
        sample_data = []
        
        legacy_schema = LegacyDatabaseSchema(
            table_name=table_name,
            columns=columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys if foreign_keys else None,
            sample_data=sample_data if sample_data else None
        )
        
        return legacy_schema
    
    @staticmethod
    def create_test_schema(table_name: str = "test_table") -> LegacyDatabaseSchema:
        """테스트용 기존 스키마 생성"""
        return LegacyDatabaseSchema(
            table_name=table_name,
            columns=[
                {"name": "id", "type": "INT", "description": "고유 식별자"},
                {"name": "name", "type": "VARCHAR(100)", "description": "이름"},
                {"name": "created_at", "type": "DATETIME", "description": "생성 시간"}
            ],
            primary_key="id",
            foreign_keys=None,
            sample_data=None
        )
    
    @staticmethod
    def create_water_treatment_schema() -> LegacyDatabaseSchema:
        """정수장 테스트용 스키마 생성"""
        return LegacyDatabaseSchema(
            table_name="water_treatment",
            columns=[
                {"name": "id", "type": "INT", "description": "고유 식별자"},
                {"name": "facility_name", "type": "VARCHAR(100)", "description": "시설명"},
                {"name": "treatment_date", "type": "DATE", "description": "처리 날짜"},
                {"name": "water_quality", "type": "FLOAT", "description": "수질 지수"},
                {"name": "treatment_volume", "type": "FLOAT", "description": "처리량"}
            ],
            primary_key="id",
            foreign_keys=None,
            sample_data=None
        )

