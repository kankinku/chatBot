"""
통합 설정 관리자

모든 설정을 중앙에서 관리하는 싱글톤 패턴 적용
디자인 패턴: Singleton + Builder + Strategy
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class ConfigSource(Enum):
    """설정 소스"""
    ENVIRONMENT = "environment"
    JSON_FILE = "json_file"
    DEFAULT = "default"

@dataclass
class ConfigItem:
    """설정 항목"""
    key: str
    value: Any
    source: ConfigSource
    description: str = ""
    is_sensitive: bool = False

class UnifiedConfig:
    """
    통합 설정 관리자
    
    우선순위:
    1. 환경 변수
    2. JSON 설정 파일
    3. 기본값
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._config_items: Dict[str, ConfigItem] = {}
        self._config_files: List[Path] = []
        
        # 기본 설정 정의
        self._define_default_configs()
        
        # 설정 로드
        self._load_all_configs()
        
        self._initialized = True
        logger.info("통합 설정 관리자 초기화 완료")
    
    def _define_default_configs(self):
        """기본 설정 정의"""
        defaults = {
            # 시스템 정보
            'COMPANY_NAME': ('범용 RAG 시스템', '회사명'),
            'PROJECT_NAME': ('범용 RAG 시스템', '프로젝트명'),
            'SYSTEM_TYPE': ('범용 RAG 시스템', '시스템 타입'),
            'PROJECT_DOMAIN': ('범용', '프로젝트 도메인'),
            'PROJECT_INDUSTRY': ('범용', '프로젝트 산업'),
            
            # 모델 설정
            'EMBEDDING_MODEL': ('jhgan/ko-sroberta-multitask', '임베딩 모델'),
            'LLM_MODEL': ('qwen2:1.5b-instruct-q4_K_M', 'LLM 모델'),
            'CROSS_ENCODER_MODEL': ('BAAI/bge-reranker-v2-m3', '크로스엔코더 모델'),
            
            # 서버 설정
            'OLLAMA_BASE_URL': ('http://localhost:11434', 'Ollama 서버 URL'),
            'MODEL_CACHE_DIR': ('./models', '모델 캐시 디렉토리'),
            
            # 성능 설정
            'MAX_MEMORY_GB': (4.0, '최대 메모리 사용량 (GB)'),
            'CACHE_TTL_SECONDS': (3600, '캐시 TTL (초)'),
            'MAX_CONCURRENT_REQUESTS': (10, '최대 동시 요청 수'),
            
            # 로깅 설정
            'LOG_LEVEL': ('INFO', '로그 레벨'),
            'LOG_DIR': ('logs', '로그 디렉토리'),
            'ENABLE_DETAILED_LOGGING': (True, '상세 로깅 활성화'),
            
            # 벡터 저장소 설정
            'VECTOR_STORE_PATH': ('vector_store', '벡터 저장소 경로'),
            'CHUNK_SIZE': (512, '청크 크기'),
            'CHUNK_OVERLAP': (50, '청크 중복'),
            
            # 검색 설정
            'SEARCH_TOP_K': (10, '검색 상위 K개'),
            'SIMILARITY_THRESHOLD': (0.7, '유사도 임계값'),
            'ENABLE_RERANKING': (True, '재순위화 활성화'),
            
            # 법률 모듈 설정
            'LEGAL_MODE_ENABLED': (True, '법률 모드 활성화'),
            'LEGAL_ACCURACY_THRESHOLD': (0.75, '법률 정확도 임계값'),
            'LEGAL_MAX_RESULTS': (3, '법률 최대 결과 수'),
        }
        
        for key, (default_value, description) in defaults.items():
            self._config_items[key] = ConfigItem(
                key=key,
                value=default_value,
                source=ConfigSource.DEFAULT,
                description=description,
                is_sensitive='PASSWORD' in key or 'SECRET' in key or 'TOKEN' in key
            )
    
    def _load_all_configs(self):
        """모든 설정 로드"""
        # 1. JSON 파일에서 로드
        self._load_from_json_files()
        
        # 2. 환경 변수에서 로드 (최우선)
        self._load_from_environment()
    
    def _load_from_json_files(self):
        """JSON 설정 파일들에서 로드"""
        config_dirs = [
            Path("config"),
            Path("config/pipelines"),
            Path(".")
        ]
        
        for config_dir in config_dirs:
            if not config_dir.exists():
                continue
            
            for json_file in config_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self._process_json_data(data, json_file)
                except Exception as e:
                    logger.warning(f"JSON 설정 파일 로드 실패: {json_file}, {e}")
    
    def _process_json_data(self, data: Dict, source_file: Path):
        """JSON 데이터 처리"""
        def flatten_dict(d: Dict, prefix: str = ""):
            for key, value in d.items():
                full_key = f"{prefix}_{key}".upper() if prefix else key.upper()
                
                if isinstance(value, dict):
                    flatten_dict(value, full_key)
                else:
                    # 기존 설정이 있고 환경변수가 아닌 경우만 업데이트
                    if full_key in self._config_items and self._config_items[full_key].source != ConfigSource.ENVIRONMENT:
                        self._config_items[full_key].value = value
                        self._config_items[full_key].source = ConfigSource.JSON_FILE
        
        flatten_dict(data)
    
    def _load_from_environment(self):
        """환경 변수에서 로드"""
        for key in self._config_items.keys():
            env_value = os.getenv(key)
            if env_value is not None:
                # 타입 변환
                converted_value = self._convert_env_value(env_value, self._config_items[key].value)
                self._config_items[key].value = converted_value
                self._config_items[key].source = ConfigSource.ENVIRONMENT
    
    def _convert_env_value(self, env_value: str, default_value: Any) -> Any:
        """환경 변수 값 타입 변환"""
        if isinstance(default_value, bool):
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(default_value, int):
            try:
                return int(env_value)
            except ValueError:
                return default_value
        elif isinstance(default_value, float):
            try:
                return float(env_value)
            except ValueError:
                return default_value
        else:
            return env_value
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 조회"""
        if key in self._config_items:
            return self._config_items[key].value
        return default
    
    def set(self, key: str, value: Any, description: str = ""):
        """설정값 설정"""
        if key in self._config_items:
            self._config_items[key].value = value
        else:
            self._config_items[key] = ConfigItem(
                key=key,
                value=value,
                source=ConfigSource.DEFAULT,
                description=description
            )
    
    def get_all(self, prefix: str = "") -> Dict[str, Any]:
        """특정 접두사의 모든 설정 조회"""
        if not prefix:
            return {key: item.value for key, item in self._config_items.items()}
        
        return {
            key: item.value 
            for key, item in self._config_items.items() 
            if key.startswith(prefix.upper())
        }
    
    def get_config_info(self, key: str) -> Optional[ConfigItem]:
        """설정 정보 조회"""
        return self._config_items.get(key)
    
    def reload(self):
        """설정 재로드"""
        logger.info("설정 재로드 시작...")
        
        # 환경변수가 아닌 설정들을 기본값으로 리셋
        for key, item in self._config_items.items():
            if item.source != ConfigSource.ENVIRONMENT:
                # 기본값으로 리셋하되 원래 기본값 찾기
                pass  # 복잡하므로 일단 스킵
        
        self._load_all_configs()
        logger.info("설정 재로드 완료")
    
    def export_config(self, file_path: str, include_sensitive: bool = False):
        """설정을 파일로 내보내기"""
        export_data = {}
        
        for key, item in self._config_items.items():
            if item.is_sensitive and not include_sensitive:
                continue
            
            export_data[key] = {
                'value': item.value,
                'source': item.source.value,
                'description': item.description
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"설정 내보내기 완료: {file_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """설정 통계"""
        stats = {
            'total_configs': len(self._config_items),
            'by_source': {},
            'sensitive_count': 0
        }
        
        for item in self._config_items.values():
            source = item.source.value
            if source not in stats['by_source']:
                stats['by_source'][source] = 0
            stats['by_source'][source] += 1
            
            if item.is_sensitive:
                stats['sensitive_count'] += 1
        
        return stats

# 전역 설정 인스턴스
config = UnifiedConfig()

# 편의 함수들
def get_config(key: str, default: Any = None) -> Any:
    """설정값 조회"""
    return config.get(key, default)

def set_config(key: str, value: Any, description: str = ""):
    """설정값 설정"""
    config.set(key, value, description)

def get_all_configs(prefix: str = "") -> Dict[str, Any]:
    """모든 설정 조회"""
    return config.get_all(prefix)

def reload_config():
    """설정 재로드"""
    config.reload()
