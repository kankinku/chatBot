"""
시스템 최적화 설정

챗봇 시스템의 성능 최적화를 위한 설정값들을 관리합니다.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json

@dataclass
class ModelConfig:
    """모델 관련 최적화 설정"""
    # 지연 로딩 설정
    lazy_loading: bool = True
    
    # 메모리 최적화 설정
    use_float16: bool = True
    low_cpu_mem_usage: bool = True
    device_map: str = "auto"
    
    # 캐시 설정
    cache_dir: str = "./models"
    max_cache_size_gb: float = 10.0
    
    # 모델별 설정
    model_configs: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.model_configs is None:
            self.model_configs = {
                "beomi/KoAlpaca-Polyglot-5.8B": {
                    "torch_dtype": "float16",
                    "max_memory_gb": 1.5,
                    "priority": "high"
                },
                "paust/pko-t5-small": {
                    "torch_dtype": "float16",
                    "max_memory_gb": 1.0,
                    "priority": "high"
                },
                "jhgan/ko-sroberta-multitask": {
                    "cache_folder": "./models",
                    "max_memory_gb": 0.5,
                    "priority": "high"
                },
                "defog/sqlcoder-7b-2": {
                    "quantization": "4bit",
                    "max_memory_gb": 4.0,
                    "priority": "medium"
                }
            }

@dataclass
class CacheConfig:
    """캐시 관련 최적화 설정"""
    # 캐시 크기 설정
    question_cache_size: int = 500
    sql_cache_size: int = 200
    vector_cache_size: int = 1000
    instant_cache_size: int = 100
    
    # TTL 설정 (초)
    question_cache_ttl: int = 1800  # 30분
    sql_cache_ttl: int = 3600       # 1시간
    vector_cache_ttl: int = 7200    # 2시간
    instant_cache_ttl: int = 86400  # 24시간
    
    # 정리 설정
    cleanup_interval: int = 300     # 5분
    auto_cleanup: bool = True
    
    # 캐시 전략
    eviction_policy: str = "lru_frequency"  # lru, lfu, lru_frequency
    preload_common_queries: bool = True

@dataclass
class DatabaseConfig:
    """데이터베이스 관련 최적화 설정"""
    # 연결 풀 설정
    connection_pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # 연결 타임아웃 설정
    connect_timeout: int = 10
    read_timeout: int = 30
    write_timeout: int = 30
    
    # 쿼리 최적화
    query_cache_enabled: bool = True
    prepared_statements: bool = True
    batch_size: int = 1000
    
    # 추가 설정
    autocommit: bool = True
    charset: str = "utf8mb4"
    max_allowed_packet: int = 16 * 1024 * 1024

@dataclass
class PerformanceConfig:
    """성능 관련 최적화 설정"""
    # 메모리 관리
    memory_warning_threshold: float = 80.0  # %
    memory_critical_threshold: float = 90.0  # %
    auto_memory_optimization: bool = True
    aggressive_gc: bool = False
    
    # 비동기 처리
    async_processing: bool = True
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    # 로깅 최적화
    async_logging: bool = True
    log_buffer_size: int = 1000
    log_rotation_size: int = 10 * 1024 * 1024  # 10MB
    
    # 모니터링
    performance_monitoring: bool = True
    memory_monitoring: bool = True
    monitoring_interval: int = 60  # 초

@dataclass
class OptimizationConfig:
    """전체 최적화 설정"""
    model: ModelConfig = None
    cache: CacheConfig = None
    database: DatabaseConfig = None
    performance: PerformanceConfig = None
    
    # 환경별 설정
    environment: str = "production"  # development, testing, production
    debug_mode: bool = False
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        
        # 환경별 설정 조정
        self._adjust_for_environment()
    
    def _adjust_for_environment(self):
        """환경에 따른 설정 조정"""
        if self.environment == "development":
            self.debug_mode = True
            self.performance.performance_monitoring = True
            self.performance.memory_monitoring = True
            self.cache.cleanup_interval = 60  # 더 자주 정리
            
        elif self.environment == "testing":
            self.cache.question_cache_size = 100
            self.cache.sql_cache_size = 50
            self.performance.max_concurrent_requests = 5
            
        elif self.environment == "production":
            self.debug_mode = False
            self.performance.aggressive_gc = True
            self.performance.auto_memory_optimization = True
            self.cache.preload_common_queries = True
    
    @classmethod
    def from_env(cls) -> 'OptimizationConfig':
        """환경 변수에서 설정 로드"""
        config = cls()
        
        # 환경 변수 오버라이드
        config.environment = os.getenv("OPTIMIZATION_ENV", "production")
        config.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        
        # 모델 설정
        config.model.lazy_loading = os.getenv("MODEL_LAZY_LOADING", "true").lower() == "true"
        config.model.use_float16 = os.getenv("MODEL_USE_FLOAT16", "true").lower() == "true"
        config.model.cache_dir = os.getenv("MODEL_CACHE_DIR", "./models")
        
        # 캐시 설정
        if os.getenv("CACHE_QUESTION_SIZE"):
            config.cache.question_cache_size = int(os.getenv("CACHE_QUESTION_SIZE"))
        if os.getenv("CACHE_SQL_SIZE"):
            config.cache.sql_cache_size = int(os.getenv("CACHE_SQL_SIZE"))
        
        # 데이터베이스 설정
        if os.getenv("DB_POOL_SIZE"):
            config.database.connection_pool_size = int(os.getenv("DB_POOL_SIZE"))
        if os.getenv("DB_POOL_TIMEOUT"):
            config.database.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT"))
        
        # 성능 설정
        if os.getenv("MEMORY_WARNING_THRESHOLD"):
            config.performance.memory_warning_threshold = float(os.getenv("MEMORY_WARNING_THRESHOLD"))
        if os.getenv("MAX_CONCURRENT_REQUESTS"):
            config.performance.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS"))
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "model": {
                "lazy_loading": self.model.lazy_loading,
                "use_float16": self.model.use_float16,
                "low_cpu_mem_usage": self.model.low_cpu_mem_usage,
                "device_map": self.model.device_map,
                "cache_dir": self.model.cache_dir,
                "max_cache_size_gb": self.model.max_cache_size_gb,
                "model_configs": self.model.model_configs
            },
            "cache": {
                "question_cache_size": self.cache.question_cache_size,
                "sql_cache_size": self.cache.sql_cache_size,
                "vector_cache_size": self.cache.vector_cache_size,
                "instant_cache_size": self.cache.instant_cache_size,
                "question_cache_ttl": self.cache.question_cache_ttl,
                "sql_cache_ttl": self.cache.sql_cache_ttl,
                "vector_cache_ttl": self.cache.vector_cache_ttl,
                "instant_cache_ttl": self.cache.instant_cache_ttl,
                "cleanup_interval": self.cache.cleanup_interval,
                "auto_cleanup": self.cache.auto_cleanup,
                "eviction_policy": self.cache.eviction_policy,
                "preload_common_queries": self.cache.preload_common_queries
            },
            "database": {
                "connection_pool_size": self.database.connection_pool_size,
                "max_overflow": self.database.max_overflow,
                "pool_timeout": self.database.pool_timeout,
                "pool_recycle": self.database.pool_recycle,
                "connect_timeout": self.database.connect_timeout,
                "read_timeout": self.database.read_timeout,
                "write_timeout": self.database.write_timeout,
                "query_cache_enabled": self.database.query_cache_enabled,
                "prepared_statements": self.database.prepared_statements,
                "batch_size": self.database.batch_size,
                "autocommit": self.database.autocommit,
                "charset": self.database.charset,
                "max_allowed_packet": self.database.max_allowed_packet
            },
            "performance": {
                "memory_warning_threshold": self.performance.memory_warning_threshold,
                "memory_critical_threshold": self.performance.memory_critical_threshold,
                "auto_memory_optimization": self.performance.auto_memory_optimization,
                "aggressive_gc": self.performance.aggressive_gc,
                "async_processing": self.performance.async_processing,
                "max_concurrent_requests": self.performance.max_concurrent_requests,
                "request_timeout": self.performance.request_timeout,
                "async_logging": self.performance.async_logging,
                "log_buffer_size": self.performance.log_buffer_size,
                "log_rotation_size": self.performance.log_rotation_size,
                "performance_monitoring": self.performance.performance_monitoring,
                "memory_monitoring": self.performance.memory_monitoring,
                "monitoring_interval": self.performance.monitoring_interval
            },
            "environment": self.environment,
            "debug_mode": self.debug_mode
        }
    
    def save_to_file(self, filepath: str):
        """설정을 파일로 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'OptimizationConfig':
        """파일에서 설정 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = cls()
        
        # 설정 적용 (간단한 버전)
        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if "cache" in data:
            for key, value in data["cache"].items():
                if hasattr(config.cache, key):
                    setattr(config.cache, key, value)
        
        if "database" in data:
            for key, value in data["database"].items():
                if hasattr(config.database, key):
                    setattr(config.database, key, value)
        
        if "performance" in data:
            for key, value in data["performance"].items():
                if hasattr(config.performance, key):
                    setattr(config.performance, key, value)
        
        config.environment = data.get("environment", "production")
        config.debug_mode = data.get("debug_mode", False)
        
        return config

# 전역 설정 인스턴스
_global_config: Optional[OptimizationConfig] = None

def get_optimization_config() -> OptimizationConfig:
    """전역 최적화 설정 반환"""
    global _global_config
    if _global_config is None:
        _global_config = OptimizationConfig.from_env()
    return _global_config

def set_optimization_config(config: OptimizationConfig):
    """전역 최적화 설정 설정"""
    global _global_config
    _global_config = config

# 편의 함수들
def get_model_config() -> ModelConfig:
    """모델 설정 반환"""
    return get_optimization_config().model

def get_cache_config() -> CacheConfig:
    """캐시 설정 반환"""
    return get_optimization_config().cache

def get_database_config() -> DatabaseConfig:
    """데이터베이스 설정 반환"""
    return get_optimization_config().database

def get_performance_config() -> PerformanceConfig:
    """성능 설정 반환"""
    return get_optimization_config().performance

if __name__ == "__main__":
    # 테스트 코드
    config = OptimizationConfig.from_env()
    
    print("현재 최적화 설정:")
    print(f"- 환경: {config.environment}")
    print(f"- 디버그 모드: {config.debug_mode}")
    print(f"- 모델 지연 로딩: {config.model.lazy_loading}")
    print(f"- 질문 캐시 크기: {config.cache.question_cache_size}")
    print(f"- DB 연결 풀 크기: {config.database.connection_pool_size}")
    print(f"- 메모리 경고 임계치: {config.performance.memory_warning_threshold}%")
    
    # 설정 파일 저장 테스트
    config.save_to_file("optimization_config.json")
    print("설정 파일 저장 완료")
    
    # 설정 파일 로드 테스트
    loaded_config = OptimizationConfig.load_from_file("optimization_config.json")
    print("설정 파일 로드 완료")
    
    print("최적화 설정 테스트 완료")
