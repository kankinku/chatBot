"""
싱글톤 관리자 - 모델과 리소스의 중앙 집중식 관리

디자인 패턴: Singleton + Factory + Resource Pool
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ResourceType(Enum):
    """리소스 타입"""
    EMBEDDING_MODEL = "embedding_model"
    LLM_MODEL = "llm_model" 
    TOKENIZER = "tokenizer"
    CACHE = "cache"
    CONFIG = "config"

@dataclass
class ResourceInfo:
    """리소스 정보"""
    resource_type: ResourceType
    name: str
    instance: Any
    memory_usage_mb: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0

class SingletonManager:
    """
    싱글톤 관리자 - 모든 리소스의 중앙 집중식 관리
    
    특징:
    - 스레드 안전한 싱글톤 패턴
    - 리소스 타입별 팩토리 패턴
    - 메모리 사용량 추적
    - 지연 로딩 지원
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._resources: Dict[str, ResourceInfo] = {}
        self._factories: Dict[ResourceType, Callable] = {}
        self._lock = threading.Lock()
        self._initialized = True
        
        logger.info("싱글톤 관리자 초기화 완료")
    
    def register_factory(self, resource_type: ResourceType, factory_func: Callable):
        """리소스 타입별 팩토리 함수 등록"""
        self._factories[resource_type] = factory_func
        logger.info(f"팩토리 등록: {resource_type.value}")
    
    def get_resource(self, resource_type: ResourceType, name: str, **kwargs) -> Any:
        """리소스 획득 (지연 로딩)"""
        resource_key = f"{resource_type.value}:{name}"
        
        with self._lock:
            # 기존 리소스 확인
            if resource_key in self._resources:
                resource_info = self._resources[resource_key]
                resource_info.access_count += 1
                resource_info.last_accessed = time.time()
                return resource_info.instance
            
            # 새 리소스 생성
            if resource_type not in self._factories:
                raise ValueError(f"등록되지 않은 리소스 타입: {resource_type.value}")
            
            factory_func = self._factories[resource_type]
            instance = factory_func(name, **kwargs)
            
            # 리소스 정보 저장
            resource_info = ResourceInfo(
                resource_type=resource_type,
                name=name,
                instance=instance,
                last_accessed=time.time(),
                access_count=1
            )
            
            self._resources[resource_key] = resource_info
            logger.info(f"리소스 생성: {resource_key}")
            
            return instance
    
    def release_resource(self, resource_type: ResourceType, name: str):
        """리소스 해제"""
        resource_key = f"{resource_type.value}:{name}"
        
        with self._lock:
            if resource_key in self._resources:
                del self._resources[resource_key]
                logger.info(f"리소스 해제: {resource_key}")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """리소스 사용 통계"""
        with self._lock:
            stats = {
                'total_resources': len(self._resources),
                'by_type': {},
                'memory_usage_mb': 0.0
            }
            
            for resource_info in self._resources.values():
                resource_type = resource_info.resource_type.value
                if resource_type not in stats['by_type']:
                    stats['by_type'][resource_type] = {
                        'count': 0,
                        'total_access': 0
                    }
                
                stats['by_type'][resource_type]['count'] += 1
                stats['by_type'][resource_type]['total_access'] += resource_info.access_count
                stats['memory_usage_mb'] += resource_info.memory_usage_mb
            
            return stats
    
    def cleanup_unused_resources(self, max_age_seconds: float = 3600):
        """사용하지 않는 리소스 정리"""
        import time
        current_time = time.time()
        
        with self._lock:
            to_remove = []
            for resource_key, resource_info in self._resources.items():
                if current_time - resource_info.last_accessed > max_age_seconds:
                    to_remove.append(resource_key)
            
            for resource_key in to_remove:
                del self._resources[resource_key]
                logger.info(f"미사용 리소스 정리: {resource_key}")
            
            return len(to_remove)

# 전역 싱글톤 인스턴스
singleton_manager = SingletonManager()

# 편의 함수들
def get_embedding_model(model_name: str = "jhgan/ko-sroberta-multitask", **kwargs):
    """임베딩 모델 획득"""
    return singleton_manager.get_resource(ResourceType.EMBEDDING_MODEL, model_name, **kwargs)

def get_llm_model(model_name: str, **kwargs):
    """LLM 모델 획득"""
    return singleton_manager.get_resource(ResourceType.LLM_MODEL, model_name, **kwargs)

def get_cache(cache_name: str = "default", **kwargs):
    """캐시 인스턴스 획득"""
    return singleton_manager.get_resource(ResourceType.CACHE, cache_name, **kwargs)

def get_config(config_name: str, **kwargs):
    """설정 인스턴스 획득"""
    return singleton_manager.get_resource(ResourceType.CONFIG, config_name, **kwargs)

# 팩토리 함수들
def _create_embedding_model(model_name: str, **kwargs):
    """임베딩 모델 팩토리"""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, cache_folder="./models", show_progress_bar=False, **kwargs)
    except Exception as e:
        logger.error(f"임베딩 모델 생성 실패: {model_name}, {e}")
        raise

def _create_llm_model(model_name: str, **kwargs):
    """LLM 모델 팩토리"""
    from core.llm.answer_generator import ModelManager
    model_manager = ModelManager()
    return model_manager.get_model(model_name, **kwargs)

def _create_cache(cache_name: str, **kwargs):
    """캐시 팩토리"""
    from core.cache.fast_cache import FastCache
    return FastCache(**kwargs)

def _create_config(config_name: str, **kwargs):
    """설정 팩토리"""
    from core.config.pipeline_config_manager import PipelineConfigManager
    return PipelineConfigManager(**kwargs)

# 팩토리 함수 등록
singleton_manager.register_factory(ResourceType.EMBEDDING_MODEL, _create_embedding_model)
singleton_manager.register_factory(ResourceType.LLM_MODEL, _create_llm_model)
singleton_manager.register_factory(ResourceType.CACHE, _create_cache)
singleton_manager.register_factory(ResourceType.CONFIG, _create_config)
