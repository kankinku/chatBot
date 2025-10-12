"""
Embedder Factory - 임베더 팩토리

임베더 생성을 담당합니다.
"""

from __future__ import annotations

from typing import Optional

from .base_embedder import BaseEmbedder
from .sbert_embedder import SBERTEmbedder
from config.model_config import EmbeddingModelConfig
from modules.core.exceptions import ConfigurationError
from modules.core.logger import get_logger

logger = get_logger(__name__)


# 전역 임베더 캐시 (메모리 누수 방지)
_embedder_cache: dict[str, BaseEmbedder] = {}


def create_embedder(
    config: Optional[EmbeddingModelConfig] = None,
    use_cache: bool = True,
) -> BaseEmbedder:
    """
    임베더 생성
    
    Args:
        config: 임베딩 모델 설정
        use_cache: 캐시 사용 여부
        
    Returns:
        BaseEmbedder 인스턴스
        
    Raises:
        ConfigurationError: 지원하지 않는 모델
    """
    config = config or EmbeddingModelConfig()
    
    # 캐시 키 생성
    cache_key = f"{config.model_name}_{config.device}"
    
    # 캐시에서 확인
    if use_cache and cache_key in _embedder_cache:
        logger.debug(f"Using cached embedder: {cache_key}")
        return _embedder_cache[cache_key]
    
    # 모델 타입 결정 (현재는 SBERT만 지원)
    if "sbert" in config.model_name.lower() or "roberta" in config.model_name.lower():
        embedder = SBERTEmbedder(config)
    else:
        # 기본값: SBERT 사용
        logger.warning(f"Unknown model type: {config.model_name}, using SBERT")
        embedder = SBERTEmbedder(config)
    
    # 캐시에 저장
    if use_cache:
        _embedder_cache[cache_key] = embedder
        logger.debug(f"Cached embedder: {cache_key}")
    
    return embedder


def clear_embedder_cache() -> None:
    """임베더 캐시 정리"""
    global _embedder_cache
    
    logger.info(f"Clearing embedder cache ({len(_embedder_cache)} items)")
    
    # 각 임베더의 정리 메서드 호출
    for embedder in _embedder_cache.values():
        try:
            if hasattr(embedder, '__del__'):
                embedder.__del__()
        except Exception:
            pass
    
    _embedder_cache.clear()
    logger.info("Embedder cache cleared")

