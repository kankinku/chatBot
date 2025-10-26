"""
SBERT Embedder

Sentence-BERT 모델 기반 텍스트 임베딩.
배치 처리, 정규화, LRU 캐싱 지원 (쿼리 1024개).
"""

from __future__ import annotations

from typing import List, Optional
from functools import lru_cache

import numpy as np

from .base_embedder import BaseEmbedder
from config.model_config import EmbeddingModelConfig
from modules.core.exceptions import EmbeddingModelLoadError, EmbeddingGenerationError, EmbeddingDimensionMismatch
from modules.core.logger import get_logger

logger = get_logger(__name__)


class SBERTEmbedder(BaseEmbedder):
    """Sentence-BERT 기반 텍스트 임베딩"""
    
    def __init__(self, config: Optional[EmbeddingModelConfig] = None):
        """
        Args:
            config: 임베딩 모델 설정
        """
        self.config = config or EmbeddingModelConfig()
        self._model = None
        self._dim = None
        self._device = None
        
        logger.info("SBERTEmbedder initializing", 
                   model=self.config.model_name,
                   device=self.config.device)
        
        self._load_model()
    
    def _load_model(self) -> None:
        """모델 로드"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Device 결정 (GPU 강제 사용)
            if self.config.device == "auto" or self.config.device == "cpu":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            self._device = device
            
            # 모델 로드
            self._model = SentenceTransformer(
                self.config.model_name,
                device=device
            )
            
            self._dim = self._model.get_sentence_embedding_dimension()
            
            logger.info("SBERT model loaded successfully",
                       model=self.config.model_name,
                       device=device,
                       dimension=self._dim)
        
        except ImportError as e:
            raise EmbeddingModelLoadError(
                self.config.model_name,
                cause=ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            ) from e
        
        except Exception as e:
            raise EmbeddingModelLoadError(
                self.config.model_name,
                cause=e
            ) from e
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        여러 텍스트를 임베딩
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            임베딩 벡터 배열 (shape: [len(texts), dim])
            
        Raises:
            EmbeddingGenerationError: 임베딩 생성 실패
        """
        if not texts:
            return np.array([]).reshape(0, self._dim)
        
        try:
            logger.debug(f"Embedding {len(texts)} texts")
            
            embeddings = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=self.config.show_progress_bar,
            )
            
            result = embeddings.astype("float32")
            
            logger.debug(f"Embeddings generated",
                        count=len(texts),
                        shape=result.shape)
            
            return result
        
        except Exception as e:
            raise EmbeddingGenerationError(
                text_sample=texts[0] if texts else "",
                cause=e
            ) from e
    
    # 🚀 최적화 7: 쿼리 임베딩 캐싱 (동일 쿼리 재사용)
    @lru_cache(maxsize=1024)
    def _cached_embed(self, text: str) -> tuple:
        """캐싱 가능한 임베딩 (튜플로 반환)"""
        embeddings = self.embed_texts([text])
        # NumPy 배열은 캐싱 불가이므로 튜플로 변환
        return tuple(embeddings[0].astype("float32").tolist())
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        단일 쿼리를 임베딩
        
        Args:
            text: 쿼리 텍스트
            
        Returns:
            임베딩 벡터 (shape: [dim])
            
        Raises:
            EmbeddingGenerationError: 임베딩 생성 실패
        """
        if not text:
            raise EmbeddingGenerationError("Empty query text")
        
        try:
            # 캐시된 임베딩 사용
            result_tuple = self._cached_embed(text)
            result = np.array(result_tuple, dtype="float32")
            
            return result
        
        except Exception as e:
            raise EmbeddingGenerationError(
                text_sample=text[:100],
                cause=e
            ) from e
    
    @property
    def dim(self) -> int:
        """임베딩 차원"""
        return int(self._dim)
    
    @property
    def model_name(self) -> str:
        """모델 이름"""
        return self.config.model_name
    
    def __del__(self):
        """메모리 정리"""
        try:
            if hasattr(self, '_model') and self._model is not None:
                # GPU 메모리 정리
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    # GPU 정리 실패는 치명적이지 않음
                    logger.debug("GPU cache cleanup failed (non-critical)", error=str(e))
                
                # 모델 참조 해제
                del self._model
                self._model = None
                
                logger.debug("SBERT model cleaned up")
        except Exception as e:
            # 모델 정리 실패는 심각한 문제가 아니므로 debug 레벨로 로깅
            logger.debug("Model cleanup failed (non-critical)", error=str(e), error_type=type(e).__name__)

