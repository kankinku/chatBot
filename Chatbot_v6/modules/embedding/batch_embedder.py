"""
Batch Embedder

대량 텍스트 임베딩을 효율적으로 처리.
배치 크기 자동 조정 및 진행률 표시.
"""

from __future__ import annotations

import math
from typing import List, Optional, Callable

import numpy as np
from tqdm import tqdm

from .base_embedder import BaseEmbedder
from modules.core.logger import get_logger

logger = get_logger(__name__)


class BatchEmbedder:
    """
    배치 임베딩 최적화
    
    대량 문서 처리 시 메모리 효율과 속도 최적화.
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        batch_size: int = 64,
        show_progress: bool = True,
    ):
        """
        Args:
            embedder: 임베더 인스턴스
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부
        """
        self.embedder = embedder
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        logger.info("BatchEmbedder initialized",
                   batch_size=batch_size)
    
    def embed_texts_batched(
        self,
        texts: List[str],
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        배치 임베딩
        
        Args:
            texts: 텍스트 리스트
            callback: 진행률 콜백 함수 callback(current, total)
            
        Returns:
            임베딩 배열
            
        Example:
            texts = ["text1", "text2", ..., "text10000"]
            
            # 일반 방법: 10,000개 한번에 → OOM 위험
            embeddings = embedder.embed_texts(texts)
            
            # 배치 방법: 64개씩 처리 → 안전
            batch_embedder = BatchEmbedder(embedder, batch_size=64)
            embeddings = batch_embedder.embed_texts_batched(texts)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedder.dim)
        
        total_texts = len(texts)
        num_batches = math.ceil(total_texts / self.batch_size)
        
        logger.info(f"Batch embedding started",
                   total_texts=total_texts,
                   batch_size=self.batch_size,
                   num_batches=num_batches)
        
        all_embeddings = []
        
        # 진행률 표시
        iterator = range(num_batches)
        if self.show_progress:
            iterator = tqdm(
                iterator,
                desc="Embedding",
                unit="batch",
                total=num_batches
            )
        
        for i in iterator:
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, total_texts)
            batch_texts = texts[start_idx:end_idx]
            
            # 배치 임베딩
            batch_embeddings = self.embedder.embed_texts(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            # 콜백 호출
            if callback:
                callback(end_idx, total_texts)
        
        # 결합
        result = np.vstack(all_embeddings)
        
        logger.info(f"Batch embedding completed",
                   total_embeddings=len(result),
                   shape=result.shape)
        
        return result
    
    def embed_with_fallback(
        self,
        texts: List[str],
        max_retries: int = 3,
    ) -> np.ndarray:
        """
        재시도 로직 포함 배치 임베딩
        
        Args:
            texts: 텍스트 리스트
            max_retries: 최대 재시도 횟수
            
        Returns:
            임베딩 배열
        """
        for attempt in range(max_retries):
            try:
                return self.embed_texts_batched(texts)
            
            except Exception as e:
                logger.warning(
                    f"Batch embedding failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    # 배치 크기 줄여서 재시도
                    old_batch_size = self.batch_size
                    self.batch_size = max(16, self.batch_size // 2)
                    logger.info(f"Reducing batch size: {old_batch_size} → {self.batch_size}")
                else:
                    raise
        
        # 도달 불가 (명시적 반환)
        raise RuntimeError("All embedding attempts failed")
    
    def estimate_time(self, num_texts: int) -> float:
        """
        예상 소요 시간 계산
        
        Args:
            num_texts: 텍스트 개수
            
        Returns:
            예상 시간 (초)
        """
        # 경험적 추정: GPU 기준 약 0.05초/배치
        num_batches = math.ceil(num_texts / self.batch_size)
        estimated_seconds = num_batches * 0.05
        
        return estimated_seconds

