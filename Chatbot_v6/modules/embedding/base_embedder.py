"""
Base Embedder - 임베더 기본 클래스

모든 임베더의 베이스 클래스입니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from modules.core.logger import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """
    임베더 베이스 클래스
    
    모든 임베더는 이 클래스를 상속받아 embed_texts, embed_query 메서드를 구현합니다.
    """
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        여러 텍스트를 임베딩
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            임베딩 벡터 배열 (shape: [len(texts), dim])
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        단일 쿼리를 임베딩
        
        Args:
            text: 쿼리 텍스트
            
        Returns:
            임베딩 벡터 (shape: [dim])
        """
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """임베딩 차원"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """모델 이름"""
        pass

