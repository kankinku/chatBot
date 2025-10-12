"""
Deduplicator - 중복 제거기

중복 컨텍스트를 제거합니다 (단일 책임).
"""

from __future__ import annotations

import re
from typing import List, Set

from config.pipeline_config import DeduplicationConfig
from modules.core.types import RetrievedSpan
from modules.core.logger import get_logger

logger = get_logger(__name__)


class Deduplicator:
    """
    중복 제거기
    
    단일 책임: 중복 제거만 수행
    """
    
    def __init__(self, config: DeduplicationConfig):
        """
        Args:
            config: 중복 제거 설정
        """
        self.config = config
        logger.debug("Deduplicator initialized")
    
    def deduplicate(self, spans: List[RetrievedSpan]) -> List[RetrievedSpan]:
        """
        중복 제거
        
        Args:
            spans: span 리스트
            
        Returns:
            중복 제거된 spans
        """
        if not spans or len(spans) <= 1:
            return spans
        
        logger.debug(f"Deduplicating {len(spans)} spans")
        
        # Jaccard 기반 중복 제거
        deduplicated = self._jaccard_deduplicate(spans)
        
        logger.debug(f"Deduplicated to {len(deduplicated)} spans")
        
        return deduplicated
    
    def _jaccard_deduplicate(self, spans: List[RetrievedSpan]) -> List[RetrievedSpan]:
        """Jaccard 유사도 기반 중복 제거"""
        if not spans:
            return []
        
        result = [spans[0]]  # 첫 번째는 항상 포함
        
        for span in spans[1:]:
            # 이미 추가된 spans와 비교
            is_duplicate = False
            
            for existing in result:
                # 길이 체크
                if len(span.chunk.text) < self.config.min_chunk_length:
                    is_duplicate = True
                    break
                
                # Jaccard 유사도 계산
                similarity = self._jaccard_similarity(
                    span.chunk.text,
                    existing.chunk.text
                )
                
                if similarity >= self.config.jaccard_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                result.append(span)
        
        return result
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard 유사도 계산"""
        # 문자 3-gram 생성
        ngrams1 = self._char_ngrams(text1, 3)
        ngrams2 = self._char_ngrams(text2, 3)
        
        if not ngrams1 and not ngrams2:
            return 0.0
        
        set1 = set(ngrams1)
        set2 = set(ngrams2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _char_ngrams(self, text: str, n: int) -> List[str]:
        """문자 n-gram 생성"""
        text = re.sub(r"\s+", " ", text.lower())
        
        return [
            text[i:i + n]
            for i in range(max(0, len(text) - n + 1))
        ]

