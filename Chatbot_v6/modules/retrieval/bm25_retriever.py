"""
BM25 Retriever - BM25 검색기

BM25 알고리즘을 사용한 키워드 기반 검색 (단일 책임).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple

from modules.core.types import Chunk
from modules.core.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """
    BM25 검색기
    
    단일 책임: BM25 알고리즘을 사용한 키워드 검색만 수행
    """
    
    def __init__(
        self,
        chunks: List[Chunk],
        n_min: int = 3,
        n_max: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Args:
            chunks: 청크 리스트
            n_min: 최소 n-gram
            n_max: 최대 n-gram
            k1: BM25 k1 파라미터
            b: BM25 b 파라미터
        """
        self.chunks = chunks
        self.n_min = n_min
        self.n_max = n_max
        self.k1 = k1
        self.b = b
        
        logger.info("BM25Retriever initializing",
                   num_chunks=len(chunks),
                   k1=k1,
                   b=b)
        
        # 인덱스 구축
        self._build_index()
        
        logger.info("BM25Retriever initialized",
                   vocab_size=len(self.idf))
    
    def _char_ngrams(self, text: str) -> List[str]:
        """문자 n-gram 생성"""
        text = re.sub(r"\s+", " ", text.lower())
        ngrams: List[str] = []
        
        for n in range(self.n_min, self.n_max + 1):
            for i in range(max(0, len(text) - n + 1)):
                ngrams.append(text[i:i + n])
        
        return ngrams
    
    def _build_index(self) -> None:
        """BM25 인덱스 구축"""
        # 문서별 n-gram 추출
        self.doc_terms: List[List[str]] = [
            self._char_ngrams(chunk.text) 
            for chunk in self.chunks
        ]
        
        # 문서 길이
        self.doc_len = [len(terms) for terms in self.doc_terms]
        
        # 평균 문서 길이
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        
        # Document Frequency 계산
        self.df: dict[str, int] = defaultdict(int)
        for terms in self.doc_terms:
            for term in set(terms):
                self.df[term] += 1
        
        # IDF 계산
        self.N = len(self.chunks)
        self.idf: dict[str, float] = {}
        
        for term, df in self.df.items():
            # BM25 IDF with log + correction
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
        
        logger.debug("BM25 index built",
                    vocab_size=len(self.idf),
                    avg_doc_len=self.avgdl)
    
    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """
        BM25 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 수
            
        Returns:
            [(청크 인덱스, 점수), ...] 리스트
        """
        # 쿼리 n-gram 추출
        q_terms = self._char_ngrams(query)
        q_tf = Counter(q_terms)
        
        scores = [0.0] * self.N
        
        # 정확한 키워드 매칭 보너스
        exact_match_boost = 2.0
        q_lower = query.lower()
        
        # BM25 점수 계산
        for i, terms in enumerate(self.doc_terms):
            tf = Counter(terms)
            dl = self.doc_len[i]
            score = 0.0
            
            # 각 쿼리 term에 대해 BM25 점수 계산
            for term, qf in q_tf.items():
                if term not in tf:
                    continue
                
                idf = self.idf.get(term, 0.0)
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
                score += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
            
            # 정확한 키워드 매칭 보너스
            doc_text = self.chunks[i].text.lower()
            if q_lower in doc_text:
                score *= exact_match_boost
            
            scores[i] = score
        
        # 점수 순으로 정렬
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        result = ranked[:top_k]
        
        logger.debug(f"BM25 search completed",
                    query_length=len(query),
                    results=len(result),
                    top_score=result[0][1] if result else 0.0)
        
        return result

