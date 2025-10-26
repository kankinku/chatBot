"""
Hybrid Retriever

BM25 키워드 검색과 Vector 의미 검색을 가중치 합산으로 결합.
병렬 처리로 성능 최적화.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections import defaultdict
from typing import List, Tuple, Dict

from modules.core.types import Chunk, RetrievedSpan
from modules.core.logger import get_logger
from modules.embedding.base_embedder import BaseEmbedder
from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever

logger = get_logger(__name__)


class HybridRetriever:
    """BM25와 Vector 검색을 가중치 합산으로 결합"""
    
    def __init__(
        self,
        chunks: List[Chunk],
        embedder: BaseEmbedder,
        vector_weight: float = 0.58,
        bm25_weight: float = 0.42,
        index_dir: str | None = None,
    ):
        """
        Args:
            chunks: 청크 리스트
            embedder: 임베더
            vector_weight: 벡터 검색 가중치
            bm25_weight: BM25 검색 가중치
            index_dir: 벡터 인덱스 디렉토리
        """
        self.chunks = chunks
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        logger.info("HybridRetriever initializing",
                   num_chunks=len(chunks),
                   vector_weight=vector_weight,
                   bm25_weight=bm25_weight)
        
        # BM25 검색기
        self.bm25 = BM25Retriever(chunks)
        
        # Vector 검색기 (GPU 강제 사용)
        self.vector = VectorRetriever(chunks, embedder, index_dir, use_gpu=True)
        
        logger.info("HybridRetriever initialized")
    
    def search(
        self,
        query: str,
        top_k: int = 50,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> Tuple[List[RetrievedSpan], Dict[str, int]]:
        """
        하이브리드 검색 (진짜 병렬 처리)
        
        Args:
            query: 검색 쿼리
            top_k: 각 검색기에서 가져올 결과 수
            vector_weight: 벡터 가중치 (None이면 기본값 사용)
            bm25_weight: BM25 가중치 (None이면 기본값 사용)
            
        Returns:
            (검색 결과 리스트, 메트릭)
        """
        import time
        
        # 가중치 설정
        v_w = vector_weight if vector_weight is not None else self.vector_weight
        b_w = bm25_weight if bm25_weight is not None else self.bm25_weight
        
        # 🚀 진짜 병렬 처리: ThreadPoolExecutor + I/O 바운드 작업 최적화
        t0 = time.time()
        
        # Vector 검색은 CPU 집약적이므로 별도 스레드에서 실행
        # BM25 검색은 매우 빠르므로 메인 스레드에서 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Vector 검색만 별도 스레드에서 실행
            vector_future = executor.submit(self.vector.search, query, top_k)
            
            # BM25 검색은 메인 스레드에서 즉시 실행
            bm25_results = self.bm25.search(query, top_k)
            
            # Vector 검색 결과 대기
            vector_results = vector_future.result()
        
        total_time_ms = int((time.time() - t0) * 1000)
        
        # 결과 병합
        merged_scores = self._merge_results(
            vector_results,
            bm25_results,
            v_w,
            b_w,
        )
        
        # RetrievedSpan 생성
        spans = self._create_spans(merged_scores, vector_results, bm25_results)
        
        # 메트릭
        metrics = {
            "total_time_ms": total_time_ms,
            "vector_results": len(vector_results),
            "bm25_results": len(bm25_results),
            "merged_results": len(spans),
            "parallel_execution": True,
        }
        
        logger.debug(f"Parallel hybrid search completed", 
                    results=len(spans), 
                    total_time_ms=total_time_ms)
        
        return spans, metrics
    
    
    def _merge_results(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        vector_weight: float,
        bm25_weight: float,
    ) -> Dict[int, float]:
        """
        검색 결과 병합 (정규화 + 가중치 합산)
        
        Args:
            vector_results: 벡터 검색 결과
            bm25_results: BM25 검색 결과
            vector_weight: 벡터 가중치
            bm25_weight: BM25 가중치
            
        Returns:
            {청크 인덱스: 병합 점수}
        """
        # 🚀 최적화 3: 간단하고 빠른 정규화 (작은 데이터셋에 최적)
        merged: Dict[int, float] = defaultdict(float)
        
        # Vector 결과 정규화
        if vector_results:
            vector_scores = [score for _, score in vector_results]
            v_min, v_max = min(vector_scores), max(vector_scores)
            v_range = v_max - v_min if v_max > v_min else 1.0
            
            for idx, score in vector_results:
                normalized = (score - v_min) / v_range
                merged[idx] += vector_weight * normalized
        
        # BM25 결과 정규화
        if bm25_results:
            bm25_scores = [score for _, score in bm25_results]
            b_min, b_max = min(bm25_scores), max(bm25_scores)
            b_range = b_max - b_min if b_max > b_min else 1.0
            
            for idx, score in bm25_results:
                normalized = (score - b_min) / b_range
                merged[idx] += bm25_weight * normalized
        
        return merged
    
    def _create_spans(
        self,
        merged_scores: Dict[int, float],
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
    ) -> List[RetrievedSpan]:
        """RetrievedSpan 객체 생성"""
        # 보조 점수 맵
        vector_map = {idx: score for idx, score in vector_results}
        bm25_map = {idx: score for idx, score in bm25_results}
        
        # 병합 점수 순으로 정렬
        ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        
        # RetrievedSpan 생성
        spans = []
        for rank, (idx, merged_score) in enumerate(ranked, start=1):
            span = RetrievedSpan(
                chunk=self.chunks[idx],
                source="hybrid",
                score=merged_score,
                rank=rank,
                aux_scores={
                    "merged": merged_score,
                    "vector": vector_map.get(idx, 0.0),
                    "bm25": bm25_map.get(idx, 0.0),
                },
            )
            spans.append(span)
        
        return spans

