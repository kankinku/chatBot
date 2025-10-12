"""
Reranker - 리랭커

검색 결과를 재순위화합니다 (단일 책임).
"""

from __future__ import annotations

import re
import time
from typing import List, Tuple, Dict

from config.pipeline_config import PipelineConfig
from modules.core.types import RetrievedSpan
from modules.core.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    리랭커
    
    단일 책임: 검색 결과 재순위화만 수행
    
    Note: Cross-encoder는 무거우므로 간단한 휴리스틱 기반 리랭킹 사용
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config
        self.use_cross_encoder = config.flags.use_cross_reranker
        
        logger.info("Reranker initialized",
                   use_cross_encoder=self.use_cross_encoder)
    
    def rerank(
        self,
        question: str,
        spans: List[RetrievedSpan],
    ) -> Tuple[List[RetrievedSpan], int]:
        """
        리랭킹
        
        Args:
            question: 질문
            spans: span 리스트
            
        Returns:
            (리랭킹된 spans, 처리 시간 ms)
        """
        if not spans:
            return [], 0
        
        t0 = time.time()
        
        logger.debug(f"Reranking {len(spans)} spans")
        
        # 간단한 휴리스틱 리랭킹
        reranked = self._heuristic_rerank(question, spans)
        
        # 임계값 필터
        filtered = self._apply_threshold(reranked)
        
        elapsed_ms = int((time.time() - t0) * 1000)
        
        logger.debug(f"Reranked to {len(filtered)} spans", time_ms=elapsed_ms)
        
        return filtered, elapsed_ms
    
    def _heuristic_rerank(
        self,
        question: str,
        spans: List[RetrievedSpan],
    ) -> List[RetrievedSpan]:
        """휴리스틱 기반 리랭킹"""
        # 키워드 추출
        q_tokens = self._extract_tokens(question)
        
        # 각 span에 대해 리랭킹 점수 계산
        for span in spans:
            ctx_text = span.chunk.text.lower()
            q_lower = question.lower()
            
            score = 0.0
            
            # 1. 정확한 매칭 (가장 높은 가중치)
            if q_lower in ctx_text:
                score += 1.0
            
            # 2. 키워드 매칭
            keyword_hits = sum(1 for token in q_tokens if token in ctx_text)
            score += 0.5 * (keyword_hits / max(1, len(q_tokens)))
            
            # 3. 기존 점수 반영
            score += 0.3 * span.score
            
            # 4. 오버랩 점수 반영
            if "overlap" in span.aux_scores:
                score += 0.2 * span.aux_scores["overlap"]
            
            span.aux_scores["rerank"] = score
        
        # 리랭킹 점수 기준 정렬
        reranked = sorted(
            spans,
            key=lambda s: s.aux_scores.get("rerank", 0.0),
            reverse=True
        )
        
        # rank 재할당
        for i, span in enumerate(reranked, start=1):
            span.rank = i
        
        return reranked
    
    def _apply_threshold(self, spans: List[RetrievedSpan]) -> List[RetrievedSpan]:
        """리랭킹 임계값 적용"""
        if not spans:
            return []
        
        # 점수 정규화
        scores = [s.aux_scores.get("rerank", 0.0) for s in spans]
        min_score = min(scores)
        max_score = max(scores)
        
        if abs(max_score - min_score) < 1e-12:
            # 모든 점수가 같으면 정규화 불가
            for span in spans:
                span.aux_scores["rerank_norm"] = 0.5
            return spans
        
        # Min-max 정규화
        for span in spans:
            raw = span.aux_scores.get("rerank", 0.0)
            normalized = (raw - min_score) / (max_score - min_score)
            span.aux_scores["rerank_norm"] = normalized
        
        # 임계값 필터
        threshold = self.config.thresholds.rerank_threshold
        filtered = [
            s for s in spans
            if s.aux_scores.get("rerank_norm", 0.0) >= threshold
        ]
        
        return filtered if filtered else spans[:1]  # 최소 1개는 유지
    
    def _extract_tokens(self, text: str) -> List[str]:
        """토큰 추출"""
        tokens = re.findall(r"[0-9A-Za-z가-힣\.\-/%]+", text.lower())
        return [t for t in tokens if len(t) >= 2]

