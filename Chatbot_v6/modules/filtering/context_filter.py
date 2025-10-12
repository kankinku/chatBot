"""
Context Filter - 컨텍스트 필터

컨텍스트 품질을 검증하고 필터링합니다 (단일 책임).
"""

from __future__ import annotations

import re
import statistics
from typing import List, Dict, Tuple

from config.pipeline_config import PipelineConfig
from modules.core.types import RetrievedSpan
from modules.core.logger import get_logger

logger = get_logger(__name__)


class ContextFilter:
    """
    컨텍스트 필터
    
    단일 책임: 컨텍스트 품질 검증 및 필터링만 수행
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config
        logger.debug("ContextFilter initialized")
    
    def filter_and_calibrate(
        self,
        spans: List[RetrievedSpan],
        question: str,
        threshold_override: float | None = None,
    ) -> Tuple[List[RetrievedSpan], Dict[str, float]]:
        """
        필터링 및 캘리브레이션
        
        Args:
            spans: 검색된 span 리스트
            question: 질문
            threshold_override: 임계값 오버라이드
            
        Returns:
            (필터링된 spans, 메트릭)
        """
        if not spans:
            return [], {"filter_in": 0, "filter_out": 0}
        
        logger.debug(f"Filtering {len(spans)} spans")
        
        # 1. 오버랩 및 키워드 필터
        spans, pre_stats = self._pre_filter(question, spans)
        
        # 2. 캘리브레이션
        spans = self._calibrate_scores(spans)
        
        # 3. 임계값 필터
        threshold = (
            threshold_override 
            if threshold_override is not None 
            else self.config.thresholds.confidence_threshold
        )
        
        filtered = [
            s for s in spans 
            if (s.calibrated_conf or 0.0) >= threshold
        ]
        
        # 4. 다양성 필터 (같은 위치 중복 제거)
        filtered = self._diversify(filtered)
        
        stats = {
            **pre_stats,
            "filter_in": len(spans),
            "filter_out": len(spans) - len(filtered),
            "filter_pass_rate": len(filtered) / len(spans) if spans else 0.0,
        }
        
        logger.debug(f"Filtered to {len(filtered)} spans", stats=stats)
        
        return filtered, stats
    
    def _pre_filter(
        self,
        question: str,
        spans: List[RetrievedSpan],
    ) -> Tuple[List[RetrievedSpan], Dict[str, float]]:
        """사전 필터링 (오버랩 + 키워드)"""
        # 키 토큰 추출
        tokens = re.findall(r"[0-9A-Za-z가-힣\.\-/%]+", question.lower())
        key_tokens = [t for t in tokens if len(t) >= 2]
        
        require_keyword = (
            len(key_tokens) > 0 and 
            self.config.thresholds.keyword_filter_min >= 1
        )
        
        kept = []
        removed = 0
        
        for span in spans:
            ctx_lower = span.chunk.text.lower()
            
            # 오버랩 계산
            overlap = self._calculate_overlap(question, span.chunk.text)
            
            # 키워드 히트
            keyword_hit = any(token in ctx_lower for token in key_tokens) if key_tokens else False
            
            # 보조 점수 추가
            span.aux_scores["overlap"] = overlap
            span.aux_scores["keyword"] = 1.0 if keyword_hit else 0.0
            
            # 필터링
            if overlap < self.config.thresholds.context_min_overlap:
                removed += 1
                continue
            
            if require_keyword and not keyword_hit:
                removed += 1
                continue
            
            kept.append(span)
        
        stats = {
            "pre_filter_in": len(spans),
            "pre_filter_out": removed,
        }
        
        return kept, stats
    
    def _calculate_overlap(self, query: str, context: str) -> float:
        """오버랩 비율 계산 (char n-gram)"""
        q = set(self._char_ngrams(query.lower(), 3, 5))
        c = set(self._char_ngrams(context.lower(), 3, 5))
        
        if not q or not c:
            return 0.0
        
        return len(q & c) / len(q)
    
    def _char_ngrams(self, text: str, n_min: int, n_max: int) -> List[str]:
        """문자 n-gram 생성"""
        text = re.sub(r"\s+", " ", text)
        ngrams = []
        
        for n in range(n_min, n_max + 1):
            for i in range(max(0, len(text) - n + 1)):
                ngrams.append(text[i:i + n])
        
        return ngrams
    
    def _calibrate_scores(self, spans: List[RetrievedSpan]) -> List[RetrievedSpan]:
        """점수 캘리브레이션"""
        if not spans:
            return spans
        
        # 소스별 점수 정규화
        all_scores = []
        for span in spans:
            # aux_scores에서 숫자 값만 추출
            numeric_scores = [
                v for v in span.aux_scores.values()
                if isinstance(v, (int, float))
            ]
            if numeric_scores:
                all_scores.append(sum(numeric_scores) / len(numeric_scores))
            else:
                all_scores.append(span.score)
        
        # 통계값 계산
        if len(all_scores) > 1:
            mu = statistics.mean(all_scores)
            sigma = statistics.pstdev(all_scores)
        else:
            mu = all_scores[0] if all_scores else 0.5
            sigma = 0.0
        
        # z-score 정규화
        for span, raw_score in zip(spans, all_scores):
            if sigma > 1e-9:
                z = (raw_score - mu) / sigma
                z = max(-3.0, min(3.0, z))
                calibrated = (z + 3.0) / 6.0  # [0, 1] 범위로
            else:
                calibrated = 0.5
            
            span.calibrated_conf = calibrated
        
        return spans
    
    def _diversify(self, spans: List[RetrievedSpan]) -> List[RetrievedSpan]:
        """다양성 확보 (같은 위치 중복 제거)"""
        seen = set()
        diversified = []
        
        for span in spans:
            key = (span.chunk.filename, span.chunk.page, span.chunk.start_offset)
            
            if key in seen:
                continue
            
            seen.add(key)
            diversified.append(span)
        
        return diversified

