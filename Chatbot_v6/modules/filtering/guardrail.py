"""
Guardrail Checker - 가드레일 체커

질문-컨텍스트 정합성을 검증합니다 (단일 책임).
"""

from __future__ import annotations

import re
from typing import List, Dict

from config.pipeline_config import PipelineConfig
from modules.core.types import RetrievedSpan
from modules.core.logger import get_logger

logger = get_logger(__name__)


class GuardrailChecker:
    """
    가드레일 체커
    
    단일 책임: 질문-컨텍스트 정합성 검증만 수행
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config
        logger.debug("GuardrailChecker initialized")
    
    def check(
        self,
        question: str,
        contexts: List[RetrievedSpan],
    ) -> Dict[str, float]:
        """
        가드레일 체크
        
        Args:
            question: 질문
            contexts: 컨텍스트 리스트
            
        Returns:
            체크 결과 메트릭
        """
        if not contexts:
            return {
                "hard_blocked": 1,
                "overlap_ratio": 0.0,
                "key_token_coverage": 0.0,
            }
        
        # 키 토큰 추출
        key_tokens = self._extract_key_tokens(question)
        
        # 전체 컨텍스트 텍스트
        all_context = " ".join([c.chunk.text for c in contexts])
        
        # 오버랩 비율
        overlap = self._calculate_overlap(question, all_context)
        
        # 키 토큰 커버리지
        coverage = self._calculate_coverage(key_tokens, all_context)
        
        # Hard block 조건
        hard_blocked = 0
        if overlap < self.config.thresholds.guard_overlap_threshold:
            hard_blocked = 1
        elif (
            len(key_tokens) >= self.config.thresholds.guard_key_tokens_min and
            coverage < 0.3
        ):
            hard_blocked = 1
        
        result = {
            "hard_blocked": hard_blocked,
            "overlap_ratio": overlap,
            "key_token_coverage": coverage,
        }
        
        logger.debug("Guardrail check", **result)
        
        return result
    
    def _extract_key_tokens(self, text: str) -> List[str]:
        """키 토큰 추출"""
        tokens = re.findall(r"[0-9A-Za-z가-힣\.\-/%]+", text.lower())
        
        # 길이 2 이상만
        key_tokens = [t for t in tokens if len(t) >= 2]
        
        # 중복 제거 (순서 유지)
        seen = set()
        unique = []
        for t in key_tokens:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        
        return unique
    
    def _calculate_overlap(self, query: str, context: str) -> float:
        """오버랩 비율 계산"""
        q_ngrams = set(self._char_ngrams(query.lower(), 3, 5))
        c_ngrams = set(self._char_ngrams(context.lower(), 3, 5))
        
        if not q_ngrams or not c_ngrams:
            return 0.0
        
        return len(q_ngrams & c_ngrams) / len(q_ngrams)
    
    def _calculate_coverage(self, key_tokens: List[str], context: str) -> float:
        """키 토큰 커버리지 계산"""
        if not key_tokens:
            return 1.0
        
        context_lower = context.lower()
        
        hits = sum(1 for token in key_tokens if token in context_lower)
        
        return hits / len(key_tokens)
    
    def _char_ngrams(self, text: str, n_min: int, n_max: int) -> List[str]:
        """문자 n-gram 생성"""
        text = re.sub(r"\s+", " ", text)
        ngrams = []
        
        for n in range(n_min, n_max + 1):
            for i in range(max(0, len(text) - n + 1)):
                ngrams.append(text[i:i + n])
        
        return ngrams

