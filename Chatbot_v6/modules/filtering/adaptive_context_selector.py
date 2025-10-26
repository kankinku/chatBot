"""
Adaptive Context Selector

질문 유형과 LLM 컨텍스트 한계를 고려한 최적 컨텍스트 선택.
"""

from __future__ import annotations

from typing import List, Dict, Set
from collections import defaultdict

from modules.core.types import RetrievedSpan
from modules.core.logger import get_logger

logger = get_logger(__name__)


class AdaptiveContextSelector:
    """
    적응형 컨텍스트 선택기
    
    LLM 컨텍스트 윈도우 한계 내에서 최적의 컨텍스트 조합 선택.
    다양성, 관련성, 신뢰도를 종합 고려.
    """
    
    def __init__(
        self,
        llm_max_ctx: int = 8192,
        ctx_usage_ratio: float = 0.6,
        min_contexts: int = 2,
        max_contexts: int = 8,
    ):
        """
        Args:
            llm_max_ctx: LLM 최대 컨텍스트 토큰 수
            ctx_usage_ratio: 컨텍스트에 할당할 비율 (0.6 = 60%)
            min_contexts: 최소 컨텍스트 수
            max_contexts: 최대 컨텍스트 수
        """
        self.llm_max_ctx = llm_max_ctx
        self.ctx_usage_ratio = ctx_usage_ratio
        self.min_contexts = min_contexts
        self.max_contexts = max_contexts
        
        # 컨텍스트에 사용 가능한 최대 토큰
        self.max_ctx_tokens = int(llm_max_ctx * ctx_usage_ratio)
        
        logger.info("AdaptiveContextSelector initialized",
                   max_ctx_tokens=self.max_ctx_tokens)
    
    def select_contexts(
        self,
        question: str,
        candidates: List[RetrievedSpan],
        question_type: str = "general",
    ) -> List[RetrievedSpan]:
        """
        최적 컨텍스트 선택
        
        Args:
            question: 질문
            candidates: 후보 컨텍스트 리스트
            question_type: 질문 유형
            
        Returns:
            선택된 컨텍스트 리스트
        """
        if not candidates:
            return []
        
        logger.debug(f"Selecting contexts",
                    candidates=len(candidates),
                    question_type=question_type)
        
        # 1. 필수 컨텍스트 (높은 점수)
        essential = self._select_essential(candidates)
        
        # 2. 다양성 확보
        diverse = self._ensure_diversity(candidates, essential)
        
        # 3. 토큰 수 제한 내에서 선택
        selected = self._select_within_token_limit(
            question,
            essential + diverse,
            question_type
        )
        
        # 4. 최소/최대 제한 적용
        selected = self._apply_count_limits(selected)
        
        logger.debug(f"Contexts selected",
                    selected=len(selected),
                    essential=len(essential),
                    diverse=len(diverse))
        
        return selected
    
    def _select_essential(
        self,
        candidates: List[RetrievedSpan],
        score_threshold: float = 0.8,
    ) -> List[RetrievedSpan]:
        """필수 컨텍스트 선택 (높은 점수)"""
        essential = [
            c for c in candidates
            if c.score > score_threshold or 
               (hasattr(c, 'calibrated_conf') and 
                c.calibrated_conf and c.calibrated_conf > score_threshold)
        ]
        
        # 최소 1개는 선택
        if not essential and candidates:
            essential = [candidates[0]]
        
        return essential[:3]  # 최대 3개
    
    def _ensure_diversity(
        self,
        candidates: List[RetrievedSpan],
        already_selected: List[RetrievedSpan],
        max_per_doc: int = 2,
    ) -> List[RetrievedSpan]:
        """다양성 확보 (다른 문서/페이지에서 선택)"""
        # 이미 선택된 문서 추적
        selected_docs: Dict[str, int] = defaultdict(int)
        selected_pages: Set[tuple] = set()
        
        for span in already_selected:
            selected_docs[span.chunk.filename] += 1
            selected_pages.add((span.chunk.filename, span.chunk.page))
        
        # 다양한 소스에서 선택
        diverse = []
        for candidate in candidates:
            if candidate in already_selected:
                continue
            
            filename = candidate.chunk.filename
            page = (candidate.chunk.filename, candidate.chunk.page)
            
            # 같은 문서에서 너무 많이 선택되지 않도록
            if selected_docs[filename] >= max_per_doc:
                continue
            
            # 같은 페이지 중복 방지
            if page in selected_pages:
                continue
            
            diverse.append(candidate)
            selected_docs[filename] += 1
            selected_pages.add(page)
            
            if len(diverse) >= 3:  # 최대 3개
                break
        
        return diverse
    
    def _select_within_token_limit(
        self,
        question: str,
        candidates: List[RetrievedSpan],
        question_type: str,
    ) -> List[RetrievedSpan]:
        """토큰 수 제한 내에서 선택"""
        selected = []
        total_tokens = self._estimate_tokens(question)  # 질문 토큰
        
        # 점수 순 정렬
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.score,
            reverse=True
        )
        
        for candidate in sorted_candidates:
            # 토큰 수 추정
            ctx_tokens = self._estimate_tokens(candidate.chunk.text)
            
            # 추가 가능한지 확인
            if total_tokens + ctx_tokens <= self.max_ctx_tokens:
                selected.append(candidate)
                total_tokens += ctx_tokens
            else:
                # 더 이상 추가 불가
                if len(selected) >= self.min_contexts:
                    break
                else:
                    # 최소 개수 미달 → 텍스트 잘라서라도 추가
                    truncated_text = self._truncate_to_fit(
                        candidate.chunk.text,
                        self.max_ctx_tokens - total_tokens
                    )
                    if truncated_text:
                        # 새 Span 생성 (텍스트만 변경)
                        from modules.core.types import Chunk
                        truncated_chunk = Chunk(
                            doc_id=candidate.chunk.doc_id,
                            filename=candidate.chunk.filename,
                            page=candidate.chunk.page,
                            start_offset=candidate.chunk.start_offset,
                            length=len(truncated_text),
                            text=truncated_text,
                        )
                        
                        truncated_span = RetrievedSpan(
                            chunk=truncated_chunk,
                            source=candidate.source,
                            score=candidate.score,
                            rank=candidate.rank,
                            aux_scores=candidate.aux_scores,
                            calibrated_conf=candidate.calibrated_conf,
                        )
                        
                        selected.append(truncated_span)
                        break
        
        return selected
    
    def _apply_count_limits(
        self,
        selected: List[RetrievedSpan]
    ) -> List[RetrievedSpan]:
        """최소/최대 개수 제한 적용"""
        # 최대 개수 초과
        if len(selected) > self.max_contexts:
            selected = selected[:self.max_contexts]
        
        # 최소 개수 미달 (이미 처리됨, 재확인)
        if len(selected) < self.min_contexts and selected:
            logger.warning(f"Selected contexts below minimum: {len(selected)}")
        
        return selected
    
    def _estimate_tokens(self, text: str) -> int:
        """
        토큰 수 추정
        
        한글/영문 토큰화를 정확하게 하려면 tokenizer 필요하지만,
        간단하게 문자 수 / 4로 추정 (GPT 기준).
        """
        return len(text) // 4
    
    def _truncate_to_fit(self, text: str, max_tokens: int) -> str:
        """토큰 수에 맞게 텍스트 자르기"""
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # 문장 경계에서 자르기
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        last_break = max(last_period, last_newline)
        
        if last_break > max_chars * 0.7:  # 70% 이상이면 사용
            return truncated[:last_break + 1]
        else:
            return truncated + "..."

