"""
Answer Generator

LLM 기반 답변 생성 및 후처리.
재시도 로직, Recovery 모드, 추출적 폴백 지원.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional

from modules.core.types import RetrievedSpan
from modules.core.logger import get_logger
from config.constants import (
    DEFAULT_MAX_ANSWER_LENGTH,
    DEFAULT_MIN_ANSWER_LENGTH,
    DEFAULT_EXTRACTIVE_FALLBACK_LENGTH,
)
from .llm_client import OllamaClient
from .prompt_builder import PromptBuilder

logger = get_logger(__name__)


class AnswerGenerator:
    """LLM 기반 답변 생성 및 품질 검증"""
    
    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        max_retries: int = 3,
        retry_backoff_ms: int = 800,
    ):
        """
        Args:
            llm_client: LLM 클라이언트
            prompt_builder: 프롬프트 빌더
            max_retries: 최대 재시도 횟수
            retry_backoff_ms: 재시도 대기 시간 (밀리초)
        """
        self.llm_client = llm_client or OllamaClient()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms
        
        logger.info("AnswerGenerator initialized",
                   max_retries=max_retries)
    
    def generate(
        self,
        question: str,
        contexts: List[RetrievedSpan],
        question_type: str = "general",
        recovery_mode: bool = False,
    ) -> str:
        """
        답변 생성
        
        Args:
            question: 질문
            contexts: 컨텍스트 리스트
            question_type: 질문 유형
            recovery_mode: 복구 모드
            
        Returns:
            생성된 답변
        """
        if not contexts:
            return "문서에서 관련 정보를 찾을 수 없습니다."
        
        # 프롬프트 생성
        prompt = self.prompt_builder.build_qa_prompt(
            question,
            contexts,
            question_type,
            recovery_mode,
        )
        
        logger.debug("Generating answer",
                    question_length=len(question),
                    num_contexts=len(contexts),
                    recovery_mode=recovery_mode)
        
        # 재시도 로직
        answer = ""
        for attempt in range(self.max_retries):
            try:
                answer = self.llm_client.generate(prompt)
                
                if answer.strip():
                    break
                
                logger.warning(f"Empty response on attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Generation failed on attempt {attempt + 1}: {e}",
                           exc_info=True)
                
                if attempt < self.max_retries - 1:
                    # 대기 후 재시도
                    backoff_s = (self.retry_backoff_ms * (attempt + 1)) / 1000.0
                    time.sleep(backoff_s)
        
        # 답변 후처리
        if answer.strip():
            answer = self._post_process(answer)
            
            # 품질 검사: 답변이 불충분한 경우 recovery 모드로 재시도 (v5 로직)
            if not recovery_mode and self._is_answer_insufficient(answer):
                logger.info("Answer quality check failed, trying recovery mode")
                try:
                    # Recovery 모드로 재시도
                    recovery_answer = self.generate(
                        question,
                        contexts,
                        question_type,
                        recovery_mode=True,  # 더 엄격한 프롬프트 사용
                    )
                    # Recovery 답변이 더 나으면 사용
                    if recovery_answer and not self._is_answer_insufficient(recovery_answer):
                        answer = recovery_answer
                        logger.info("Recovery mode produced better answer")
                except Exception as e:
                    logger.warning(f"Recovery mode failed: {e}")
        else:
            # 추출적 폴백
            answer = self._extractive_fallback(contexts)
        
        # 최종 품질 검사 후 fallback
        if self._is_answer_insufficient(answer):
            logger.warning("Final answer quality check failed, using extractive fallback")
            answer = self._extractive_fallback(contexts)
        
        logger.debug("Answer generated",
                    answer_length=len(answer))
        
        return answer
    
    def _post_process(self, text: str) -> str:
        """답변 후처리"""
        if not text:
            return ""
        
        # 기본 정리
        text = text.strip()
        
        # [답변] 형식 제거
        if text.startswith("[답변]"):
            text = text[4:].strip()
        
        # 길이 제한
        if len(text) > DEFAULT_MAX_ANSWER_LENGTH:
            sentences = text.split('.')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '.') <= DEFAULT_MAX_ANSWER_LENGTH:
                    truncated += sentence + '.'
                else:
                    break
            text = truncated.rstrip('.') + '.'
        
        # 개행 정리
        text = text.replace("\\n", " ")
        text = text.replace("\n", " ")
        
        # 연속 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 출처 표시 제거
        text = re.sub(r'\(문서\s*\d+\)', '', text)
        text = re.sub(r'문서\s*\d+에서\s*확인할\s*수\s*있습니다', '', text)
        text = re.sub(r'문서\s*\d+의\s*.*?부분에서', '', text)
        
        # 이모지 및 특수 기호 제거
        text = re.sub(r'❍|●|○|◆|◇|■|□|▲|△|▼|▽', '', text)
        text = re.sub(r'[^\w\s가-힣.,!?()\-]', '', text)
        
        # 최종 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _is_answer_insufficient(self, answer: str) -> bool:
        """
        답변이 불충분한지 판단 (v5 로직)
        
        다음 경우 불충분으로 판단:
        - 너무 짧은 답변 (< 10자)
        - "문서에서 확인할 수 없습니다" 또는 유사 표현만 있는 경우
        - "..." 또는 공백만 있는 경우
        """
        if not answer:
            return True
        
        answer = answer.strip()
        
        # 너무 짧음
        if len(answer) < DEFAULT_MIN_ANSWER_LENGTH:
            return True
        
        # "확인할 수 없습니다" 류의 답변
        insufficient_patterns = [
            r"^문서에서\s*확인할\s*수\s*없습니다",
            r"^정보를?\s*찾을\s*수\s*없습니다",
            r"^알\s*수\s*없습니다",
            r"^모르겠습니다",
            r"^답변할\s*수\s*없습니다",
            r"^\.\.\.$",  # 점만
            r"^없습니다\.?$",
        ]
        
        for pattern in insufficient_patterns:
            if re.match(pattern, answer, re.IGNORECASE):
                return True
        
        return False
    
    def _extractive_fallback(self, contexts: List[RetrievedSpan]) -> str:
        """
        추출적 폴백 (컨텍스트에서 직접 추출) - v5 개선 버전
        
        가장 관련성 높은 컨텍스트에서 의미있는 문장을 추출합니다.
        """
        if not contexts:
            return "문서에서 관련 정보를 찾을 수 없습니다."
        
        # 상위 3개 컨텍스트 병합
        combined_texts = []
        for i, context in enumerate(contexts[:3]):
            text = context.chunk.text.strip()
            if text:
                combined_texts.append(text)
        
        if not combined_texts:
            return "문서에서 관련 정보를 찾을 수 없습니다."
        
        # 가장 긴/의미있는 문장 선택
        best_text = combined_texts[0]
        
        # 문장 단위로 분리
        sentences = re.split(r'[.!?]\s+', best_text)
        
        if sentences:
            # 의미있는 문장 찾기 (너무 짧지 않고, 특수문자만 있지 않은)
            meaningful_sentences = [
                s for s in sentences 
                if len(s.strip()) > 20 and re.search(r'[가-힣a-zA-Z]', s)
            ]
            
            if meaningful_sentences:
                # 첫 2-3 문장 사용
                result = '. '.join(meaningful_sentences[:3])
                if len(result) > DEFAULT_EXTRACTIVE_FALLBACK_LENGTH + 100:
                    result = result[:DEFAULT_EXTRACTIVE_FALLBACK_LENGTH + 100] + "..."
                return result + ("." if not result.endswith('.') else "")
        
        # 폴백: 첫 N자
        return best_text[:DEFAULT_EXTRACTIVE_FALLBACK_LENGTH] + "..."

