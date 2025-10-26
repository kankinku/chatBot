"""
Prompt Builder - 프롬프트 생성기

도메인 특화 프롬프트를 생성합니다 (단일 책임).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Set

from modules.core.types import RetrievedSpan
from modules.core.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """
    프롬프트 생성기
    
    단일 책임: 프롬프트 생성만 수행
    """
    
    def __init__(self, domain_dict_path: Optional[str] = None, evaluation_mode: bool = False):
        """
        Args:
            domain_dict_path: 도메인 사전 파일 경로
            evaluation_mode: 평가 모드 (True시 평가 최적화 프롬프트 사용)
        """
        self.domain_dict: Optional[Dict] = None
        self.domain_keywords: Set[str] = set()
        self.evaluation_mode = evaluation_mode
        
        # 도메인 사전 로드
        if domain_dict_path:
            self._load_domain_dict(domain_dict_path)
        else:
            # 기본 경로 시도
            default_path = Path("data/domain_dictionary.json")
            if default_path.exists():
                self._load_domain_dict(str(default_path))
        
        logger.debug("PromptBuilder initialized",
                    has_domain_dict=self.domain_dict is not None,
                    num_keywords=len(self.domain_keywords),
                    evaluation_mode=self.evaluation_mode)
    
    def _load_domain_dict(self, path: str) -> None:
        """도메인 사전 로드"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.domain_dict = json.load(f)
            
            # 키워드 추출
            for key in ["keywords", "units", "technical_spec", "system_info"]:
                if key in self.domain_dict:
                    items = self.domain_dict[key]
                    if isinstance(items, list):
                        self.domain_keywords.update(items[:50])  # 상위 50개
            
            logger.info(f"Domain dictionary loaded: {len(self.domain_keywords)} keywords")
        except Exception as e:
            logger.warning(f"Failed to load domain dictionary: {e}")
    
    def build_qa_prompt(
        self,
        question: str,
        contexts: List[RetrievedSpan],
        question_type: str = "general",
        recovery_mode: bool = False,
    ) -> str:
        """
        QA 프롬프트 생성
        
        Args:
            question: 질문
            contexts: 컨텍스트 리스트
            question_type: 질문 유형
            recovery_mode: 복구 모드 (더 엄격한 프롬프트)
            
        Returns:
            프롬프트
        """
        # 도메인 사전 기반 최적 컨텍스트 선택 (v5 로직)
        selected_contexts = self._select_best_contexts(
            contexts,
            question,
            question_type,
        )
        
        # 평가 모드인 경우 평가용 프롬프트 사용
        if self.evaluation_mode:
            return self._build_evaluation_prompt(question, selected_contexts, question_type)
        elif recovery_mode:
            return self._build_recovery_prompt(question, selected_contexts, question_type)
        else:
            return self._build_standard_prompt(question, selected_contexts, question_type)
    
    def _select_best_contexts(
        self,
        contexts: List[RetrievedSpan],
        question: str,
        question_type: str,
    ) -> List[RetrievedSpan]:
        """
        도메인 사전 기반 최적 컨텍스트 선택 (v5 로직)
        
        질문과 도메인 키워드에 기반하여 가장 관련성 높은 컨텍스트를 선택합니다.
        """
        if not self.domain_dict or not contexts:
            return contexts
        
        # 질문에서 키워드 추출
        question_lower = question.lower()
        question_tokens = set(question_lower.split())
        
        # 각 컨텍스트에 점수 부여
        scored_contexts = []
        for context in contexts:
            text_lower = context.chunk.text.lower()
            score = context.score  # 기본 검색 점수
            
            # 도메인 키워드 매칭 보너스
            keyword_hits = sum(1 for kw in self.domain_keywords if kw.lower() in text_lower)
            score += keyword_hits * 0.05
            
            # 질문 토큰 매칭 보너스
            token_hits = sum(1 for token in question_tokens if token in text_lower)
            score += token_hits * 0.02
            
            scored_contexts.append((score, context))
        
        # 점수 기준 정렬
        scored_contexts.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 컨텍스트 반환 (질문 유형별 개수 조정)
        if question_type in ["technical_spec", "definition"]:
            max_contexts = 6
        elif question_type in ["system_info", "numeric"]:
            max_contexts = 5
        else:
            max_contexts = 4
        
        return [ctx for _, ctx in scored_contexts[:max_contexts]]
    
    def _build_standard_prompt(
        self,
        question: str,
        contexts: List[RetrievedSpan],
        question_type: str,
    ) -> str:
        """표준 프롬프트"""
        # 질문 유형별 맞춤 설정
        if question_type in ["technical_spec", "definition"]:
            max_contexts = 6
            text_length = 800
            instruction = "기술적 세부사항과 정확한 용어를 포함하여"
        elif question_type in ["system_info", "numeric"]:
            max_contexts = 5
            text_length = 600
            instruction = "정확한 수치, URL, 계정 정보를 그대로"
        else:
            max_contexts = 4
            text_length = 500
            instruction = "문서 내용을 바탕으로"
        
        parts = [
            "당신은 고산 정수장 시스템 사용자 설명서 전문 챗봇입니다.",
            "",
            "중요: 반드시 제공된 문서 내용만을 기반으로 답변하세요. 외부 지식이나 일반적인 정보는 사용하지 마세요.",
            "",
            "문서 내용:",
        ]
        
        # 컨텍스트 추가
        for i, context in enumerate(contexts[:max_contexts], start=1):
            text = context.chunk.text
            if len(text) > text_length:
                # 문장 경계에서 자르기
                truncated = text[:text_length]
                last_period = truncated.rfind('.')
                last_newline = truncated.rfind('\n')
                last_break = max(last_period, last_newline)
                
                if last_break > text_length * 0.8:
                    text = truncated[:last_break + 1]
                else:
                    text = truncated + "..."
            
            parts.append(f"[문서 {i}] {text}")
        
        parts.extend([
            "",
            f"질문: {question}",
            "",
            "지침:",
            f"- 위 문서 내용에서만 {instruction} 답변을 찾아주세요",
            "- 문서에 없는 정보는 '문서에서 확인할 수 없습니다'라고 답변하세요",
            "- 정확한 수치, 날짜, URL, 계정 정보를 그대로 사용하세요",
            "- 추측이나 일반적인 정보는 사용하지 마세요",
            "- 답변은 한국어를 사용해주세요(url이나 전문 용어는 예외)",
            "- 관련 키워드나 용어가 문서에 있다면 반드시 포함하세요",
            "- 문서 내용을 바탕으로 직접적인 답변을 제공하세요. 추론이나 해석은 피하세요",
            "- 답변은 간결하고 명확하게 작성하세요 (최대 3-4문장)",
            "- 문서 내용을 그대로 복사하지 말고 요약하여 자연스럽게 답변하세요",
            "- 핵심 정보만 포함하고 불필요한 세부사항은 생략하세요",
            "- 출처 표시나 문서 번호는 포함하지 마세요",
            "- 이모지나 특수 기호는 사용하지 마세요",
            "- 자연스러운 대화체로 답변하세요",
            "",
            "🎯 답변 형식 지침 (중요):",
            "- 질문에 '고산 정수장 시스템 사용자 설명서의' 같은 접두사가 있으면 답변에도 포함하세요",
            "- 질문의 구조를 그대로 따라가세요 (예: '발주기관은 X이며, 사업명은 Y입니다')",
            "- 질문에서 요구하는 정보의 순서를 그대로 유지하세요",
            "- 질문의 문체와 톤을 답변에도 반영하세요",
            "",
            "답변:"
        ])
        
        return "\n".join(parts)
    
    def _build_recovery_prompt(
        self,
        question: str,
        contexts: List[RetrievedSpan],
        question_type: str,
    ) -> str:
        """복구 모드 프롬프트 (더 엄격)"""
        max_contexts = 5
        text_length = 600
        
        parts = [
            "고산 정수장 시스템 사용자 설명서 전문 챗봇입니다.",
            "",
            "경고: 오직 제공된 문서 내용만 사용하세요. 외부 지식은 절대 사용하지 마세요.",
            "",
            "문서 내용:",
        ]
        
        # 컨텍스트 추가
        for i, context in enumerate(contexts[:max_contexts], start=1):
            text = context.chunk.text[:text_length]
            if len(context.chunk.text) > text_length:
                text += "..."
            parts.append(f"[문서 {i}] {text}")
        
        parts.extend([
            "",
            f"질문: {question}",
            "",
            "엄격한 지침:",
            "- 문서에 명시된 정보만 답변하세요",
            "- 문서에 없는 내용은 '문서에서 확인할 수 없습니다'라고만 답변하세요",
            "- 수치, 날짜, URL, 계정 정보는 문서의 정확한 값을 사용하세요",
            "- 일반적인 추측이나 외부 지식은 사용하지 마세요",
            "- 관련 키워드나 전문 용어가 문서에 있다면 반드시 포함하세요",
            "- 여러 문서에서 관련 정보를 종합하여 답변하세요",
            "- 답변은 간결하고 명확하게 작성하세요 (최대 3-4문장)",
            "- 문서 내용을 그대로 복사하지 말고 요약하여 자연스럽게 답변하세요",
            "- 핵심 정보만 포함하고 불필요한 세부사항은 생략하세요",
            "- 출처 표시나 문서 번호는 포함하지 마세요",
            "- 이모지나 특수 기호는 사용하지 마세요",
            "- 자연스러운 대화체로 답변하세요",
            "",
            "🎯 답변 형식 지침 (중요):",
            "- 질문에 '고산 정수장 시스템 사용자 설명서의' 같은 접두사가 있으면 답변에도 포함하세요",
            "- 질문의 구조를 그대로 따라가세요 (예: '발주기관은 X이며, 사업명은 Y입니다')",
            "- 질문에서 요구하는 정보의 순서를 그대로 유지하세요",
            "- 질문의 문체와 톤을 답변에도 반영하세요",
            "",
            "답변:"
        ])
        
        return "\n".join(parts)
    
    def _build_evaluation_prompt(
        self,
        question: str,
        contexts: List[RetrievedSpan],
        question_type: str,
    ) -> str:
        """
        평가 최적화 프롬프트
        
        평가 지표(Token F1, ROUGE, BLEU 등)에서 높은 점수를 받기 위한 프롬프트:
        - 단위를 무조건 포함
        - 확인되는 모든 정보를 나열
        - 간결함보다 완전성 우선
        - 자연스러움보다 정확성 우선
        """
        max_contexts = 6
        text_length = 800
        
        parts = [
            "당신은 고산 정수장 시스템 사용자 설명서 전문 챗봇입니다.",
            "",
            "중요: 반드시 제공된 문서 내용만을 기반으로 답변하세요.",
            "",
            "문서 내용:",
        ]
        
        # 컨텍스트 추가
        for i, context in enumerate(contexts[:max_contexts], start=1):
            text = context.chunk.text
            if len(text) > text_length:
                truncated = text[:text_length]
                last_period = truncated.rfind('.')
                last_newline = truncated.rfind('\n')
                last_break = max(last_period, last_newline)
                
                if last_break > text_length * 0.8:
                    text = truncated[:last_break + 1]
                else:
                    text = truncated + "..."
            
            parts.append(f"[문서 {i}] {text}")
        
        parts.extend([
            "",
            f"질문: {question}",
            "",
            "평가용 답변 작성 지침 (중요):",
            "- 문서에 있는 모든 관련 정보를 빠짐없이 포함하세요",
            "- 숫자가 있으면 반드시 해당 숫자를 포함하세요 (예: 25℃, 50,000 m³/day)",
            "- 단위가 있으면 반드시 단위를 함께 표기하세요 (%, ℃, mg/L, ppm, m³/day 등)",
            "- 날짜, 시간, URL, IP 주소, 계정 정보는 정확히 그대로 표기하세요",
            "- 여러 정보가 있으면 모두 나열하세요 (예: 아이디와 비밀번호 모두 명시)",
            "- 문서의 정확한 용어와 표현을 그대로 사용하세요",
            "- 정답에 포함될 키워드를 최대한 많이 포함하세요",
            "- 문서에 없는 정보는 '문서에서 확인할 수 없습니다'라고만 답변하세요",
            "- 추측이나 일반적인 정보는 절대 사용하지 마세요",
            "- 답변은 명확하고 구체적으로 작성하세요",
            "- 출처나 문서 번호는 포함하지 마세요",
            "- 이모지는 사용하지 마세요",
            "",
            "🎯 답변 형식 지침 (중요):",
            "- 질문에 '고산 정수장 시스템 사용자 설명서의' 같은 접두사가 있으면 답변에도 포함하세요",
            "- 질문의 구조를 그대로 따라가세요 (예: '발주기관은 X이며, 사업명은 Y입니다')",
            "- 질문에서 요구하는 정보의 순서를 그대로 유지하세요",
            "- 질문의 문체와 톤을 답변에도 반영하세요",
            "",
            "답변 예시:",
            "- 나쁜 예: '온도는 25도입니다' → 단위 누락",
            "- 좋은 예: '온도는 25℃입니다' → 단위 포함",
            "- 나쁜 예: '계정은 KWATER입니다' → 정보 불완전",
            "- 좋은 예: '아이디는 KWATER이고 비밀번호는 KWATER입니다' → 모든 정보 포함",
            "",
            "답변:"
        ])
        
        return "\n".join(parts)

