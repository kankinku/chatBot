"""
환상(Hallucination) 방지 모듈

챗봇이 주어진 정보 외의 데이터로 추론하는 것을 막는 강화된 검증 시스템
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class HallucinationType(Enum):
    """환상 유형"""
    GENERAL_KNOWLEDGE = "general_knowledge"  # 일반 지식 사용
    SPECULATION = "speculation"  # 추측성 답변
    OUTSIDE_CONTEXT = "outside_context"  # 컨텍스트 외 정보
    NUMERIC_HALLUCINATION = "numeric_hallucination"  # 수치 환상
    MODEL_HALLUCINATION = "model_hallucination"  # 모델명 환상

@dataclass
class HallucinationDetection:
    """환상 감지 결과"""
    is_hallucination: bool
    hallucination_type: Optional[HallucinationType]
    confidence: float
    detected_patterns: List[str]
    suggested_correction: Optional[str]

class HallucinationPrevention:
    """환상 방지 시스템"""
    
    def __init__(self):
        """환상 방지 시스템 초기화"""
        
        # 환상 패턴 정의
        self.hallucination_patterns = {
            HallucinationType.GENERAL_KNOWLEDGE: [
                r"일반적으로\s+알려진",
                r"보통의\s+경우",
                r"대부분의\s+경우",
                r"일반적인\s+원칙",
                r"일반적으로\s+사용되는",
                r"표준적으로",
                r"전형적으로",
                r"대체로",
                r"보통",
                r"일반적으로\s+권장되는"
            ],
            
            HallucinationType.SPECULATION: [
                r"추정되며",
                r"추정됩니다",
                r"예상됩니다",
                r"가능성이\s+높습니다",
                r"아마도",
                r"아마",
                r"~일\s+것\s+입니다",
                r"~할\s+것\s+으로\s+보입니다",
                r"~로\s+보입니다",
                r"~로\s+추정됩니다"
            ],
            
            HallucinationType.OUTSIDE_CONTEXT: [
                r"다른\s+문서에\s+따르면",
                r"일반적인\s+정수처리",
                r"표준\s+정수처리",
                r"일반적인\s+AI\s+모델",
                r"일반적으로\s+사용되는\s+모델",
                r"보통\s+사용되는",
                r"일반적인\s+방법",
                r"표준\s+방법"
            ],
            
            HallucinationType.NUMERIC_HALLUCINATION: [
                r"\d+(?:\.\d+)?\s*(?:mg/L|NTU|RPM|℃|°C|%|시간|분|초|m³/h|㎥/h)",
                r"\d+(?:\.\d+)?\s*(?:~|~|to|to)\s*\d+(?:\.\d+)?",
                r"상관계수\s*\d+(?:\.\d+)?",
                r"MAE:\s*\d+(?:\.\d+)?",
                r"MSE:\s*\d+(?:\.\d+)?",
                r"RMSE:\s*\d+(?:\.\d+)?",
                r"R²:\s*\d+(?:\.\d+)?"
            ],
            
            HallucinationType.MODEL_HALLUCINATION: [
                r"LSTM\s*\(Long\s*Short-Term\s*Memory",
                r"XGB\s*\(Extreme\s*Gradient\s*Boosting",
                r"Random\s*Forest",
                r"Support\s*Vector\s*Machine",
                r"Neural\s*Network",
                r"Deep\s*Learning",
                r"Machine\s*Learning"
            ]
        }
        
        # 금지된 일반 지식 표현
        self.forbidden_general_knowledge = [
            "일반적으로 알려진",
            "보통의 경우",
            "대부분의 경우",
            "일반적인 원칙",
            "표준적으로",
            "전형적으로",
            "대체로",
            "보통",
            "일반적으로 권장되는"
        ]
        
        # 허용된 문서 기반 표현
        self.allowed_document_expressions = [
            "문서에 따르면",
            "제공된 문서에서",
            "문서에 명시된",
            "문서에 기재된",
            "문서에 기록된",
            "문서에 설명된"
        ]
        
        logger.info("환상 방지 시스템 초기화 완료")
    
    def detect_hallucination(self, answer: str, context: str, question: str) -> HallucinationDetection:
        """답변에서 환상 패턴 감지"""
        try:
            answer_lower = answer.lower()
            context_lower = context.lower()
            question_lower = question.lower()
            
            detected_patterns = []
            hallucination_type = None
            confidence = 0.0
            
            # 1. 일반 지식 사용 감지
            general_knowledge_detected = self._detect_general_knowledge(answer_lower)
            if general_knowledge_detected:
                detected_patterns.extend(general_knowledge_detected)
                hallucination_type = HallucinationType.GENERAL_KNOWLEDGE
                confidence += 0.8
            
            # 2. 추측성 답변 감지
            speculation_detected = self._detect_speculation(answer_lower)
            if speculation_detected:
                detected_patterns.extend(speculation_detected)
                if not hallucination_type:
                    hallucination_type = HallucinationType.SPECULATION
                confidence += 0.6
            
            # 3. 컨텍스트 외 정보 감지
            outside_context_detected = self._detect_outside_context(answer_lower, context_lower)
            if outside_context_detected:
                detected_patterns.extend(outside_context_detected)
                if not hallucination_type:
                    hallucination_type = HallucinationType.OUTSIDE_CONTEXT
                confidence += 0.7
            
            # 4. 수치 환상 감지
            numeric_hallucination = self._detect_numeric_hallucination(answer, context)
            if numeric_hallucination:
                detected_patterns.extend(numeric_hallucination)
                if not hallucination_type:
                    hallucination_type = HallucinationType.NUMERIC_HALLUCINATION
                confidence += 0.9
            
            # 5. 모델명 환상 감지
            model_hallucination = self._detect_model_hallucination(answer_lower, context_lower)
            if model_hallucination:
                detected_patterns.extend(model_hallucination)
                if not hallucination_type:
                    hallucination_type = HallucinationType.MODEL_HALLUCINATION
                confidence += 0.9
            
            # 6. 문서-답변 일치도 검사
            document_match_ratio = self._calculate_document_match_ratio(answer_lower, context_lower)
            if document_match_ratio < 0.3:  # 30% 미만 일치시 환상 가능성 높음
                detected_patterns.append(f"문서 일치도 낮음: {document_match_ratio:.2f}")
                if not hallucination_type:
                    hallucination_type = HallucinationType.OUTSIDE_CONTEXT
                confidence += 0.5
            
            # 7. 질문-답변 관련성 검사
            question_relevance = self._calculate_question_relevance(answer_lower, question_lower)
            if question_relevance < 0.4:  # 40% 미만 관련시 환상 가능성
                detected_patterns.append(f"질문 관련도 낮음: {question_relevance:.2f}")
                if not hallucination_type:
                    hallucination_type = HallucinationType.OUTSIDE_CONTEXT
                confidence += 0.4
            
            is_hallucination = len(detected_patterns) > 0 and confidence > 0.5
            
            # 수정 제안 생성
            suggested_correction = None
            if is_hallucination:
                suggested_correction = self._generate_correction_suggestion(
                    answer, context, question, hallucination_type, detected_patterns
                )
            
            return HallucinationDetection(
                is_hallucination=is_hallucination,
                hallucination_type=hallucination_type,
                confidence=min(confidence, 1.0),
                detected_patterns=detected_patterns,
                suggested_correction=suggested_correction
            )
            
        except Exception as e:
            logger.error(f"환상 감지 중 오류: {e}")
            return HallucinationDetection(
                is_hallucination=False,
                hallucination_type=None,
                confidence=0.0,
                detected_patterns=[],
                suggested_correction=None
            )
    
    def _detect_general_knowledge(self, answer_lower: str) -> List[str]:
        """일반 지식 사용 감지"""
        detected = []
        
        for pattern in self.hallucination_patterns[HallucinationType.GENERAL_KNOWLEDGE]:
            matches = re.findall(pattern, answer_lower)
            if matches:
                detected.extend([f"일반 지식 사용: {match}" for match in matches])
        
        return detected
    
    def _detect_speculation(self, answer_lower: str) -> List[str]:
        """추측성 답변 감지"""
        detected = []
        
        for pattern in self.hallucination_patterns[HallucinationType.SPECULATION]:
            matches = re.findall(pattern, answer_lower)
            if matches:
                detected.extend([f"추측성 표현: {match}" for match in matches])
        
        return detected
    
    def _detect_outside_context(self, answer_lower: str, context_lower: str) -> List[str]:
        """컨텍스트 외 정보 감지"""
        detected = []
        
        # 컨텍스트에 없는 정보가 답변에 포함되었는지 검사
        answer_sentences = re.split(r'[.!?]\s+', answer_lower)
        context_sentences = re.split(r'[.!?]\s+', context_lower)
        
        for answer_sentence in answer_sentences:
            if len(answer_sentence.strip()) < 10:
                continue
            
            # 답변 문장이 컨텍스트와 얼마나 일치하는지 검사
            sentence_tokens = re.findall(r'\w+', answer_sentence)
            if len(sentence_tokens) < 3:
                continue
            
            # 컨텍스트에서 해당 문장과 유사한 문장 찾기
            max_similarity = 0
            for context_sentence in context_sentences:
                context_tokens = re.findall(r'\w+', context_sentence)
                if len(context_tokens) < 3:
                    continue
                
                # 토큰 기반 유사도 계산
                common_tokens = set(sentence_tokens) & set(context_tokens)
                similarity = len(common_tokens) / max(len(sentence_tokens), len(context_tokens))
                max_similarity = max(max_similarity, similarity)
            
            # 유사도가 너무 낮으면 컨텍스트 외 정보로 판단
            if max_similarity < 0.2:
                detected.append(f"컨텍스트 외 정보: {answer_sentence[:50]}...")
        
        return detected
    
    def _detect_numeric_hallucination(self, answer: str, context: str) -> List[str]:
        """수치 환상 감지"""
        detected = []
        
        # 답변에서 수치 패턴 추출
        answer_numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        context_numbers = re.findall(r'\d+(?:\.\d+)?', context)
        
        # 답변의 수치가 컨텍스트에 없는 경우 감지
        for number in answer_numbers:
            if number not in context_numbers:
                # 단위와 함께 검사
                number_with_unit = re.search(rf'{re.escape(number)}\s*(?:mg/L|NTU|RPM|℃|°C|%|시간|분|초|m³/h|㎥/h)', answer)
                if number_with_unit:
                    detected.append(f"컨텍스트에 없는 수치: {number_with_unit.group()}")
        
        return detected
    
    def _detect_model_hallucination(self, answer_lower: str, context_lower: str) -> List[str]:
        """모델명 환상 감지"""
        detected = []
        
        for pattern in self.hallucination_patterns[HallucinationType.MODEL_HALLUCINATION]:
            matches = re.findall(pattern, answer_lower)
            for match in matches:
                # 해당 모델명이 컨텍스트에 있는지 확인
                if match.lower() not in context_lower:
                    detected.append(f"컨텍스트에 없는 모델명: {match}")
        
        return detected
    
    def _calculate_document_match_ratio(self, answer_lower: str, context_lower: str) -> float:
        """문서-답변 일치도 계산"""
        try:
            # 토큰 추출
            answer_tokens = set(re.findall(r'\w+', answer_lower))
            context_tokens = set(re.findall(r'\w+', context_lower))
            
            if not answer_tokens:
                return 0.0
            
            # 공통 토큰 비율 계산
            common_tokens = answer_tokens & context_tokens
            match_ratio = len(common_tokens) / len(answer_tokens)
            
            return match_ratio
            
        except Exception:
            return 0.0
    
    def _calculate_question_relevance(self, answer_lower: str, question_lower: str) -> float:
        """질문-답변 관련성 계산"""
        try:
            # 질문과 답변의 키워드 추출
            question_tokens = set(re.findall(r'\w+', question_lower))
            answer_tokens = set(re.findall(r'\w+', answer_lower))
            
            if not question_tokens or not answer_tokens:
                return 0.0
            
            # 공통 키워드 비율 계산
            common_keywords = question_tokens & answer_tokens
            relevance = len(common_keywords) / len(question_tokens)
            
            return relevance
            
        except Exception:
            return 0.0
    
    def _generate_correction_suggestion(self, answer: str, context: str, question: str, 
                                      hallucination_type: HallucinationType, 
                                      detected_patterns: List[str]) -> str:
        """수정 제안 생성"""
        
        if hallucination_type == HallucinationType.GENERAL_KNOWLEDGE:
            return "제공된 문서에서 해당 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
        
        elif hallucination_type == HallucinationType.SPECULATION:
            return "제공된 문서에서 해당 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
        
        elif hallucination_type == HallucinationType.OUTSIDE_CONTEXT:
            return "제공된 문서에서 해당 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
        
        elif hallucination_type == HallucinationType.NUMERIC_HALLUCINATION:
            return "제공된 문서에서 해당 수치 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
        
        elif hallucination_type == HallucinationType.MODEL_HALLUCINATION:
            return "제공된 문서에서 해당 모델 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
        
        else:
            return "제공된 문서에서 해당 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
    
    def create_strict_prompt(self, context: str, question: str) -> str:
        """엄격한 프롬프트 생성 (환상 방지 강화)"""
        
        strict_instructions = """
중요한 답변 지침:
1. 제공된 문서에 명시된 내용만을 기반으로 답변하세요.
2. 일반적인 지식이나 추측을 사용하지 마세요.
3. "일반적으로 알려진", "보통의 경우" 등의 표현을 사용하지 마세요.
4. 문서에 없는 수치나 모델명을 언급하지 마세요.
5. 문서에 해당 정보가 없으면 "문서에 해당 정보가 명시되어 있지 않습니다"라고 답하세요.
6. 추측이나 가정을 바탕으로 한 답변을 하지 마세요.
7. 문서의 구체적인 내용을 인용하여 답변하세요.
"""
        
        base_prompt = f"""
문서 내용:
{context}

질문: {question}

{strict_instructions}

위 문서 내용을 바탕으로 질문에 답변하세요. 문서에 명시된 내용만을 사용하고, 일반적인 지식이나 추측은 사용하지 마세요.
"""
        
        return base_prompt
    
    def validate_answer_strictly(self, answer: str, context: str, question: str) -> Tuple[str, bool]:
        """엄격한 답변 검증 및 수정"""
        
        # 환상 감지
        detection = self.detect_hallucination(answer, context, question)
        
        if detection.is_hallucination and detection.confidence > 0.7:
            logger.warning(f"환상 감지됨: {detection.hallucination_type.value}, 신뢰도: {detection.confidence:.2f}")
            logger.warning(f"감지된 패턴: {detection.detected_patterns}")
            
            # 수정된 답변 반환
            corrected_answer = detection.suggested_correction or "제공된 문서에서 해당 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
            return corrected_answer, True
        
        return answer, False

# 편의 함수
def create_hallucination_prevention() -> HallucinationPrevention:
    """환상 방지 시스템 생성"""
    return HallucinationPrevention()
