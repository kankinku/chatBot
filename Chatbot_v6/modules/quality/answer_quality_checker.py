"""
Answer Quality Checker

답변 품질을 자동으로 평가하고 환각 감지.
"""

from __future__ import annotations

import re
from typing import List, Dict, Tuple

from modules.core.types import RetrievedSpan
from modules.core.logger import get_logger

logger = get_logger(__name__)


class AnswerQualityChecker:
    """
    답변 품질 검증기
    
    환각 감지, 컨텍스트 일치도, 완전성 평가.
    """
    
    def __init__(
        self,
        hallucination_threshold: float = 0.5,
        alignment_threshold: float = 0.3,
        completeness_threshold: float = 0.4,
    ):
        """
        Args:
            hallucination_threshold: 환각 임계값
            alignment_threshold: 정렬 임계값
            completeness_threshold: 완전성 임계값
        """
        self.hallucination_threshold = hallucination_threshold
        self.alignment_threshold = alignment_threshold
        self.completeness_threshold = completeness_threshold
        
        logger.info("AnswerQualityChecker initialized")
    
    def check_quality(
        self,
        question: str,
        answer: str,
        contexts: List[RetrievedSpan],
    ) -> Dict[str, float]:
        """
        답변 품질 종합 평가
        
        Args:
            question: 질문
            answer: 답변
            contexts: 컨텍스트 리스트
            
        Returns:
            품질 점수 딕셔너리
        """
        scores = {}
        
        # 1. 컨텍스트 일치도
        scores['context_alignment'] = self._check_alignment(
            answer, [c.chunk.text for c in contexts]
        )
        
        # 2. 환각 점수
        scores['hallucination_score'] = self._detect_hallucination(
            answer, [c.chunk.text for c in contexts]
        )
        
        # 3. 완전성
        scores['completeness'] = self._check_completeness(
            question, answer
        )
        
        # 4. 유창성
        scores['fluency'] = self._check_fluency(answer)
        
        # 5. 정보 밀도
        scores['information_density'] = self._calculate_density(answer)
        
        # 전체 품질 점수
        scores['overall_quality'] = self._calculate_overall_quality(scores)
        
        logger.debug("Quality check completed", **scores)
        
        return scores
    
    def should_use_fallback(self, scores: Dict[str, float]) -> bool:
        """
        폴백 사용 여부 결정
        
        Args:
            scores: 품질 점수
            
        Returns:
            True면 폴백 사용
        """
        # 환각이 심하면 폴백
        if scores['hallucination_score'] > self.hallucination_threshold:
            logger.warning("High hallucination detected, using fallback")
            return True
        
        # 컨텍스트 일치도가 너무 낮으면 폴백
        if scores['context_alignment'] < self.alignment_threshold:
            logger.warning("Low context alignment, using fallback")
            return True
        
        # 완전성이 너무 낮으면 폴백
        if scores['completeness'] < self.completeness_threshold:
            logger.warning("Low completeness, using fallback")
            return True
        
        return False
    
    def _check_alignment(self, answer: str, contexts: List[str]) -> float:
        """
        컨텍스트 일치도 검사
        
        답변의 내용이 컨텍스트에 얼마나 근거하는지 확인.
        """
        if not contexts or not answer:
            return 0.0
        
        # 답변의 주요 n-gram 추출
        answer_ngrams = self._extract_ngrams(answer, n=3)
        
        if not answer_ngrams:
            return 0.0
        
        # 컨텍스트에서 매칭되는 n-gram 비율
        all_context = " ".join(contexts).lower()
        matched = sum(1 for ng in answer_ngrams if ng in all_context)
        
        alignment_score = matched / len(answer_ngrams)
        
        return alignment_score
    
    def _detect_hallucination(self, answer: str, contexts: List[str]) -> float:
        """
        환각 감지
        
        답변에 컨텍스트에 없는 숫자, 고유명사, 사실 등이 있는지 확인.
        높을수록 환각이 심함 (0.0 = 환각 없음, 1.0 = 심각한 환각).
        """
        if not answer or not contexts:
            return 0.0
        
        # 컨텍스트 통합
        all_context = " ".join(contexts).lower()
        answer_lower = answer.lower()
        
        hallucination_indicators = 0
        total_checks = 0
        
        # 1. 숫자 체크
        answer_numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        for num in answer_numbers:
            total_checks += 1
            if num not in all_context:
                hallucination_indicators += 1
        
        # 2. 고유명사 체크 (한글 대문자 시작 단어, 영문 대문자 단어)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', answer)
        for noun in proper_nouns:
            total_checks += 1
            if noun.lower() not in all_context:
                hallucination_indicators += 1
        
        # 3. 단위 체크
        units = re.findall(r'\d+\s*(?:mg/L|ppm|℃|m³/day|NTU|pH)', answer)
        for unit in units:
            total_checks += 1
            # 숫자 부분만 체크
            num_part = re.search(r'\d+', unit).group()
            if num_part not in all_context:
                hallucination_indicators += 1
        
        if total_checks == 0:
            return 0.0
        
        hallucination_score = hallucination_indicators / total_checks
        
        return hallucination_score
    
    def _check_completeness(self, question: str, answer: str) -> float:
        """
        완전성 검사
        
        질문의 요구사항을 답변이 얼마나 충족하는지 확인.
        """
        if not answer:
            return 0.0
        
        # 질문의 키워드 추출
        question_keywords = self._extract_keywords(question)
        
        if not question_keywords:
            return 0.5  # 기본값
        
        # 답변에 질문 키워드가 포함되어 있는지 확인
        answer_lower = answer.lower()
        matched = sum(1 for kw in question_keywords if kw in answer_lower)
        
        completeness_score = matched / len(question_keywords)
        
        # 답변 길이도 고려 (너무 짧으면 감점)
        if len(answer) < 20:
            completeness_score *= 0.5
        elif len(answer) < 50:
            completeness_score *= 0.8
        
        return completeness_score
    
    def _check_fluency(self, answer: str) -> float:
        """
        유창성 검사
        
        문법, 구두점, 자연스러움 확인.
        """
        if not answer:
            return 0.0
        
        score = 1.0
        
        # 1. 반복 패턴 감지 (감점)
        words = answer.split()
        if len(words) > 5:
            # 연속 3단어 반복
            for i in range(len(words) - 5):
                trigram = " ".join(words[i:i+3])
                rest = " ".join(words[i+3:])
                if trigram in rest:
                    score -= 0.2
                    break
        
        # 2. 구두점 체크
        has_punctuation = any(p in answer for p in '.!?')
        if not has_punctuation:
            score -= 0.1
        
        # 3. 문장 길이 변화 (자연스러움)
        sentences = re.split(r'[.!?]\s+', answer)
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                if avg_len > 50:  # 너무 긴 문장
                    score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_density(self, answer: str) -> float:
        """
        정보 밀도 계산
        
        답변에 얼마나 많은 정보가 담겨있는지 확인.
        """
        if not answer:
            return 0.0
        
        words = answer.split()
        if not words:
            return 0.0
        
        # 정보성 단어 (숫자, 고유명사, 긴 단어)
        info_words = 0
        for word in words:
            # 숫자 포함
            if re.search(r'\d', word):
                info_words += 1
            # 5자 이상 한글/영문
            elif re.search(r'[가-힣]{5,}|[A-Za-z]{5,}', word):
                info_words += 1
        
        density = info_words / len(words)
        
        return density
    
    def _calculate_overall_quality(self, scores: Dict[str, float]) -> float:
        """전체 품질 점수 계산 (가중 평균)"""
        weights = {
            'context_alignment': 0.35,
            'hallucination_score': -0.30,  # 음수 (낮을수록 좋음)
            'completeness': 0.25,
            'fluency': 0.10,
            'information_density': 0.10,
        }
        
        total_score = 0.0
        for key, weight in weights.items():
            if key in scores:
                total_score += scores[key] * weight
        
        # 0-1 범위로 정규화
        overall = max(0.0, min(1.0, (total_score + 0.3) / 0.6))
        
        return overall
    
    def _extract_ngrams(self, text: str, n: int = 3) -> List[str]:
        """텍스트에서 n-gram 추출"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
        
        return ngrams
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        # 조사, 접속사 제거
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과', 
                     '도', '만', '에서', '로', '으로', '이며', '입니다', '있습니다'}
        
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
        keywords = [w for w in words if len(w) >= 2 and w not in stopwords]
        
        return keywords

