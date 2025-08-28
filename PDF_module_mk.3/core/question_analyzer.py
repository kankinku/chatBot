"""
질문 분석 및 컨텍스트 관리 모듈

이 모듈은 사용자의 자연어 질문을 분석하고, 이전 대화 내용을 고려하여
적절한 컨텍스트를 유지하는 기능을 제공합니다.
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import logging
logger = logging.getLogger(__name__)

# 키워드 향상 모듈 import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.keyword_enhancer import KeywordEnhancer

class QuestionType(Enum):
    """질문 유형 분류"""
    FACTUAL = "factual"              # 사실 질문 (무엇, 언제, 어디서)
    CONCEPTUAL = "conceptual"        # 개념 질문 (어떻게, 왜)
    COMPARATIVE = "comparative"      # 비교 질문 (차이점, 유사점)
    PROCEDURAL = "procedural"        # 절차 질문 (방법, 단계)
    ANALYTICAL = "analytical"        # 분석 질문 (분석, 평가)
    FOLLOW_UP = "follow_up"         # 후속 질문 (이전 답변 관련)
    CLARIFICATION = "clarification"  # 명확화 질문 (구체적 설명 요구)

@dataclass
class ConversationItem:
    """대화 항목 데이터 클래스"""
    question: str
    answer: str
    timestamp: datetime
    question_type: QuestionType
    relevant_chunks: List[str]  # 답변에 사용된 청크 ID들
    confidence_score: float = 0.0
    metadata: Optional[Dict] = None

@dataclass 
class AnalyzedQuestion:
    """분석된 질문 데이터 클래스"""
    original_question: str
    processed_question: str
    question_type: QuestionType
    keywords: List[str]
    entities: List[str]
    intent: str
    context_keywords: List[str]  # 이전 대화에서 가져온 키워드
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

class QuestionAnalyzer:
    """
    질문 분석 및 컨텍스트 관리 클래스
    
    주요 기능:
    1. 질문 유형 분류
    2. 키워드 및 개체명 추출
    3. 이전 대화 컨텍스트 관리
    4. 질문 의도 분석
    5. 후속 질문 처리
    """
    
    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask", domain: str = "general"):
        """
        QuestionAnalyzer 초기화
        
        Args:
            embedding_model: 임베딩 모델 이름
            domain: 도메인 (general, technical, business, academic 등)
        """
        # 임베딩 모델 로드
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"질문 분석용 임베딩 모델 로드: {embedding_model}")
        except Exception as e:
            logger.warning(f"한국어 모델 로드 실패, 기본 모델 사용: {e}")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # TF-IDF 벡터라이저 (키워드 추출용)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words=None
        )
        
        # 키워드 향상 모듈 초기화
        self.keyword_enhancer = KeywordEnhancer(domain=domain)
        
        # 대화 기록 저장
        self.conversation_history: List[ConversationItem] = []
        
        # 질문 유형 분류를 위한 패턴
        self.question_patterns = {
            QuestionType.FACTUAL: [
                r'무엇', r'뭐', r'언제', r'어디', r'누구', r'몇', r'얼마',
                r'어떤', r'어느', r'몇 개', r'몇 명'
            ],
            QuestionType.CONCEPTUAL: [
                r'어떻게', r'왜', r'이유', r'원인', r'목적', r'의미',
                r'정의', r'개념', r'원리'
            ],
            QuestionType.COMPARATIVE: [
                r'차이', r'비교', r'다른점', r'같은점', r'유사', r'반대',
                r'vs', r'대비', r'구별'
            ],
            QuestionType.PROCEDURAL: [
                r'방법', r'단계', r'과정', r'절차', r'순서', r'어떻게 하',
                r'진행', r'실행'
            ],
            QuestionType.ANALYTICAL: [
                r'분석', r'평가', r'장단점', r'문제점', r'해결', r'개선',
                r'영향', r'결과', r'효과'
            ],
            QuestionType.FOLLOW_UP: [
                r'그럼', r'그러면', r'그것', r'이것', r'저것', r'앞서',
                r'위에서', r'이전', r'방금', r'더', r'추가로'
            ],
            QuestionType.CLARIFICATION: [
                r'구체적으로', r'자세히', r'정확히', r'명확히', r'구체적',
                r'상세히', r'더 자세히', r'보다 구체적'
            ]
        }
        
        # 한국어 불용어 (간단한 버전)
        self.stopwords = {
            '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
            '와', '과', '도', '만', '까지', '부터', '께서', '한테', '에게',
            '그', '저', '이', '그것', '저것', '이것', '것', '수', '때'
        }
        
        logger.info(f"QuestionAnalyzer 초기화 완료 (도메인: {domain})")
    
    def analyze_question(self, question: str, 
                        use_conversation_context: bool = True) -> AnalyzedQuestion:
        """
        질문을 종합적으로 분석
        
        Args:
            question: 사용자 질문
            use_conversation_context: 이전 대화 컨텍스트 사용 여부
            
        Returns:
            분석된 질문 객체
        """
        # 1. 질문 전처리
        processed_question = self._preprocess_question(question)
        
        # 2. 질문 유형 분류
        question_type = self._classify_question_type(processed_question)
        
        # 3. 키워드 추출
        keywords = self._extract_keywords(processed_question)
        
        # 4. 개체명 추출 (간단한 규칙 기반)
        entities = self._extract_entities(processed_question)
        
        # 5. 질문 의도 분석
        intent = self._analyze_intent(processed_question, question_type)
        
        # 6. 컨텍스트 키워드 추출
        context_keywords = []
        if use_conversation_context and self.conversation_history:
            context_keywords = self._extract_context_keywords(processed_question)
        
        # 7. 임베딩 생성
        embedding = self.embedding_model.encode(processed_question)
        
        analyzed_question = AnalyzedQuestion(
            original_question=question,
            processed_question=processed_question,
            question_type=question_type,
            keywords=keywords,
            entities=entities,
            intent=intent,
            context_keywords=context_keywords,
            embedding=embedding,
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "has_context": len(context_keywords) > 0
            }
        )
        
        logger.info(f"질문 분석 완료: {question_type.value}, 키워드: {len(keywords)}개")
        return analyzed_question
    
    def _preprocess_question(self, question: str) -> str:
        """
        질문 전처리
        
        Args:
            question: 원본 질문
            
        Returns:
            전처리된 질문
        """
        # 공백 정리
        question = re.sub(r'\s+', ' ', question.strip())
        
        # 불필요한 문자 제거
        question = re.sub(r'[^\w\s가-힣?!.]', ' ', question)
        
        # 연속된 공백 제거
        question = re.sub(r'\s+', ' ', question)
        
        return question.strip()
    
    def _classify_question_type(self, question: str) -> QuestionType:
        """
        질문 유형 분류
        
        Args:
            question: 전처리된 질문
            
        Returns:
            질문 유형
        """
        question_lower = question.lower()
        
        # 각 유형별 패턴 매칭 점수 계산
        type_scores = {}
        
        for q_type, patterns in self.question_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    score += 1
            type_scores[q_type] = score
        
        # 가장 높은 점수의 유형 반환
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        
        # 기본값: FACTUAL
        return QuestionType.FACTUAL
    
    def _extract_keywords(self, question: str) -> List[str]:
        """
        질문에서 키워드 추출 (개선된 버전)
        
        Args:
            question: 전처리된 질문
            
        Returns:
            키워드 리스트
        """
        # 1. 기본 토큰화 및 불용어 제거
        tokens = question.split()
        basic_keywords = []
        
        for token in tokens:
            # 불용어 제거
            if token not in self.stopwords and len(token) > 1:
                # 특수문자 제거
                clean_token = re.sub(r'[^\w가-힣]', '', token)
                if clean_token and len(clean_token) > 1:
                    basic_keywords.append(clean_token)
        
        # 2. 키워드 향상 (동의어, 약어 확장 등)
        enhanced_keywords = self.keyword_enhancer.enhance_keywords(basic_keywords)
        
        # 3. 도메인 특화 키워드 추가
        domain_keywords = self.keyword_enhancer.extract_domain_specific_keywords(question)
        enhanced_keywords.extend(domain_keywords)
        
        # 4. 중복 제거 및 정렬
        unique_keywords = list(set(enhanced_keywords))
        unique_keywords.sort(key=len, reverse=True)  # 긴 키워드 우선
        
        # 5. 키워드 가중치 계산 및 상위 키워드 선택
        keyword_weights = []
        for keyword in unique_keywords:
            weight = self.keyword_enhancer.calculate_keyword_weight(keyword, question)
            keyword_weights.append((keyword, weight))
        
        # 가중치로 정렬하고 상위 15개 선택
        keyword_weights.sort(key=lambda x: x[1], reverse=True)
        top_keywords = [kw for kw, _ in keyword_weights[:15]]
        
        return top_keywords
    
    def _extract_entities(self, question: str) -> List[str]:
        """
        간단한 개체명 추출 (규칙 기반)
        
        Args:
            question: 전처리된 질문
            
        Returns:
            개체명 리스트
        """
        entities = []
        
        # 숫자 패턴
        numbers = re.findall(r'\d+(?:\.\d+)?', question)
        entities.extend([f"NUMBER_{num}" for num in numbers])
        
        # 날짜 패턴 (간단한 형태만)
        dates = re.findall(r'\d{4}년|\d+월|\d+일', question)
        entities.extend([f"DATE_{date}" for date in dates])
        
        # 단위 패턴
        units = re.findall(r'\d+(?:개|명|번|회|년|월|일|시간|분|초)', question)
        entities.extend([f"UNIT_{unit}" for unit in units])
        
        return entities
    
    def _analyze_intent(self, question: str, question_type: QuestionType) -> str:
        """
        질문 의도 분석
        
        Args:
            question: 전처리된 질문
            question_type: 질문 유형
            
        Returns:
            질문 의도
        """
        intent_keywords = {
            "정보_요청": ["알려주", "알고 싶", "궁금", "설명", "무엇", "뭐"],
            "비교_요청": ["차이", "비교", "다른", "같은", "유사"],
            "방법_문의": ["어떻게", "방법", "어떤 식", "어떤 방식"],
            "확인_요청": ["맞는지", "정확한지", "확인", "검증"],
            "추가_정보": ["더", "추가", "상세", "자세히", "구체적"],
            "명확화": ["정확히", "명확히", "확실히", "분명히"]
        }
        
        question_lower = question.lower()
        intent_scores = {}
        
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            intent_scores[intent] = score
        
        # 가장 높은 점수의 의도 반환
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        # 질문 유형에 따른 기본 의도
        default_intents = {
            QuestionType.FACTUAL: "정보_요청",
            QuestionType.CONCEPTUAL: "정보_요청", 
            QuestionType.COMPARATIVE: "비교_요청",
            QuestionType.PROCEDURAL: "방법_문의",
            QuestionType.ANALYTICAL: "정보_요청",
            QuestionType.FOLLOW_UP: "추가_정보",
            QuestionType.CLARIFICATION: "명확화"
        }
        
        return default_intents.get(question_type, "정보_요청")
    
    def _extract_context_keywords(self, question: str) -> List[str]:
        """
        이전 대화에서 관련 키워드 추출
        
        Args:
            question: 현재 질문
            
        Returns:
            컨텍스트 키워드 리스트
        """
        if not self.conversation_history:
            return []
        
        context_keywords = []
        
        # 최근 3개 대화에서 키워드 추출
        recent_conversations = self.conversation_history[-3:]
        
        for conv_item in recent_conversations:
            # 이전 질문과 답변에서 키워드 추출
            prev_text = conv_item.question + " " + conv_item.answer
            prev_keywords = self._extract_keywords(prev_text)
            
            # 현재 질문과의 유사도 계산
            current_keywords = set(self._extract_keywords(question))
            prev_keywords_set = set(prev_keywords)
            
            # 공통 키워드가 있으면 관련성이 높음
            common_keywords = current_keywords & prev_keywords_set
            if common_keywords:
                context_keywords.extend(prev_keywords)
        
        # 중복 제거 및 빈도 기반 정렬
        from collections import Counter
        keyword_counts = Counter(context_keywords)
        
        # 상위 5개 키워드만 반환
        return [keyword for keyword, _ in keyword_counts.most_common(5)]
    
    def add_conversation_item(self, question: str, answer: str, 
                            relevant_chunks: List[str],
                            confidence_score: float = 0.0) -> None:
        """
        대화 항목을 기록에 추가
        
        Args:
            question: 질문
            answer: 답변
            relevant_chunks: 답변에 사용된 청크 ID들
            confidence_score: 답변 신뢰도
        """
        # 질문 분석
        analyzed_q = self.analyze_question(question, use_conversation_context=False)
        
        conv_item = ConversationItem(
            question=question,
            answer=answer,
            timestamp=datetime.now(),
            question_type=analyzed_q.question_type,
            relevant_chunks=relevant_chunks,
            confidence_score=confidence_score,
            metadata={
                "keywords": analyzed_q.keywords,
                "intent": analyzed_q.intent
            }
        )
        
        self.conversation_history.append(conv_item)
        
        # 메모리 관리: 최대 50개 대화만 유지
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
        
        logger.info(f"대화 항목 추가: {analyzed_q.question_type.value}")
    
    def get_conversation_context(self, max_items: int = 3) -> List[Dict]:
        """
        최근 대화 컨텍스트 반환
        
        Args:
            max_items: 최대 반환할 대화 항목 수
            
        Returns:
            대화 컨텍스트 리스트
        """
        recent_conversations = self.conversation_history[-max_items:]
        
        context = []
        for conv_item in recent_conversations:
            context.append({
                "question": conv_item.question,
                "answer": conv_item.answer,
                "question_type": conv_item.question_type.value,
                "timestamp": conv_item.timestamp.isoformat(),
                "confidence": conv_item.confidence_score
            })
        
        return context
    
    def find_similar_previous_questions(self, current_question: str, 
                                      threshold: float = 0.7,
                                      top_k: int = 3) -> List[Tuple[ConversationItem, float]]:
        """
        유사한 이전 질문 찾기
        
        Args:
            current_question: 현재 질문
            threshold: 유사도 임계값
            top_k: 반환할 최대 결과 수
            
        Returns:
            (대화항목, 유사도) 튜플 리스트
        """
        if not self.conversation_history:
            return []
        
        # 현재 질문 임베딩
        current_embedding = self.embedding_model.encode([current_question])
        
        # 이전 질문들과 유사도 계산
        similarities = []
        
        for conv_item in self.conversation_history:
            # 이전 질문 임베딩
            prev_embedding = self.embedding_model.encode([conv_item.question])
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(current_embedding, prev_embedding)[0][0]
            
            if similarity >= threshold:
                similarities.append((conv_item, similarity))
        
        # 유사도 순으로 정렬하여 상위 k개 반환
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_conversation_history(self, file_path: str) -> None:
        """
        대화 기록을 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로
        """
        conversation_data = []
        
        for conv_item in self.conversation_history:
            item_dict = asdict(conv_item)
            item_dict['timestamp'] = conv_item.timestamp.isoformat()
            item_dict['question_type'] = conv_item.question_type.value
            conversation_data.append(item_dict)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"대화 기록 저장 완료: {file_path}")
    
    def load_conversation_history(self, file_path: str) -> None:
        """
        파일에서 대화 기록 로드
        
        Args:
            file_path: 로드할 파일 경로
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            self.conversation_history = []
            
            for item_dict in conversation_data:
                conv_item = ConversationItem(
                    question=item_dict['question'],
                    answer=item_dict['answer'],
                    timestamp=datetime.fromisoformat(item_dict['timestamp']),
                    question_type=QuestionType(item_dict['question_type']),
                    relevant_chunks=item_dict['relevant_chunks'],
                    confidence_score=item_dict.get('confidence_score', 0.0),
                    metadata=item_dict.get('metadata', {})
                )
                self.conversation_history.append(conv_item)
            
            logger.info(f"대화 기록 로드 완료: {len(self.conversation_history)}개 항목")
            
        except Exception as e:
            logger.error(f"대화 기록 로드 실패: {e}")

# 유틸리티 함수들
def calculate_question_complexity(analyzed_question: AnalyzedQuestion) -> float:
    """
    질문 복잡도 계산
    
    Args:
        analyzed_question: 분석된 질문
        
    Returns:
        복잡도 점수 (0.0 ~ 1.0)
    """
    complexity_score = 0.0
    
    # 키워드 수에 따른 복잡도
    keyword_complexity = min(len(analyzed_question.keywords) / 10.0, 0.3)
    complexity_score += keyword_complexity
    
    # 질문 유형에 따른 복잡도
    type_complexity = {
        QuestionType.FACTUAL: 0.1,
        QuestionType.CONCEPTUAL: 0.3,
        QuestionType.COMPARATIVE: 0.4,
        QuestionType.PROCEDURAL: 0.3,
        QuestionType.ANALYTICAL: 0.5,
        QuestionType.FOLLOW_UP: 0.2,
        QuestionType.CLARIFICATION: 0.2
    }
    complexity_score += type_complexity.get(analyzed_question.question_type, 0.3)
    
    # 컨텍스트 의존성
    if analyzed_question.context_keywords:
        complexity_score += 0.2
    
    return min(complexity_score, 1.0)

if __name__ == "__main__":
    # 테스트 코드
    analyzer = QuestionAnalyzer()
    
    test_questions = [
        "이 문서에서 주요 개념이 무엇인가요?",
        "앞서 말한 방법과 다른 접근법은 어떤 것들이 있나요?",
        "A와 B의 차이점을 구체적으로 설명해주세요.",
        "이 과정을 단계별로 어떻게 진행하나요?"
    ]
    
    for question in test_questions:
        analyzed = analyzer.analyze_question(question)
        print(f"질문: {question}")
        print(f"유형: {analyzed.question_type.value}")
        print(f"의도: {analyzed.intent}")
        print(f"키워드: {analyzed.keywords}")
        print("---")
    
    print("QuestionAnalyzer 모듈이 정상적으로 로드되었습니다.")
