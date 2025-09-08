"""
질문 분석 모듈 (최적화 버전)

빠른 질문 분석을 위한 간소화된 분석기
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """질문 유형 분류 (단순화)"""
    GREETING = "greeting"            # 인사말
    FACTUAL = "factual"              # 사실 질문
    CONCEPTUAL = "conceptual"        # 개념 질문
    DATABASE_QUERY = "database_query"  # 데이터베이스 질의
    QUANTITATIVE = "quantitative"    # 정량적 질문
    UNKNOWN = "unknown"              # 알 수 없음

@dataclass
class ConversationItem:
    """대화 항목 데이터 클래스"""
    question: str
    answer: str
    timestamp: datetime
    confidence_score: float = 0.0
    metadata: Optional[Dict] = None

@dataclass 
class AnalyzedQuestion:
    """분석된 질문 데이터 클래스 (단순화)"""
    original_question: str
    processed_question: str
    question_type: QuestionType
    keywords: List[str]
    entities: List[str]
    intent: str
    context_keywords: List[str]
    # SQL 관련 필드 제거됨
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    enhanced_question: Optional[str] = None

class QuestionAnalyzer:
    """질문 분석기 (최적화)"""
    
    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask"):
        """QuestionAnalyzer 초기화"""
        # 임베딩 모델 로드
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"질문 분석용 임베딩 모델 로드: {embedding_model}")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            self.embedding_model = None
        
        # 대화 히스토리 (단순화)
        self.conversation_history: List[ConversationItem] = []
        
        # 질문 유형 패턴 (범용 RAG 시스템용)
        self.question_patterns = {
            QuestionType.GREETING: [
                r'안녕', r'안녕하세요', r'안녕하십니까', r'반갑습니다', r'반가워', r'하이', r'hi', r'hello'
            ],
            QuestionType.FACTUAL: [
                r'무엇', r'언제', r'어디서', r'누가', r'어떤'
            ],
            QuestionType.CONCEPTUAL: [
                r'어떻게', r'왜', r'원리', r'개념', r'정의'
            ],
            QuestionType.DATABASE_QUERY: [
                r'몇', r'개수', r'건수', r'총', r'평균', r'최대', r'최소',
                r'비율', r'순위', r'통계', r'분석', r'수치', r'데이터',
                r'집계', r'합계', r'누적', r'누계', r'분포', r'추세',
                r'변화량', r'증감률', r'증감 폭', r'대비', r'비교',
                r'기간별', r'월별', r'분기별', r'연도별', r'지역별',
                r'구별', r'카테고리별', r'유형별', r'얼마나', r'어느 정도'
            ],
            QuestionType.QUANTITATIVE: [
                r'얼마나', r'비율', r'순위', r'분석', r'데이터', r'통계'
            ]
        }
        
        # 키워드 추출 패턴 초기화
        self.keyword_patterns = self._load_keyword_patterns()
        
        logger.info("질문 분석기 초기화 완료")
    
    def _load_keyword_patterns(self) -> List[str]:
        """파이프라인 설정에서 키워드 패턴 로드"""
        patterns = []
        
        try:
            # 파이프라인 설정 파일에서 키워드 로드
            config_path = Path("config/pipelines/pdf_pipeline.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 기본 키워드들
                keywords = config.get("keywords", [])
                domain_keywords = config.get("domain_specific_keywords", [])
                
                # 모든 키워드를 패턴으로 변환
                all_keywords = list(set(keywords + domain_keywords))
                
                for keyword in all_keywords:
                    # 키워드를 정규식 패턴으로 변환
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    patterns.append(pattern)
                
                logger.debug(f"파이프라인 설정에서 {len(patterns)}개 키워드 패턴 로드")
            else:
                logger.warning("파이프라인 설정 파일을 찾을 수 없습니다. 기본 패턴을 사용합니다.")
                # 기본 패턴들 (기존 하드코딩된 패턴들)
                patterns = [
                    r'\b\w+구\b',  # 지역명
                    r'\b\w+시\b',  # 시명
                    r'\b\w+군\b',  # 군명
                    r'\b\w+동\b',  # 동명
                    r'\b\w+읍\b',  # 읍명
                    r'\b\w+면\b',  # 면명
                    r'\b\w+교차로\b',  # 교차로명
                    r'\b\w+역\b',  # 역명
                    r'\b\w+정류장\b',  # 정류장명
                    r'\b\w+센터\b',  # 센터명
                    r'\b\w+기관\b',  # 기관명
                    r'\b\w+회사\b',  # 회사명
                    r'\b\w+기업\b',  # 기업명
                    r'\b\w+조직\b',  # 조직명
                    r'\b\w+부서\b',  # 부서명
                    r'\b\w+팀\b',  # 팀명
                    r'\b\w+시스템\b',  # 시스템명
                    r'\b\w+서비스\b',  # 서비스명
                    r'\b\w+플랫폼\b',  # 플랫폼명
                    r'\b\w+애플리케이션\b',  # 애플리케이션명
                    r'\b\w+앱\b',  # 앱명
                    r'\b\w+프로그램\b',  # 프로그램명
                    r'\b\w+소프트웨어\b',  # 소프트웨어명
                    r'\b\w+하드웨어\b',  # 하드웨어명
                    r'\b\w+장비\b',  # 장비명
                    r'\b\w+기기\b',  # 기기명
                    r'\b\w+설비\b',  # 설비명
                    r'\b\w+시설\b',  # 시설명
                    r'\b\w+건물\b',  # 건물명
                    r'\b\w+건축물\b',  # 건축물명
                    r'\b\w+구조물\b',  # 구조물명
                    r'\b\w+인프라\b',  # 인프라명
                    r'\b\w+네트워크\b',  # 네트워크명
                    r'\b\w+서버\b',  # 서버명
                    r'\b\w+데이터베이스\b',  # 데이터베이스명
                    r'\b\w+DB\b',  # DB명
                    r'\b\w+API\b',  # API명
                    r'\b\w+인터페이스\b',  # 인터페이스명
                    r'\b\w+UI\b',  # UI명
                    r'\b\w+UX\b',  # UX명
                    r'\b\w+웹사이트\b',  # 웹사이트명
                    r'\b\w+홈페이지\b',  # 홈페이지명
                    r'\b\w+포털\b',  # 포털명
                    r'\b\w+사이트\b',  # 사이트명
                    r'\b\w+도메인\b',  # 도메인명
                    r'\b\w+URL\b',  # URL명
                    r'\b\w+링크\b',  # 링크명
                    r'\b\w+파일\b',  # 파일명
                    r'\b\w+문서\b',  # 문서명
                    r'\b\w+자료\b',  # 자료명
                    r'\b\w+보고서\b',  # 보고서명
                    r'\b\w+리포트\b',  # 리포트명
                    r'\b\w+매뉴얼\b',  # 매뉴얼명
                    r'\b\w+가이드\b',  # 가이드명
                    r'\b\w+설명서\b',  # 설명서명
                    r'\b\w+백서\b',  # 백서명
                    r'\b\w+화이트페이퍼\b',  # 화이트페이퍼명
                    r'\b\w+기술문서\b',  # 기술문서명
                    r'\b\w+기술자료\b',  # 기술자료명
                    r'\b\w+참고자료\b',  # 참고자료명
                    r'\b\w+참고문헌\b',  # 참고문헌명
                    r'\b\w+법률\b',  # 법률명
                    r'\b\w+규정\b',  # 규정명
                    r'\b\w+법규\b',  # 법규명
                    r'\b\w+법령\b',  # 법령명
                    r'\b\w+조례\b',  # 조례명
                    r'\b\w+규칙\b',  # 규칙명
                    r'\b\w+지침\b',  # 지침명
                    r'\b\w+가이드라인\b',  # 가이드라인명
                    r'\b\w+정책\b',  # 정책명
                    r'\b\w+방침\b',  # 방침명
                    r'\b\w+기준\b',  # 기준명
                    r'\b\w+표준\b',  # 표준명
                    r'\b\w+규격\b',  # 규격명
                    r'\b\w+사양\b',  # 사양명
                    r'\b\w+요구사항\b',  # 요구사항명
                    r'\b\w+규제\b',  # 규제명
                    r'\b\w+제재\b',  # 제재명
                    r'\b\w+처벌\b',  # 처벌명
                    r'\b\w+벌칙\b',  # 벌칙명
                    r'\b\w+과태료\b',  # 과태료명
                    r'\b\w+행정처분\b',  # 행정처분명
                    r'\b\w+허가\b',  # 허가명
                    r'\b\w+인가\b',  # 인가명
                    r'\b\w+승인\b',  # 승인명
                    r'\b\w+등록\b',  # 등록명
                    r'\b\w+신고\b',  # 신고명
                    r'\b\w+신청\b',  # 신청명
                    r'\b\w+제출\b',  # 제출명
                    r'\b\w+의무\b',  # 의무명
                    r'\b\w+책임\b',  # 책임명
                    r'\b\w+면책\b',  # 면책명
                    r'\b\w+배상\b',  # 배상명
                    r'\b\w+손해배상\b',  # 손해배상명
                    r'\b\w+책임보험\b',  # 책임보험명
                    r'\b\w+개인정보\b',  # 개인정보명
                    r'\b\w+개인정보보호\b',  # 개인정보보호명
                    r'\b\w+개인정보처리\b',  # 개인정보처리명
                    r'\b\w+개인정보수집\b',  # 개인정보수집명
                    r'\b\w+보안\b',  # 보안명
                    r'\b\w+보안정책\b',  # 보안정책명
                    r'\b\w+보안규정\b',  # 보안규정명
                    r'\b\w+보안지침\b',  # 보안지침명
                    r'\b\w+보안가이드\b',  # 보안가이드명
                    r'\b\w+저작권\b',  # 저작권명
                ]
                
        except Exception as e:
            logger.error(f"키워드 패턴 로드 실패: {e}")
            # 기본 패턴들 사용
            patterns = [
                r'\b사고\b', r'\b목록\b', r'\b확인\b', r'\b방법\b', r'\b알려줘\b',
                r'\b교통\b', r'\b교통사고\b', r'\b사고목록\b', r'\b사고정보\b',
                r'\b사고데이터\b', r'\b사고통계\b', r'\b사고분석\b', r'\b사고보고서\b',
                r'\b사고리포트\b', r'\b사고자료\b', r'\b사고문서\b', r'\b사고파일\b'
            ]
        
        return patterns
    
    def analyze_question(self, question: str, use_conversation_context: bool = True) -> AnalyzedQuestion:
        """질문 분석 (최적화)"""
        import time
        total_start_time = time.time()
        
        # 1. 기본 전처리
        preprocess_start = time.time()
        processed_question = self._preprocess_question(question)
        preprocess_time = time.time() - preprocess_start
        
        # 2. 질문 유형 분류
        classify_start = time.time()
        question_type = self._classify_question_type(processed_question)
        classify_time = time.time() - classify_start
        
        # 3. 키워드 추출
        keyword_start = time.time()
        keywords = self._extract_keywords(processed_question)
        keyword_time = time.time() - keyword_start
        
        # 4. 개체명 추출
        entity_start = time.time()
        entities = self._extract_entities(processed_question)
        entity_time = time.time() - entity_start
        
        # 5. 의도 분석
        intent_start = time.time()
        intent = self._analyze_intent(processed_question, question_type)
        intent_time = time.time() - intent_start
        
        # 6. 컨텍스트 키워드 (단순화)
        context_start = time.time()
        context_keywords = []
        if use_conversation_context and self.conversation_history:
            context_keywords = self._extract_context_keywords()
        context_time = time.time() - context_start
        
        # 7. SQL 요구사항 확인 제거됨
        sql_time = 0.0
        
        # 8. 임베딩 생성 (가장 오래 걸릴 수 있는 부분)
        embedding_start = time.time()
        embedding = None
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(processed_question)
            except Exception as e:
                logger.warning(f"임베딩 생성 실패: {e}")
        embedding_time = time.time() - embedding_start
        
        # 9. 향상된 질문 생성
        enhance_start = time.time()
        enhanced_question = self._enhance_question(processed_question, keywords, entities)
        enhance_time = time.time() - enhance_start
        
        # 10. 메타데이터 생성
        metadata = {
            "processing_times": {
                "preprocess": preprocess_time,
                "classify": classify_time,
                "keyword": keyword_time,
                "entity": entity_time,
                "intent": intent_time,
                "context": context_time,
                "sql": sql_time,
                "embedding": embedding_time,
                "enhance": enhance_time,
                "total": time.time() - total_start_time
            },
            "keyword_count": len(keywords),
            "entity_count": len(entities),
            "context_keyword_count": len(context_keywords)
        }
        
        # 결과 로깅
        logger.info(f"질문 분석 완료: {question_type.value}, 키워드: {len(keywords)}개")
        
        return AnalyzedQuestion(
            original_question=question,
            processed_question=processed_question,
            question_type=question_type,
            keywords=keywords,
            entities=entities,
            intent=intent,
            context_keywords=context_keywords,
            embedding=embedding,
            metadata=metadata,
            enhanced_question=enhanced_question
        )
    
    def _preprocess_question(self, question: str) -> str:
        """질문 전처리"""
        # 소문자 변환
        question = question.lower()
        
        # 특수문자 정리
        question = re.sub(r'[^\w\s가-힣]', ' ', question)
        
        # 연속된 공백 정리
        question = re.sub(r'\s+', ' ', question)
        
        return question.strip()
    
    def _classify_question_type(self, question: str) -> QuestionType:
        """질문 유형 분류"""
        for question_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    return question_type
        
        return QuestionType.UNKNOWN
    
    def _extract_keywords(self, question: str) -> List[str]:
        """키워드 추출"""
        keywords = []
        
        # 패턴 기반 키워드 추출
        for pattern in self.keyword_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            keywords.extend(matches)
        
        # 중복 제거 및 정렬
        keywords = list(set(keywords))
        keywords.sort()
        
        return keywords
    
    def _extract_entities(self, question: str) -> List[str]:
        """개체명 추출 (개선된 버전)"""
        entities = []
        
        # 세종시 동/읍/면 패턴
        sejong_patterns = [
            r'([가-힣]+동)',  # 동 패턴
            r'([가-힣]+읍)',  # 읍 패턴
            r'([가-힣]+면)',  # 면 패턴
            r'세종특별자치시([가-힣]+)',  # 세종특별자치시 패턴
        ]
        
        for pattern in sejong_patterns:
            matches = re.findall(pattern, question)
            entities.extend(matches)
        
        # 교차로명 패턴 (세종특별자치시 형식)
        intersection_patterns = [
            r'세종특별자치시[가-힣]+\(\d+\)',  # 세종특별자치시조치원읍(1) 형식
            r'세종특별자치시[가-힣]+',  # 세종특별자치시조치원읍 형식
        ]
        
        for pattern in intersection_patterns:
            matches = re.findall(pattern, question)
            entities.extend(matches)
        
        # 기존 패턴들
        existing_patterns = [
            r'\b\w+구\b',  # 지역명
            r'\b\w+교차로\b',  # 교차로명
            r'\b\w+역\b',  # 역명
        ]
        
        for pattern in existing_patterns:
            matches = re.findall(pattern, question)
            entities.extend(matches)
        
        # 교차로명 정규화
        normalized_entities = []
        for entity in entities:
            if "세종특별자치시" in entity:
                normalized = self._normalize_intersection_name(entity)
                normalized_entities.append(normalized)
            else:
                normalized_entities.append(entity)
        
        return list(set(normalized_entities))  # 중복 제거
    
    def _normalize_intersection_name(self, intersection_name: str) -> str:
        """교차로명 정규화 (세종시 형식 -> 지역명)"""
        # 교차로 매핑 테이블
        intersection_mapping = {
            # 조치원읍 교차로들
            "세종특별자치시조치원읍": "조치원읍",
            "세종특별자치시조치원읍(1)": "조치원읍",
            "세종특별자치시조치원읍(2)": "조치원읍",
            "세종특별자치시조치원읍(3)": "조치원읍",
            "세종특별자치시조치원읍(4)": "조치원읍",
            "세종특별자치시조치원읍(5)": "조치원읍",
            "세종특별자치시조치원읍(6)": "조치원읍",
            "세종특별자치시조치원읍(7)": "조치원읍",
            "세종특별자치시조치원읍(8)": "조치원읍",
            "세종특별자치시조치원읍(9)": "조치원읍",
            "세종특별자치시조치원읍(10)": "조치원읍",
            
            # 연기면 교차로들
            "세종특별자치시연기면": "연기면",
            "세종특별자치시연기면(1)": "연기면",
            "세종특별자치시연기면(2)": "연기면",
            "세종특별자치시연기면(3)": "연기면",
            "세종특별자치시연기면(4)": "연기면",
            "세종특별자치시연기면(5)": "연기면",
            
            # 연동면 교차로들
            "세종특별자치시연동면": "연동면",
            "세종특별자치시연동면(1)": "연동면",
            "세종특별자치시연동면(2)": "연동면",
            "세종특별자치시연동면(3)": "연동면",
            "세종특별자치시연동면(4)": "연동면",
            "세종특별자치시연동면(5)": "연동면",
            "세종특별자치시연동면(6)": "연동면",
            "세종특별자치시연동면(7)": "연동면",
            "세종특별자치시연동면(8)": "연동면",
            "세종특별자치시연동면(9)": "연동면",
            "세종특별자치시연동면(10)": "연동면",
            
            # 기타 지역들
            "세종특별자치시": "세종특별자치시",
        }
        
        return intersection_mapping.get(intersection_name, intersection_name)
    
    def _analyze_intent(self, question: str, question_type: QuestionType) -> str:
        """의도 분석"""
        if question_type == QuestionType.GREETING:
            return "greeting"
        elif question_type == QuestionType.DATABASE_QUERY:
            return "database_query"
        elif question_type == QuestionType.FACTUAL:
            return "factual_inquiry"
        elif question_type == QuestionType.CONCEPTUAL:
            return "conceptual_inquiry"
        elif question_type == QuestionType.QUANTITATIVE:
            return "quantitative_analysis"
        else:
            return "general_inquiry"
    
    def _extract_context_keywords(self) -> List[str]:
        """컨텍스트 키워드 추출 (단순화)"""
        if not self.conversation_history:
            return []
        
        # 최근 대화에서 키워드 추출
        recent_keywords = []
        for item in self.conversation_history[-3:]:  # 최근 3개 대화만
            keywords = self._extract_keywords(item.question)
            recent_keywords.extend(keywords)
        
        return list(set(recent_keywords))
    
    # SQL 요구사항 확인 메서드 제거됨
    
    def _enhance_question(self, question: str, keywords: List[str], entities: List[str]) -> str:
        """질문 향상"""
        enhanced_parts = [question]
        
        # 키워드 추가
        if keywords:
            enhanced_parts.append(f"키워드: {', '.join(keywords[:5])}")
        
        # 개체명 추가
        if entities:
            enhanced_parts.append(f"개체: {', '.join(entities[:3])}")
        
        return " | ".join(enhanced_parts)
    
    def add_conversation_item(self, question: str, answer: str, confidence_score: float = 0.0):
        """대화 항목 추가"""
        item = ConversationItem(
            question=question,
            answer=answer,
            timestamp=datetime.now(),
            confidence_score=confidence_score
        )
        self.conversation_history.append(item)
        
        # 대화 히스토리 크기 제한 (최근 10개만 유지)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def clear_conversation_history(self):
        """대화 히스토리 초기화"""
        self.conversation_history.clear()
        logger.info("대화 히스토리 초기화 완료")
