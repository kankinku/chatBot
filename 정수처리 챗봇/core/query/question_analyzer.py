"""
질문 분석 모듈 (최적화 버전)

빠른 질문 분석을 위한 간소화된 분석기
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# 한국어 의존성 파싱을 위한 라이브러리
try:
    import spacy
    from spacy import displacy
    # 한국어 모델 로드 시도
    try:
        nlp = spacy.load("ko_core_news_sm")
        DEPENDENCY_PARSING_AVAILABLE = True
    except OSError:
        # 한국어 모델이 없으면 영어 모델로 대체 (한국어 텍스트도 어느 정도 처리 가능)
        try:
            nlp = spacy.load("en_core_web_sm")
            DEPENDENCY_PARSING_AVAILABLE = True
        except OSError:
            nlp = None
            DEPENDENCY_PARSING_AVAILABLE = False
except ImportError:
    nlp = None
    DEPENDENCY_PARSING_AVAILABLE = False

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
    answer_target: str  # 답변 목표 (구체적인 목적, 예: "약품 주입량")
    target_type: str    # 목표 유형 (quantitative_value, qualitative_definition, procedural, etc.)
    value_intent: str   # 값의 의도 (definition, value, process, comparison)
    confidence_score: float  # 추출 신뢰도 (0.0 ~ 1.0)
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
        
        # 후속 질문 판별을 위한 간단 지시어/연결어 목록 (한국어 중심)
        self._followup_markers = {
            "그건", "그거", "그럼", "그렇다면", "자세히", "자세하게", "더", "더줘", "더 알려줘",
            "계속", "이어서", "추가로", "그 다음", "그 이후", "좀 더", "좀더"
        }
        
        # Answer Target 추출을 위한 개선된 패턴 정의
        self.answer_target_patterns = {
            # 정량적 값을 묻는 질문 (얼마나, 몇, 수치 등)
            "quantitative_value": {
                "value_indicators": ["얼마나", "몇", "수치", "값", "비율", "농도", "속도", "압력", "온도", "유량", "주입률", "개도율", "효율", "성능", "지표", "량"],
                "patterns": [
                    r'([가-힣\w\s]+)\s*(?:이|가|을|를)\s*(?:얼마나|몇)',
                    r'(?:얼마나|몇)\s*([가-힣\w\s]+)',
                    r'([가-힣\w\s]+)\s*(?:량|수치|값|비율|농도|속도|압력|온도|유량|주입률|개도율|효율|성능|지표)',
                    r'([가-힣\w\s]+)\s*(?:과|와)\s*(?:가장|높은|낮은)\s*(?:상관관계|관계)',
                ]
            },
            # 정의/개념을 묻는 질문 (무엇, 어떤, 정의 등)
            "qualitative_definition": {
                "value_indicators": ["무엇", "어떤", "정의", "개념", "특징", "장점", "단점", "목적", "기능", "역할", "방법", "원리"],
                "patterns": [
                    r'([가-힣\w\s]+)\s*(?:은|는|이|가)\s*(?:무엇|어떤)',
                    r'(?:무엇|어떤)\s*([가-힣\w\s]+)',
                    r'([가-힣\w\s]+)\s*(?:의|에서)\s*(?:목적|기능|역할|방법|원리|개념|정의)',
                    r'([가-힣\w\s]+)\s*(?:이|가)\s*(?:예측하는|제어하는|관리하는)',
                ]
            },
            # 절차/과정을 묻는 질문 (어떻게, 절차 등)
            "procedural": {
                "value_indicators": ["어떻게", "절차", "과정", "단계", "순서", "방식", "조치", "대응", "결정"],
                "patterns": [
                    r'([가-힣\w\s]+)\s*(?:은|는)\s*(?:어떻게)',
                    r'(?:어떻게)\s*([가-힣\w\s]+)',
                    r'([가-힣\w\s]+)\s*(?:의|에서)\s*(?:절차|과정|단계|순서|방식)',
                ]
            },
            # 비교/관계를 묻는 질문
            "comparative": {
                "value_indicators": ["비교", "차이", "대비", "관계", "상관관계", "영향", "높은", "낮은", "가장"],
                "patterns": [
                    r'([가-힣\w\s]+)\s*(?:과|와)\s*(?:가장|높은|낮은)\s*(?:상관관계|관계)',
                    r'([가-힣\w\s]+)\s*(?:의|에서)\s*(?:영향|관계)',
                ]
            }
        }
    
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
        
        # 5-1. Answer Target 추출
        target_start = time.time()
        answer_target, target_type, value_intent, confidence_score = self._extract_answer_target(processed_question)
        target_time = time.time() - target_start
        
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
        # 10-a. 키워드쌍(주체/속성) 추출
        try:
            keyword_pair = self._extract_keyword_pair(processed_question)
        except Exception:
            keyword_pair = {"subject": "", "attribute": ""}

        metadata = {
            "processing_times": {
                "preprocess": preprocess_time,
                "classify": classify_time,
                "keyword": keyword_time,
                "entity": entity_time,
                "intent": intent_time,
                "target": target_time,
                "context": context_time,
                "sql": sql_time,
                "embedding": embedding_time,
                "enhance": enhance_time,
                "total": time.time() - total_start_time
            },
            "keyword_count": len(keywords),
            "entity_count": len(entities),
            "context_keyword_count": len(context_keywords),
            "keyword_pair": keyword_pair,
            "answer_target": answer_target,
            "target_type": target_type,
            "value_intent": value_intent,
            "confidence_score": confidence_score
        }
        
        # 결과 로깅
        logger.info(f"질문 분석 완료: {question_type.value}, 키워드: {len(keywords)}개, 목표: {answer_target} ({target_type}, {value_intent})")
        
        return AnalyzedQuestion(
            original_question=question,
            processed_question=processed_question,
            question_type=question_type,
            keywords=keywords,
            entities=entities,
            intent=intent,
            context_keywords=context_keywords,
            answer_target=answer_target,
            target_type=target_type,
            value_intent=value_intent,
            confidence_score=confidence_score,
            embedding=embedding,
            metadata=metadata,
            enhanced_question=enhanced_question
        )
    
    def detect_follow_up(self, question: str) -> Dict[str, Any]:
        """후속 질문 여부를 간단 휴리스틱+임베딩으로 판정한다.
        반환: {is_follow_up: bool, confidence: float, reason: str}
        """
        q = (question or "").strip()
        if not q:
            return {"is_follow_up": False, "confidence": 0.0, "reason": "empty"}

        # 1) 길이/형식 휴리스틱 (한국어는 문자수 기준 사용)
        char_len = len(q)
        short_heuristic = char_len < 8  # 매우 짧은 발화

        # 2) 지시어/연결어 포함 여부
        lowered = q.lower()
        has_marker = any(m in lowered for m in self._followup_markers)

        # 3) 도메인 키워드 유무(간단 대리변수: 내부 키워드 추출 결과 개수)
        processed = self._preprocess_question(q)
        domain_keywords = self._extract_keywords(processed)
        has_domain_term = len(domain_keywords) > 0

        # 4) 최근 대화와의 임베딩 유사도(있으면 사용)
        sim_score = 0.0
        if self.embedding_model and self.conversation_history:
            try:
                last = self.conversation_history[-1]
                base_text = f"{last.question} \n {last.answer[:200]}"
                vec_q = self.embedding_model.encode(processed)
                vec_b = self.embedding_model.encode(base_text)
                # 코사인 유사도
                a = vec_q / (np.linalg.norm(vec_q) + 1e-8)
                b = vec_b / (np.linalg.norm(vec_b) + 1e-8)
                sim_score = float(np.dot(a, b))
            except Exception:
                sim_score = 0.0

        # 의사결정: 안전 방향(후속 과대판단)
        # - 매우 짧음 또는 지시어 포함 → 후속 가중치 큼
        # - 도메인 키워드가 없고(sim 낮아도) 독립 의미 빈약 → 후속
        reasons: List[str] = []
        score = 0.0
        if short_heuristic:
            score += 0.5
            reasons.append("short")
        if has_marker:
            score += 0.35
            reasons.append("marker")
        if not has_domain_term:
            score += 0.25
            reasons.append("no_domain_term")
        # 유사도는 보조: 높으면 약간 가산
        if sim_score >= 0.45:
            score += 0.15
            reasons.append("high_sim")

        is_follow_up = score >= 0.5
        confidence = min(0.95, max(0.2, score))
        return {"is_follow_up": is_follow_up, "confidence": confidence, "reason": ",".join(reasons), "similarity": sim_score}

    def get_last_route(self) -> Optional[str]:
        """최근 대화 항목의 라우트를 메타데이터에서 복원한다."""
        if not self.conversation_history:
            return None
        last = self.conversation_history[-1]
        if last.metadata and isinstance(last.metadata, dict):
            return last.metadata.get("route")
        return None
    
    def _preprocess_question(self, question: str) -> str:
        """질문 전처리 (복합명사 표준화 포함)"""
        # 소문자 변환
        question = question.lower()
        
        # 복합명사 표준화 (공백 제거)
        question = self._normalize_compound_nouns(question)
        
        # 특수문자 정리
        question = re.sub(r'[^\w\s가-힣]', ' ', question)
        
        # 연속된 공백 정리
        question = re.sub(r'\s+', ' ', question)
        
        return question.strip()
    
    def _normalize_compound_nouns(self, question: str) -> str:
        """복합명사 표준화 (공백 제거)"""
        # 정수처리 도메인 특화 복합명사 매핑
        compound_noun_mappings = {
            # 공정 관련
            r'착수\s+공정': '착수공정',
            r'약품\s+공정': '약품공정', 
            r'혼화\s+응집\s+공정': '혼화응집공정',
            r'혼화\s+공정': '혼화공정',
            r'응집\s+공정': '응집공정',
            r'침전\s+공정': '침전공정',
            r'여과\s+공정': '여과공정',
            r'소독\s+공정': '소독공정',
            r'슬러지\s+처리\s+공정': '슬러지처리공정',
            r'슬러지\s+공정': '슬러지공정',
            
            # 시스템 관련
            r'ems\s+시스템': 'ems시스템',
            r'pms\s+시스템': 'pms시스템',
            r'스마트\s+정수장': '스마트정수장',
            r'ai\s+모델': 'ai모델',
            r'머신러닝\s+모델': '머신러닝모델',
            r'딥러닝\s+모델': '딥러닝모델',
            
            # 장비/설비 관련
            r'교반기\s+회전속도': '교반기회전속도',
            r'펌프\s+세부\s+현황': '펌프세부현황',
            r'밸브\s+개도율': '밸브개도율',
            r'수위\s+목표값': '수위목표값',
            r'잔류염소\s+농도': '잔류염소농도',
            r'원수\s+탁도': '원수탁도',
            r'응집제\s+주입률': '응집제주입률',
            r'슬러지\s+발생량': '슬러지발생량',
            
            # 기술/방법 관련
            r'수위\s+목표값\s+결정\s+방법': '수위목표값결정방법',
            r'회전속도\s+계산\s+방법': '회전속도계산방법',
            r'세척\s+주기\s+결정': '세척주기결정',
            r'운전\s+최적화\s+방법': '운전최적화방법',
            
            # 성능 지표 관련
            r'성능\s+지표': '성능지표',
            r'정확도\s+지표': '정확도지표',
            r'효율\s+지표': '효율지표',
            r'mae\s+평균\s+절대\s+오차': 'mae평균절대오차',
            r'mse\s+평균\s+제곱\s+오차': 'mse평균제곱오차',
            r'rmse\s+평균\s+제곱근\s+오차': 'rmse평균제곱근오차',
            r'r²\s+결정계수': 'r²결정계수',
            r'r2\s+결정계수': 'r2결정계수',
        }
        
        # 각 매핑 적용
        for pattern, replacement in compound_noun_mappings.items():
            question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
        
        return question
    
    def _classify_question_type(self, question: str) -> QuestionType:
        """질문 유형 분류"""
        for question_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    return question_type
        
        return QuestionType.UNKNOWN
    
    def _extract_keywords(self, question: str) -> List[str]:
        """키워드 추출 (복합명사 고려)"""
        keywords = []
        
        # 1. 복합명사 추출 (공백이 있는 복합명사도 포함)
        compound_keywords = self._extract_compound_keywords(question)
        keywords.extend(compound_keywords)
        
        # 2. 패턴 기반 키워드 추출
        for pattern in self.keyword_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            keywords.extend(matches)
        
        # 3. 단어 단위 키워드 추출 (기존 방식)
        word_keywords = self._extract_word_keywords(question)
        keywords.extend(word_keywords)
        
        # 중복 제거 및 정렬
        keywords = list(set(keywords))
        keywords.sort()
        
        return keywords
    
    def _extract_compound_keywords(self, question: str) -> List[str]:
        """복합명사 키워드 추출"""
        compound_keywords = []
        
        # 정수처리 도메인 특화 복합명사 패턴
        compound_patterns = [
            # 공정 관련 (공백 포함/미포함 모두)
            r'(착수\s*공정|착수공정)',
            r'(약품\s*공정|약품공정)',
            r'(혼화\s*응집\s*공정|혼화응집공정)',
            r'(혼화\s*공정|혼화공정)',
            r'(응집\s*공정|응집공정)',
            r'(침전\s*공정|침전공정)',
            r'(여과\s*공정|여과공정)',
            r'(소독\s*공정|소독공정)',
            r'(슬러지\s*처리\s*공정|슬러지처리공정)',
            r'(슬러지\s*공정|슬러지공정)',
            
            # 시스템 관련
            r'(ems\s*시스템|ems시스템)',
            r'(pms\s*시스템|pms시스템)',
            r'(스마트\s*정수장|스마트정수장)',
            r'(ai\s*모델|ai모델)',
            r'(머신러닝\s*모델|머신러닝모델)',
            r'(딥러닝\s*모델|딥러닝모델)',
            
            # 장비/설비 관련
            r'(교반기\s*회전속도|교반기회전속도)',
            r'(펌프\s*세부\s*현황|펌프세부현황)',
            r'(밸브\s*개도율|밸브개도율)',
            r'(수위\s*목표값|수위목표값)',
            r'(잔류염소\s*농도|잔류염소농도)',
            r'(원수\s*탁도|원수탁도)',
            r'(응집제\s*주입률|응집제주입률)',
            r'(슬러지\s*발생량|슬러지발생량)',
            
            # 성능 지표 관련
            r'(성능\s*지표|성능지표)',
            r'(정확도\s*지표|정확도지표)',
            r'(효율\s*지표|효율지표)',
            r'(mae|mse|rmse|r²|r2)',
        ]
        
        for pattern in compound_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # 그룹이 여러 개인 경우 첫 번째 유효한 것 선택
                    for m in match:
                        if m and len(m.strip()) > 1:
                            compound_keywords.append(m.strip())
                            break
                else:
                    if match and len(match.strip()) > 1:
                        compound_keywords.append(match.strip())
        
        return compound_keywords
    
    def _extract_word_keywords(self, question: str) -> List[str]:
        """단어 단위 키워드 추출"""
        # 단어 분할 (한글, 영어, 숫자)
        words = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', question)
        
        # 필터링
        valid_words = []
        for word in words:
            if len(word) >= 2 and not word.isdigit():
                valid_words.append(word.lower())
        
        return valid_words
    
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

    def _extract_keyword_pair(self, question: str) -> Dict[str, str]:
        """질문으로부터 [주체/대상], [행위/속성]에 해당하는 명사쌍을 경량 규칙으로 추출한다.
        - 우선순위 규칙:
          1) "에서" 패턴과 "무엇/어떻게" 계열 감지 → '에서' 이후 명사구를 주체로, 물음 패턴은 속성으로 표기
          2) 수량형(몇/개/분/비율 등) → 첫 명사를 대상, 속성은 수량/범위
          3) 백오프: 공백 토큰 중 한글/영문/숫자 포함 토큰 상위 2개를 사용
        """
        import re as _re
        sent = question.strip()
        # 토큰 분할(간단): 공백 기준
        tokens = [t for t in sent.split() if t]
        subject = ""
        attribute = ""

        # 1) "에서" + 질문어
        if "에서" in tokens and any(w in tokens for w in ["무엇", "어떻게", "왜"]):
            try:
                idx = tokens.index("에서")
                # 이후 1~2 토큰을 주체 후보로
                subj_tokens = tokens[idx+1:idx+3]
                subject = " ".join([t for t in subj_tokens if _re.search(r"[가-힣A-Za-z0-9]", t)])
                attribute = "행위/속성"
            except Exception:
                pass

        # 2) 수량형 질문
        if not subject:
            if _re.search(r"(몇|개|분|시간|비율|율|수치)", sent):
                # 첫 의미 토큰을 대상, 속성은 수량
                for t in tokens:
                    if _re.search(r"[가-힣A-Za-z0-9]", t):
                        subject = t
                        attribute = "수량/범위"
                        break

        # 3) 백오프: 상위 의미 토큰 2개
        if not subject:
            nouns = [t for t in tokens if _re.search(r"[가-힣A-Za-z0-9]", t)]
            if nouns:
                subject = nouns[0]
                attribute = nouns[1] if len(nouns) > 1 else (attribute or "속성")

        return {"subject": subject, "attribute": attribute}
    
    def _extract_answer_target(self, question: str) -> tuple[str, str, str, float]:
        """다층적 접근법으로 질문에서 답변 목표(Answer Target)를 추출한다.
        
        정수처리 도메인 특화 패턴을 추가하여 더 정확한 목표 추출을 수행합니다.
        
        Returns:
            tuple: (answer_target, target_type, value_intent, confidence_score)
            - answer_target: 구체적인 답변 목표 (예: "약품 주입량", "슬러지 발생량")
            - target_type: 목표 유형 (quantitative_value, qualitative_definition, procedural, comparative)
            - value_intent: 값의 의도 (definition, value, process, comparison)
            - confidence_score: 추출 신뢰도 (0.0 ~ 1.0)
        """
        question_lower = question.lower().strip()
        
        # 다층적 추출 결과들을 저장
        extraction_results = []
        
        # 1단계: 정수처리 도메인 특화 패턴 매칭 (최우선)
        domain_result = self._extract_target_with_domain_patterns(question_lower)
        if domain_result[0]:
            extraction_results.append((domain_result, 0.95))  # 매우 높은 신뢰도
        
        # 2단계: 의존성 파싱 기반 추출 (가장 정확)
        if DEPENDENCY_PARSING_AVAILABLE:
            dep_result = self._extract_target_with_dependency_parsing(question)
            if dep_result[0]:  # 결과가 있으면
                extraction_results.append((dep_result, 0.9))  # 높은 신뢰도
        
        # 3단계: 개선된 패턴 매칭
        pattern_result = self._extract_target_with_improved_patterns(question_lower)
        if pattern_result[0]:
            extraction_results.append((pattern_result, 0.7))  # 중간 신뢰도
        
        # 4단계: 의미 기반 분류
        semantic_result = self._extract_target_with_semantic_analysis(question)
        if semantic_result[0]:
            extraction_results.append((semantic_result, 0.6))  # 중간 신뢰도
        
        # 5단계: 기존 패턴 매칭 (백업)
        legacy_result = self._extract_target_with_legacy_patterns(question_lower)
        if legacy_result[0]:
            extraction_results.append((legacy_result, 0.5))  # 낮은 신뢰도
        
        # 5단계: 앙상블 방법으로 최종 결과 결정
        if extraction_results:
            final_result, final_confidence = self._ensemble_extraction_results(extraction_results)
            return final_result + (final_confidence,)
        
        # 모든 방법이 실패한 경우 기본 추출
        basic_target = self._extract_basic_target(question_lower)
        return basic_target, "general", "general", 0.1  # 낮은 신뢰도
    
    def _extract_target_with_domain_patterns(self, question_lower: str) -> tuple[str, str, str]:
        """정수처리 도메인 특화 패턴으로 Answer Target 추출"""
        # 정수처리 도메인 특화 패턴 정의
        domain_patterns = {
            # 정량적 값 질문 (수치, 성능 지표 등)
            "quantitative_value": {
                "patterns": [
                    # 성능 지표 관련
                    r'([가-힣\w\s]+)\s*(?:모델|알고리즘|시스템)\s*(?:의|에서)\s*(?:성능|지표|결과|정확도|효율)',
                    r'([가-힣\w\s]+)\s*(?:성능|지표|결과|정확도|효율)\s*(?:은|는|이|가)\s*(?:어떻게|얼마나)',
                    r'([가-힣\w\s]+)\s*(?:mae|mse|rmse|r²|r2|정확도|오차|성능)',
                    # 수치/값 관련
                    r'([가-힣\w\s]+)\s*(?:량|수치|값|비율|농도|속도|압력|온도|유량|주입률|개도율|효율)',
                    r'([가-힣\w\s]+)\s*(?:이|가)\s*(?:얼마나|몇)',
                    r'(?:얼마나|몇)\s*([가-힣\w\s]+)',
                ],
                "examples": [
                    "N-beats 모델의 성능", "약품 주입률", "슬러지 발생량", "회전속도", "잔류염소 농도"
                ]
            },
            # 정의/개념 질문 (무엇, 어떤, 목적 등)
            "qualitative_definition": {
                "patterns": [
                    # AI 모델 목표/기능 관련
                    r'([가-힣\w\s]+)\s*(?:공정|모델|시스템|ai|알고리즘)\s*(?:의|에서)\s*(?:목표|기능|역할|목적)',
                    r'([가-힣\w\s]+)\s*(?:이|가)\s*(?:예측하는|제어하는|관리하는|최적화하는)\s*(?:것은|것이)\s*(?:무엇|어떤)',
                    r'([가-힣\w\s]+)\s*(?:은|는)\s*(?:무엇|어떤)',
                    # 시스템/기능 관련
                    r'([가-힣\w\s]+)\s*(?:의|에서)\s*(?:주요\s*)?(?:기능|목적|역할|방법|원리|개념|정의)',
                ],
                "examples": [
                    "여과 공정에서 AI 모델의 목표", "EMS의 주요 기능", "PMS의 목적"
                ]
            },
            # 절차/과정 질문 (어떻게, 절차 등)
            "procedural": {
                "patterns": [
                    r'([가-힣\w\s]+)\s*(?:은|는)\s*(?:어떻게)\s*([가-힣\w\s]+)',
                    r'([가-힣\w\s]+)\s*(?:의|에서)\s*(?:절차|과정|단계|순서|방식|방법)',
                    r'(?:어떻게)\s*([가-힣\w\s]+)',
                ],
                "examples": [
                    "수위 목표값 결정 방법", "회전속도 계산 방법", "세척 주기 결정"
                ]
            },
            # 비교/관계 질문
            "comparative": {
                "patterns": [
                    r'([가-힣\w\s]+)\s*(?:과|와)\s*(?:가장|높은|낮은)\s*(?:상관관계|관계|영향)',
                    r'([가-힣\w\s]+)\s*(?:의|에서)\s*(?:영향|관계|상관관계)',
                    r'([가-힣\w\s]+)\s*(?:에서|의)\s*(?:가장|높은|낮은)\s*([가-힣\w\s]+)',
                ],
                "examples": [
                    "슬러지 발생량과 가장 높은 상관관계", "원수 탁도의 영향"
                ]
            },
            # 확인/가능성 질문
            "verification": {
                "patterns": [
                    r'([가-힣\w\s]+)\s*(?:에서|의)\s*(?:확인\s*가능|접근\s*가능|사용\s*가능|제공\s*가능)',
                    r'([가-힣\w\s]+)\s*(?:을|를)\s*(?:확인|접근|사용|제공)\s*(?:할\s*수|가능)',
                    r'([가-힣\w\s]+)\s*(?:은|는)\s*(?:가능|불가능|유효|무효)',
                ],
                "examples": [
                    "EMS에서 펌프 세부 현황 확인 가능", "PMS에서 확인 가능한 정보"
                ]
            }
        }
        
        # 각 패턴 타입별로 매칭 시도
        for target_type, config in domain_patterns.items():
            patterns = config["patterns"]
            
            for pattern in patterns:
                matches = re.findall(pattern, question_lower, re.IGNORECASE)
                if matches:
                    # 여러 그룹이 매칭된 경우 조합
                    if isinstance(matches[0], tuple):
                        best_match = " ".join([m for m in matches[0] if m]).strip()
                    else:
                        best_match = max(matches, key=len).strip()
                    
                    if best_match and len(best_match) > 1:
                        # 정수처리 도메인 특화 키워드 강화
                        enhanced_target = self._enhance_domain_target(best_match, target_type)
                        value_intent = self._determine_value_intent(target_type, question_lower)
                        return enhanced_target, target_type, value_intent
        
        return "", "", ""
    
    def _enhance_domain_target(self, target: str, target_type: str) -> str:
        """정수처리 도메인 특화 키워드로 목표를 강화"""
        # 도메인 특화 키워드 매핑
        domain_keywords = {
            "quantitative_value": {
                "성능": "모델 성능 지표",
                "지표": "성능 지표",
                "mae": "MAE (평균 절대 오차)",
                "mse": "MSE (평균 제곱 오차)", 
                "rmse": "RMSE (평균 제곱근 오차)",
                "r²": "R² (결정계수)",
                "r2": "R² (결정계수)",
                "주입률": "약품 주입률",
                "발생량": "슬러지 발생량",
                "회전속도": "교반기 회전속도",
                "농도": "잔류염소 농도",
                "개도율": "밸브 개도율"
            },
            "qualitative_definition": {
                "목표": "AI 모델의 목표",
                "기능": "시스템의 주요 기능",
                "역할": "시스템의 역할",
                "목적": "시스템의 목적"
            },
            "procedural": {
                "결정": "목표값 결정 방법",
                "계산": "수치 계산 방법",
                "최적화": "운전 최적화 방법"
            },
            "comparative": {
                "상관관계": "변수 간 상관관계",
                "영향": "변수의 영향도",
                "관계": "변수 간 관계"
            },
            "verification": {
                "확인": "정보 확인 가능성",
                "접근": "시스템 접근 가능성",
                "제공": "정보 제공 가능성"
            }
        }
        
        # 타입별 키워드로 강화
        if target_type in domain_keywords:
            for keyword, enhancement in domain_keywords[target_type].items():
                if keyword in target:
                    return enhancement
        
        return target
    
    def _extract_target_with_dependency_parsing(self, question: str) -> tuple[str, str, str]:
        """의존성 파싱을 사용한 Answer Target 추출"""
        if not DEPENDENCY_PARSING_AVAILABLE or not nlp:
            return "", "", ""
        
        try:
            doc = nlp(question)
            
            # 질문어(WH-word) 찾기
            wh_words = []
            for token in doc:
                if token.text.lower() in ["무엇", "어떤", "얼마나", "몇", "어떻게", "왜", "언제", "어디", "누가"]:
                    wh_words.append(token)
            
            if not wh_words:
                return "", "", ""
            
            # 가장 중요한 질문어 선택 (보통 첫 번째)
            main_wh = wh_words[0]
            
            # 질문어와 관련된 명사구 추출
            target_phrases = []
            for token in doc:
                # 질문어의 의존 관계를 따라가며 명사구 추출
                if self._is_related_to_wh_word(token, main_wh, doc):
                    phrase = self._extract_noun_phrase(token, doc)
                    if phrase and len(phrase) > 1:
                        target_phrases.append(phrase)
            
            if target_phrases:
                # 가장 긴 명사구 선택 (더 구체적)
                best_phrase = max(target_phrases, key=len)
                target_type = self._classify_target_type_by_wh_word(main_wh.text.lower())
                value_intent = self._determine_value_intent(target_type, question)
                return best_phrase, target_type, value_intent
            
            return "", "", ""
        except Exception as e:
            logger.warning(f"의존성 파싱 실패: {e}")
            return "", "", ""
    
    def _is_related_to_wh_word(self, token, wh_word, doc) -> bool:
        """토큰이 질문어와 관련이 있는지 확인"""
        # 직접적인 의존 관계
        if token.head == wh_word or wh_word.head == token:
            return True
        
        # 간접적인 의존 관계 (2단계까지)
        for dep_token in doc:
            if (dep_token.head == wh_word and token.head == dep_token) or \
               (dep_token.head == token and wh_word.head == dep_token):
                return True
        
        return False
    
    def _extract_noun_phrase(self, token, doc) -> str:
        """토큰을 중심으로 명사구 추출"""
        phrase_tokens = [token]
        
        # 의존하는 토큰들 추가
        for child in token.children:
            if child.dep_ in ["det", "amod", "compound", "nmod"]:
                phrase_tokens.append(child)
        
        # 정렬하여 원래 순서 유지
        phrase_tokens.sort(key=lambda x: x.i)
        return " ".join([t.text for t in phrase_tokens])
    
    def _classify_target_type_by_wh_word(self, wh_word: str) -> str:
        """질문어에 따른 목표 타입 분류"""
        if wh_word in ["얼마나", "몇"]:
            return "quantitative_value"
        elif wh_word in ["무엇", "어떤"]:
            return "qualitative_definition"
        elif wh_word in ["어떻게"]:
            return "procedural"
        elif wh_word in ["왜"]:
            return "comparative"
        else:
            return "general"
    
    def _extract_target_with_improved_patterns(self, question_lower: str) -> tuple[str, str, str]:
        """개선된 패턴 매칭으로 Answer Target 추출"""
        # 더 정교한 패턴들
        improved_patterns = {
            "quantitative_value": [
                # "X가 얼마나 Y인가요?" 패턴
                r'([가-힣\w\s]+(?:량|수치|값|비율|농도|속도|압력|온도|유량|주입률|개도율|효율|성능|지표))\s*(?:이|가)\s*(?:얼마나|몇)',
                # "얼마나 X인가요?" 패턴
                r'(?:얼마나|몇)\s*([가-힣\w\s]+(?:량|수치|값|비율|농도|속도|압력|온도|유량|주입률|개도율|효율|성능|지표))',
                # "X의 Y는 얼마나?" 패턴
                r'([가-힣\w\s]+)\s*(?:의|에서)\s*([가-힣\w\s]+(?:량|수치|값|비율|농도|속도|압력|온도|유량|주입률|개도율|효율|성능|지표))\s*(?:이|가)\s*(?:얼마나|몇)',
            ],
            "qualitative_definition": [
                # "X는 무엇인가요?" 패턴
                r'([가-힣\w\s]+)\s*(?:은|는)\s*(?:무엇|어떤)',
                # "X의 Y는 무엇인가요?" 패턴
                r'([가-힣\w\s]+)\s*(?:의|에서)\s*([가-힣\w\s]+)\s*(?:은|는)\s*(?:무엇|어떤)',
                # "X가 Y하는 것은 무엇인가요?" 패턴
                r'([가-힣\w\s]+)\s*(?:이|가)\s*([가-힣\w\s]+)\s*(?:하는|하는)\s*(?:것은|것이)\s*(?:무엇|어떤)',
            ],
            "procedural": [
                # "X는 어떻게 Y하나요?" 패턴
                r'([가-힣\w\s]+)\s*(?:은|는)\s*(?:어떻게)\s*([가-힣\w\s]+)',
                # "X의 Y는 어떻게 결정되나요?" 패턴
                r'([가-힣\w\s]+)\s*(?:의|에서)\s*([가-힣\w\s]+)\s*(?:은|는)\s*(?:어떻게)\s*(?:결정|처리|관리)',
            ],
            "comparative": [
                # "X와 Y의 관계는?" 패턴
                r'([가-힣\w\s]+)\s*(?:과|와)\s*([가-힣\w\s]+)\s*(?:의|에서)\s*(?:관계|상관관계|영향)',
                # "X에서 가장 Y한 것은?" 패턴
                r'([가-힣\w\s]+)\s*(?:에서|의)\s*(?:가장|높은|낮은)\s*([가-힣\w\s]+)',
            ]
        }
        
        for target_type, patterns in improved_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, question_lower, re.IGNORECASE)
                if matches:
                    # 여러 그룹이 매칭된 경우 조합
                    if isinstance(matches[0], tuple):
                        best_match = " ".join([m for m in matches[0] if m]).strip()
                    else:
                        best_match = max(matches, key=len).strip()
                    
                    if best_match and len(best_match) > 1:
                        value_intent = self._determine_value_intent(target_type, question_lower)
                        return best_match, target_type, value_intent
        
        return "", "", ""
    
    def _extract_target_with_semantic_analysis(self, question: str) -> tuple[str, str, str]:
        """의미 기반 분석으로 Answer Target 추출"""
        if not self.embedding_model:
            return "", "", ""
        
        try:
            # 질문을 임베딩으로 변환
            question_embedding = self.embedding_model.encode(question)
            
            # 미리 정의된 질문 템플릿들과 비교
            templates = {
                "quantitative_value": [
                    "얼마나 많은 양이 필요한가요?",
                    "수치는 얼마나 되나요?",
                    "비율은 어떻게 되나요?",
                    "농도는 얼마나 되나요?",
                ],
                "qualitative_definition": [
                    "무엇인가요?",
                    "어떤 것인가요?",
                    "정의는 무엇인가요?",
                    "목적은 무엇인가요?",
                ],
                "procedural": [
                    "어떻게 처리하나요?",
                    "어떻게 결정되나요?",
                    "과정은 어떻게 되나요?",
                    "절차는 어떻게 되나요?",
                ],
                "comparative": [
                    "관계는 어떻게 되나요?",
                    "영향은 무엇인가요?",
                    "비교하면 어떻게 되나요?",
                    "상관관계는 어떻게 되나요?",
                ]
            }
            
            best_similarity = 0
            best_type = "general"
            
            for target_type, template_list in templates.items():
                for template in template_list:
                    template_embedding = self.embedding_model.encode(template)
                    similarity = np.dot(question_embedding, template_embedding) / (
                        np.linalg.norm(question_embedding) * np.linalg.norm(template_embedding)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_type = target_type
            
            # 유사도가 임계값 이상이면 의미 기반 추출 시도
            if best_similarity > 0.3:
                # 질문에서 핵심 명사구 추출
                target = self._extract_semantic_target(question, best_type)
                if target:
                    value_intent = self._determine_value_intent(best_type, question)
                    return target, best_type, value_intent
            
            return "", "", ""
        except Exception as e:
            logger.warning(f"의미 기반 분석 실패: {e}")
            return "", "", ""
    
    def _extract_semantic_target(self, question: str, target_type: str) -> str:
        """의미 기반으로 목표 추출"""
        # 질문에서 핵심 명사구 추출
        words = question.split()
        
        # 타입별로 다른 추출 전략
        if target_type == "quantitative_value":
            # 수치 관련 단어 주변 추출
            for i, word in enumerate(words):
                if any(indicator in word for indicator in ["량", "수치", "값", "비율", "농도", "속도", "압력", "온도", "유량", "주입률", "개도율", "효율", "성능", "지표"]):
                    # 앞뒤 2-3개 단어 추출
                    start = max(0, i-2)
                    end = min(len(words), i+3)
                    return " ".join(words[start:end])
        
        elif target_type == "qualitative_definition":
            # "무엇", "어떤" 주변 추출
            for i, word in enumerate(words):
                if word in ["무엇", "어떤"]:
                    # 앞 2-3개 단어 추출
                    start = max(0, i-3)
                    end = i
                    return " ".join(words[start:end])
        
        # 기본 추출
        return " ".join(words[:3]) if len(words) >= 3 else " ".join(words)
    
    def _extract_target_with_legacy_patterns(self, question_lower: str) -> tuple[str, str, str]:
        """기존 패턴 매칭 방법 (백업용)"""
        # 기존 로직을 그대로 사용
        for target_type, config in self.answer_target_patterns.items():
            patterns = config["patterns"]
            value_indicators = config["value_indicators"]
            
            # 패턴 기반 매칭
            for pattern in patterns:
                matches = re.findall(pattern, question_lower, re.IGNORECASE)
                if matches:
                    best_match = max(matches, key=len).strip()
                    if best_match and len(best_match) > 1:
                        value_intent = self._determine_value_intent(target_type, question_lower)
                        return best_match, target_type, value_intent
            
            # 값 지시어 기반 매칭
            for indicator in value_indicators:
                if indicator in question_lower:
                    target = self._extract_target_around_indicator(question_lower, indicator)
                    if target:
                        value_intent = self._determine_value_intent(target_type, question_lower)
                        return target, target_type, value_intent
        
        return "", "", ""
    
    def _ensemble_extraction_results(self, extraction_results: List[Tuple[Tuple[str, str, str], float]]) -> tuple[tuple[str, str, str], float]:
        """앙상블 방법으로 최종 결과 결정"""
        if not extraction_results:
            return (("", "", ""), 0.0)
        
        # 신뢰도 순으로 정렬
        extraction_results.sort(key=lambda x: x[1], reverse=True)
        
        # 가장 높은 신뢰도의 결과를 기본으로 사용
        best_result, best_confidence = extraction_results[0]
        best_target, best_type, best_intent = best_result
        
        # 다른 결과들과 일치성 확인
        consensus_count = 1
        total_confidence = best_confidence
        
        for (target, target_type, intent), confidence in extraction_results[1:]:
            # 목표가 유사하면 일치로 간주
            if self._is_similar_target(best_target, target):
                consensus_count += 1
                total_confidence += confidence
                # 더 구체적인 목표로 업데이트
                if len(target) > len(best_target):
                    best_target = target
                    best_type = target_type
                    best_intent = intent
        
        # 일치도가 높으면 신뢰도 증가
        if consensus_count >= 2:
            logger.info(f"앙상블 일치: {consensus_count}개 방법이 유사한 결과 도출")
            # 일치하는 방법들의 평균 신뢰도 계산
            final_confidence = min(0.95, total_confidence / consensus_count + 0.1)
        else:
            final_confidence = best_confidence
        
        return ((best_target, best_type, best_intent), final_confidence)
    
    def _is_similar_target(self, target1: str, target2: str) -> bool:
        """두 목표가 유사한지 확인"""
        if not target1 or not target2:
            return False
        
        # 단어 단위로 비교
        words1 = set(target1.lower().split())
        words2 = set(target2.lower().split())
        
        # 교집합 비율 계산
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return False
        
        similarity = len(intersection) / len(union)
        return similarity > 0.5  # 50% 이상 일치하면 유사하다고 판단
    
    def _determine_value_intent(self, target_type: str, question: str) -> str:
        """값의 의도를 결정한다."""
        if target_type == "quantitative_value":
            return "value"
        elif target_type == "qualitative_definition":
            return "definition"
        elif target_type == "procedural":
            return "process"
        elif target_type == "comparative":
            return "comparison"
        else:
            return "general"
    
    def _extract_target_around_indicator(self, question: str, indicator: str) -> str:
        """지시어 주변에서 목표를 추출한다."""
        try:
            # 지시어 위치 찾기
            indicator_pos = question.find(indicator)
            if indicator_pos == -1:
                return ""
            
            # 지시어 앞뒤로 문맥 추출
            start = max(0, indicator_pos - 15)
            end = min(len(question), indicator_pos + len(indicator) + 15)
            context = question[start:end].strip()
            
            # 불필요한 단어 제거
            context = re.sub(r'\b(은|는|이|가|을|를|의|에|에서|로|으로|와|과|도|만|부터|까지|인가요|되나요|입니까)\b', ' ', context)
            context = re.sub(r'\s+', ' ', context).strip()
            
            # 핵심 명사구 추출 (2-4개 단어)
            words = context.split()
            if len(words) >= 2:
                # 지시어 앞의 명사구 우선 추출
                target_words = []
                for word in words:
                    if word != indicator and len(word) > 1:
                        target_words.append(word)
                        if len(target_words) >= 3:  # 최대 3개 단어
                            break
                return ' '.join(target_words) if target_words else ""
            
            return context if len(context) > 1 else ""
        except Exception:
            return ""
    
    def _extract_context_around_keyword(self, question: str, keyword: str) -> str:
        """키워드 주변의 문맥을 추출한다."""
        try:
            # 키워드 위치 찾기
            keyword_pos = question.find(keyword)
            if keyword_pos == -1:
                return ""
            
            # 키워드 앞뒤로 최대 10글자씩 추출
            start = max(0, keyword_pos - 10)
            end = min(len(question), keyword_pos + len(keyword) + 10)
            context = question[start:end].strip()
            
            # 불필요한 단어 제거
            context = re.sub(r'\b(은|는|이|가|을|를|의|에|에서|로|으로|와|과|도|만|부터|까지)\b', ' ', context)
            context = re.sub(r'\s+', ' ', context).strip()
            
            return context if len(context) > 1 else ""
        except Exception:
            return ""
    
    def _extract_basic_target(self, question: str) -> str:
        """기본적인 답변 목표를 추출한다."""
        # 질문에서 핵심 명사 추출
        # "무엇", "어떻게", "얼마나" 등의 질문어 제거
        cleaned = re.sub(r'\b(무엇|어떻게|얼마나|몇|어디|언제|누가|왜|어떤)\b', '', question)
        
        # 조사 제거
        cleaned = re.sub(r'\b(은|는|이|가|을|를|의|에|에서|로|으로|와|과|도|만|부터|까지)\b', ' ', cleaned)
        
        # 연속된 공백 정리
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 첫 번째 의미있는 단어/구 추출
        words = cleaned.split()
        if words:
            # 첫 2-3개 단어를 조합하여 목표로 사용
            target = ' '.join(words[:min(3, len(words))])
            return target if len(target) > 1 else "일반적인 정보"
        
        return "일반적인 정보"
    
    def add_conversation_item(self, question: str, answer: str, confidence_score: float = 0.0, metadata: Optional[Dict] = None):
        """대화 항목 추가"""
        item = ConversationItem(
            question=question,
            answer=answer,
            timestamp=datetime.now(),
            confidence_score=confidence_score,
            metadata=metadata or {}
        )
        self.conversation_history.append(item)
        
        # 대화 히스토리 크기 제한 (최근 10개만 유지)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def clear_conversation_history(self):
        """대화 히스토리 초기화"""
        self.conversation_history.clear()
        logger.info("대화 히스토리 초기화 완료")
