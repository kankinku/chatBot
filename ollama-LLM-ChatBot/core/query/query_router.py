"""
쿼리 라우터 - SBERT 기반 라우팅

SBERT는 라우팅용으로만 사용하여 질문을 적절한 처리 파이프라인으로 분기
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import os

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logging.warning("sentence-transformers 라이브러리를 찾을 수 없습니다.")

from core.config.pipeline_config_manager import PipelineConfigManager

logger = logging.getLogger(__name__)

class QueryRoute(Enum):
    """쿼리 라우팅 경로"""
    PDF_SEARCH = "pdf_search"      # PDF 문서 검색
    SQL_QUERY = "sql_query"        # SQL 데이터베이스 쿼리
    GREETING = "greeting"          # 인사말
    UNKNOWN = "unknown"            # 알 수 없음

@dataclass
class RouteResult:
    """라우팅 결과"""
    route: QueryRoute
    confidence: float
    reasoning: str
    metadata: Optional[Dict] = None

class QueryRouter:
    """
    쿼리 라우터 - SBERT 기반 빠른 라우팅
    
    기능:
    1. 질문을 적절한 처리 파이프라인으로 라우팅
    2. PDF 검색 vs SQL 쿼리 구분
    3. 빠른 의사결정을 위한 임베딩 기반 유사도 계산
    """
    
    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask"):
        """쿼리 라우터 초기화"""
        
        self.embedding_model = None
        if SBERT_AVAILABLE:
            try:
                # 오프라인 로딩 강제: 모델이 없으면 예외 발생
                self.embedding_model = SentenceTransformer(embedding_model, cache_folder="./models")
                logger.info(f"✅ 라우팅용 SBERT 모델 로드: {embedding_model}")
            except Exception as e:
                logger.error(f"SBERT 모델 로드 실패(오프라인 필요): {e}")
                raise
        
        # 환경 변수에서 회사/프로젝트 정보 읽기
        self.company_name = os.getenv('COMPANY_NAME', '범용 RAG 시스템')
        self.project_name = os.getenv('PROJECT_NAME', '범용 RAG 시스템')
        self.system_type = os.getenv('SYSTEM_TYPE', '범용 RAG 시스템')
        
        logger.info(f"쿼리 라우터 초기화: {self.company_name} - {self.project_name}")
        
        # 파이프라인 설정 관리자 초기화
        self.pipeline_config_manager = PipelineConfigManager()
        
        # 라우팅을 위한 참조 질문들 (JSON 설정에서 로드)
        self.reference_questions = {}
        self._load_reference_questions_from_config()
        
        # 참조 질문들의 임베딩 미리 계산
        self.reference_embeddings = {}
        if self.embedding_model:
            self._precompute_embeddings()
    
    def _load_reference_questions_from_config(self):
        """JSON 설정에서 참조 질문 로드"""
        try:
            # 각 파이프라인별로 참조 질문 로드
            pipeline_mapping = {
                "SQL_QUERY": QueryRoute.SQL_QUERY,
                "PDF_SEARCH": QueryRoute.PDF_SEARCH,
                "GREETING": QueryRoute.GREETING
            }
            
            for pipeline_name, route in pipeline_mapping.items():
                reference_questions = self.pipeline_config_manager.get_pipeline_reference_questions(pipeline_name)
                if reference_questions:
                    self.reference_questions[route] = reference_questions
                    logger.info(f"참조 질문 로드 완료: {pipeline_name} ({len(reference_questions)}개)")
                else:
                    logger.warning(f"참조 질문을 찾을 수 없습니다: {pipeline_name}")
            
            logger.info(f"총 {len(self.reference_questions)}개 파이프라인의 참조 질문 로드 완료")
            
        except Exception as e:
            logger.error(f"참조 질문 로드 실패: {e}")
            # 폴백: 기본 참조 질문 사용
            self._load_fallback_reference_questions()
    
    def _load_fallback_reference_questions(self):
        """폴백 참조 질문 로드"""
        self.reference_questions = {
            QueryRoute.SQL_QUERY: [
                "몇 개인가요?", "개수는?", "총 몇 개?", "평균은?", "최대값은?", "최소값은?",
                "상위 10개는?", "하위 5개는?", "비율은?", "순위는?", "통계를 보여주세요"
            ],
            QueryRoute.PDF_SEARCH: [
                "어떻게 사용하나요?", "사용법은?", "설정 방법", "운용 방법", "조작법",
                "메뉴얼", "가이드", "설명서", "무엇인가요?", "정의는?", "개념은?"
            ],
            QueryRoute.GREETING: [
                "안녕하세요", "반갑습니다", "안녕", "하이", "hi", "hello",
                "도움이 필요해요", "처음 사용해요", "질문이 있어요"
            ]
        }
        logger.info("폴백 참조 질문 로드 완료")
    
    def _precompute_embeddings(self):
        """참조 질문들의 임베딩 미리 계산"""
        try:
            for route, questions in self.reference_questions.items():
                embeddings = self.embedding_model.encode(questions)
                self.reference_embeddings[route] = embeddings
                logger.debug(f"임베딩 계산 완료: {route.value} ({len(questions)}개)")
            
            logger.info("참조 질문 임베딩 사전 계산 완료")
        except Exception as e:
            logger.error(f"임베딩 사전 계산 실패: {e}")
    
    def route_query(self, question: str) -> RouteResult:
        """
        질문을 적절한 파이프라인으로 라우팅 (범용 RAG 시스템용)
        
        Args:
            question: 사용자 질문
            
        Returns:
            라우팅 결과
        """
        # 우선 규칙: 특정 키워드가 있으면 즉시 분류
        question_lower = question.lower()
        
        # JSON 설정에서 키워드 로드
        sql_keywords = self.pipeline_config_manager.get_pipeline_keywords("SQL_QUERY")
        pdf_keywords = self.pipeline_config_manager.get_pipeline_keywords("PDF_SEARCH")
        # 인사는 명확한 인사 표현만 허용
        greeting_keywords = [
            "안녕", "안녕하세요", "안녕하십니까", "반갑습니다", "반가워", "하이", "hi", "hello"
        ]
        
        # 특별한 우선순위 규칙: "방법" 키워드가 있으면 PDF 검색으로 라우팅
        if '방법' in question_lower:
            return RouteResult(
                route=QueryRoute.PDF_SEARCH,
                confidence=0.95,
                reasoning="우선순위 규칙: '방법' 키워드 감지로 PDF 검색 라우팅"
            )
        
        # "사건" 키워드가 있으면 PDF 검색으로 라우팅
        if '사건' in question_lower:
            return RouteResult(
                route=QueryRoute.PDF_SEARCH,
                confidence=0.95,
                reasoning="우선순위 규칙: '사건' 키워드 감지로 PDF 검색 라우팅"
            )
        
        # "목록" 키워드가 있으면 PDF 검색으로 라우팅
        if '목록' in question_lower:
            return RouteResult(
                route=QueryRoute.PDF_SEARCH,
                confidence=0.95,
                reasoning="우선순위 규칙: '목록' 키워드 감지로 PDF 검색 라우팅"
            )
        
        # 키워드 기반 우선 분류
        sql_score = sum(1 for keyword in sql_keywords if keyword in question_lower)
        pdf_score = sum(1 for keyword in pdf_keywords if keyword in question_lower)
        greeting_score = sum(1 for keyword in greeting_keywords if keyword in question_lower)
        
        # 인사말이 가장 높은 경우
        if greeting_score > 0 and greeting_score >= max(sql_score, pdf_score):
            return RouteResult(
                route=QueryRoute.GREETING,
                confidence=0.9,
                reasoning=f"키워드 기반: 인사말 패턴 감지 (점수: {greeting_score})"
            )
        
        # 정량적 데이터 질문이 강한 경우
        if sql_score > pdf_score and sql_score > 1:
            return RouteResult(
                route=QueryRoute.SQL_QUERY,
                confidence=0.85,
                reasoning=f"키워드 기반: 정량적 데이터 질문 감지 (SQL: {sql_score} vs PDF: {pdf_score})"
            )
        
        # 기능/법률 질문이 강한 경우
        if pdf_score > sql_score and pdf_score > 1:
            return RouteResult(
                route=QueryRoute.PDF_SEARCH,
                confidence=0.85,
                reasoning=f"키워드 기반: 기능/법률 질문 감지 (PDF: {pdf_score} vs SQL: {sql_score})"
            )
        
        # SBERT 기반 분류 (키워드가 명확하지 않은 경우)
        if not self.embedding_model or not self.reference_embeddings:
            # SBERT를 사용할 수 없는 경우 기본값
            return RouteResult(
                route=QueryRoute.PDF_SEARCH,
                confidence=0.5,
                reasoning="SBERT 사용 불가: 기본값으로 PDF 검색 선택"
            )
        
        try:
            # 질문 임베딩 생성
            query_embedding = self.embedding_model.encode([question])[0]
            
            # 각 라우트와의 유사도 계산
            route_scores = {}
            
            for route, ref_embeddings in self.reference_embeddings.items():
                # 코사인 유사도 계산
                similarities = np.dot(ref_embeddings, query_embedding) / (
                    np.linalg.norm(ref_embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                
                # 최대 유사도 사용
                max_similarity = float(np.max(similarities))
                route_scores[route] = max_similarity
                
                logger.debug(f"라우트 {route.value}: 최대 유사도 {max_similarity:.3f}")
            
            # 최고 점수 라우트 선택
            best_route = max(route_scores, key=route_scores.get)
            best_score = route_scores[best_route]
            
            # 신뢰도 임계값 검사
            if best_score < 0.3:
                return RouteResult(
                    route=QueryRoute.PDF_SEARCH,  # 기본값을 PDF 검색으로 변경
                    confidence=best_score,
                    reasoning=f"모든 라우트의 유사도가 낮음 (최대: {best_score:.3f}), 기본값으로 PDF 검색 선택"
                )
            
            # SQL vs PDF 구분 로직
            sql_score = route_scores.get(QueryRoute.SQL_QUERY, 0.0)
            pdf_score = route_scores.get(QueryRoute.PDF_SEARCH, 0.0)
            greeting_score = route_scores.get(QueryRoute.GREETING, 0.0)
            
            # 인사말이 가장 높은 경우
            if greeting_score > 0.5 and greeting_score > max(sql_score, pdf_score):
                return RouteResult(
                    route=QueryRoute.GREETING,
                    confidence=greeting_score,
                    reasoning=f"SBERT 기반: 인사말로 분류 (유사도: {greeting_score:.3f})",
                    metadata={"scores": route_scores}
                )
            
            # SQL과 PDF 중 선택 (임계값 조정)
            if sql_score > pdf_score and sql_score > 0.35:
                return RouteResult(
                    route=QueryRoute.SQL_QUERY,
                    confidence=sql_score,
                    reasoning=f"SBERT 기반: SQL 쿼리로 분류 (SQL: {sql_score:.3f} vs PDF: {pdf_score:.3f})",
                    metadata={"scores": route_scores}
                )
            elif pdf_score > 0.35:
                return RouteResult(
                    route=QueryRoute.PDF_SEARCH,
                    confidence=pdf_score,
                    reasoning=f"SBERT 기반: PDF 검색으로 분류 (PDF: {pdf_score:.3f} vs SQL: {sql_score:.3f})",
                    metadata={"scores": route_scores}
                )
            else:
                # 둘 다 낮은 경우 기본값 (PDF 검색)
                return RouteResult(
                    route=QueryRoute.PDF_SEARCH,
                    confidence=max(sql_score, pdf_score),
                    reasoning=f"SBERT 기반: 기본 라우트 (PDF 검색) 선택",
                    metadata={"scores": route_scores}
                )
                
        except Exception as e:
            logger.error(f"SBERT 라우팅 실패: {e}")
            return self._rule_based_routing(question)
    
    def _rule_based_routing(self, question: str) -> RouteResult:
        """
        규칙 기반 라우팅 (SBERT 폴백)
        
        Args:
            question: 사용자 질문
            
        Returns:
            라우팅 결과
        """
        question_lower = question.lower()
        
        # 인사말 패턴 (명확한 인사만)
        greeting_patterns = [
            '안녕', '안녕하세요', '안녕하십니까', '반갑습니다', '반가워', '하이', 'hi', 'hello'
        ]
        
        # 인사말 패턴 매칭 (더 정확한 매칭)
        for pattern in greeting_patterns:
            if pattern in question_lower:
                return RouteResult(
                    route=QueryRoute.GREETING,
                    confidence=0.9,
                    reasoning=f"규칙 기반: 인사말 패턴 매칭 ('{pattern}')"
                )
        
        # "방법" 키워드가 있으면 우선적으로 PDF 검색으로 라우팅
        if '방법' in question_lower:
            return RouteResult(
                route=QueryRoute.PDF_SEARCH,
                confidence=0.9,
                reasoning="규칙 기반: '방법' 키워드 감지 - PDF 검색으로 라우팅"
            )
        
        # SQL 쿼리 패턴 (정량적 데이터 질문)
        sql_patterns = [
            '몇', '개수', '건수', '총', '평균', '최대', '최소', '상위', '하위',
            '비율', '순위', '통계', '분석', '수치', '데이터', '집계', '합계',
            '누적', '누계', '분포', '추세', '변화량', '증감률', '증감 폭',
            '대비', '비교', '기간별', '월별', '분기별', '연도별', '지역별',
            '구별', '카테고리별', '유형별', '얼마나', '어느 정도'
        ]
        
        sql_score = sum(1 for pattern in sql_patterns if pattern in question_lower)
        
        # PDF 검색 패턴 (기능/법률 질문)
        pdf_patterns = [
            '어떻게', '사용법', '설정', '운용', '조작', '메뉴얼', '가이드',
            '설명서', '설치', '운영', '관리', '점검', '유지보수', '트러블슈팅',
            '문제 해결', '오류 해결', '장애 대응', '복구', '백업', '업데이트',
            '연동', '연결', '통합', '연계', 'API', '인터페이스', '기능',
            '특징', '장점', '단점', '제한사항', '주의사항', '권장사항',
            '법률', '규정', '법규', '법령', '조례', '규칙', '지침', '정책',
            '방침', '기준', '표준', '규격', '사양', '요구사항', '규제',
            '제재', '처벌', '벌칙', '과태료', '행정처분', '허가', '인가',
            '승인', '등록', '신고', '신청', '제출', '의무', '책임', '면책',
            '배상', '손해배상', '책임보험', '개인정보', '보안', '저작권',
            '지적재산권', '특허', '상표', '라이선스', '이용약관', '무엇',
            '정의', '개념', '의미', '내용', '원리', '구조', '구성', '아키텍처',
            '설계', '개발', '구현', '제작', '제조', '생산', '공급', '유통',
            '판매', '마케팅', '홍보', '광고', '브랜딩', '브랜드', '품질',
            '품질관리', '품질보증', '품질검사', '품질평가', '인증', '검증',
            '평가', '모니터링', '감시', '관찰', '점검', '검사', '검토',
            '리포트', '보고서', '문서', '자료', '참고자료', '참고문헌',
            '백서', '화이트페이퍼', '기술문서', '기술자료'
        ]
        
        pdf_score = sum(1 for pattern in pdf_patterns if pattern in question_lower)
        
        # 점수 기반 라우팅
        if sql_score > pdf_score and sql_score > 0:
            confidence = min(0.7 + sql_score * 0.1, 1.0)
            return RouteResult(
                route=QueryRoute.SQL_QUERY,
                confidence=confidence,
                reasoning=f"규칙 기반: 정량적 데이터 질문 패턴 매칭 (점수: {sql_score})"
            )
        elif pdf_score > 0:
            confidence = min(0.7 + pdf_score * 0.1, 1.0)
            return RouteResult(
                route=QueryRoute.PDF_SEARCH,
                confidence=confidence,
                reasoning=f"규칙 기반: 기능/법률 질문 패턴 매칭 (점수: {pdf_score})"
            )
        else:
            # 패턴이 없는 경우 기본값 (PDF 검색)
            return RouteResult(
                route=QueryRoute.PDF_SEARCH,
                confidence=0.5,
                reasoning="규칙 기반: 패턴 없음, 기본값으로 PDF 검색 선택"
            )
    
    def add_reference_question(self, route: QueryRoute, question: str):
        """참조 질문 추가"""
        if route not in self.reference_questions:
            self.reference_questions[route] = []
        
        self.reference_questions[route].append(question)
        
        # 임베딩 다시 계산
        if self.embedding_model:
            self._precompute_embeddings()
        
        logger.info(f"참조 질문 추가: {route.value} - {question}")
    
    def get_route_statistics(self) -> Dict[str, int]:
        """라우트별 참조 질문 통계"""
        return {
            route.value: len(questions) 
            for route, questions in self.reference_questions.items()
        }

if __name__ == "__main__":
    # 테스트 코드
    router = QueryRouter()
    
    test_questions = [
        "안녕하세요",
        "강남구 교차로가 몇 개인가요?",
        "IFRO 시스템이 무엇인가요?",
        "평균 교통량은 얼마인가요?",
        "시스템 사용 방법을 알려주세요",
        "교통사고가 가장 많은 지역은?"
    ]
    
    for question in test_questions:
        result = router.route_query(question)
        print(f"\n질문: {question}")
        print(f"라우트: {result.route.value}")
        print(f"신뢰도: {result.confidence:.3f}")
        print(f"이유: {result.reasoning}")
