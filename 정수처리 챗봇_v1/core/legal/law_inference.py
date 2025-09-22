"""
법률 유추 모듈

사용자의 모호한 질문에 대해 적절한 법률을 유추하고,
검증 루프를 통해 신뢰도 있는 법률 후보를 제안합니다.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np

from .legal_schema import LegalChunk, LegalNormalizer
from .legal_retriever import LegalRetriever, SearchResult
from .legal_reranker import LegalReranker, RerankResult

import logging
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """유추 설정"""
    max_candidates: int = 5           # 최대 법률 후보 수
    min_confidence: float = 0.6      # 최소 신뢰도
    verification_threshold: float = 0.7  # 검증 임계값
    domain_weight: float = 0.3       # 도메인 분류 가중치
    keyword_weight: float = 0.2      # 키워드 매칭 가중치
    semantic_weight: float = 0.5     # 의미 유사도 가중치

@dataclass
class LawCandidate:
    """법률 후보"""
    law_id: str                    # 법률 ID
    law_title: str                 # 법률명
    confidence: float              # 신뢰도
    reasoning: str                 # 추론 근거
    evidence_chunks: List[LegalChunk]  # 근거 청크
    domains: List[str]             # 관련 도메인
    keywords: List[str]            # 매칭된 키워드
    verification_score: float      # 검증 점수

@dataclass
class InferenceResult:
    """유추 결과"""
    query: str                     # 원본 질의
    candidates: List[LawCandidate] # 법률 후보들
    confidence: float              # 전체 신뢰도
    needs_clarification: bool      # 추가 질문 필요 여부
    clarification_questions: List[str]  # 명확화 질문들
    metadata: Dict[str, Any]       # 메타데이터

class LawInference:
    """법률 유추 클래스"""
    
    def __init__(self, 
                 retriever: LegalRetriever,
                 reranker: Optional[LegalReranker] = None,
                 config: Optional[InferenceConfig] = None,
                 embedding_model: Optional[SentenceTransformer] = None):
        """유추기 초기화"""
        self.retriever = retriever
        self.reranker = reranker
        self.config = config or InferenceConfig()
        self.normalizer = LegalNormalizer()
        
        # 임베딩 모델
        self.embedding_model = embedding_model or retriever.embedding_model
        
        # 도메인 분류를 위한 키워드 확장
        self._initialize_domain_knowledge()
        
        # 질의 패턴 분석기
        self._initialize_query_patterns()
        
        logger.info("법률 유추기 초기화 완료")
    
    def _initialize_domain_knowledge(self):
        """도메인 지식 초기화 (공통 normalizer 사용)"""
        # LegalNormalizer의 확장된 도메인 키워드 사용
        self.domain_keywords = self.normalizer.domain_keywords
        
        # LegalNormalizer의 확장된 별칭 사전 사용
        self.law_aliases = {}
        for full_name, alias in self.normalizer.aliases_dict.items():
            if alias not in self.law_aliases:
                self.law_aliases[alias] = []
            self.law_aliases[alias].append(full_name)
    
    def _initialize_query_patterns(self):
        """질의 패턴 분석기 초기화"""
        self.query_patterns = {
            'definition': [
                r'(.+)(?:란|이란|는|은)\s*무엇',
                r'(.+)의?\s*정의',
                r'(.+)의?\s*의미',
                r'(.+)(?:가|이)\s*뭔가요'
            ],
            'procedure': [
                r'(.+)\s*절차',
                r'(.+)\s*방법',
                r'(.+)\s*과정',
                r'어떻게\s*(.+)',
                r'(.+)\s*하는\s*방법'
            ],
            'penalty': [
                r'(.+)\s*벌금',
                r'(.+)\s*처벌',
                r'(.+)\s*형량',
                r'(.+)\s*제재',
                r'(.+)\s*과태료'
            ],
            'requirement': [
                r'(.+)\s*조건',
                r'(.+)\s*요건',
                r'(.+)\s*기준',
                r'(.+)\s*자격'
            ],
            'exception': [
                r'(.+)\s*예외',
                r'(.+)\s*제외',
                r'(.+)\s*면제',
                r'(.+)(?:가|이)\s*안\s*되는'
            ]
        }
    
    def infer_relevant_laws(self, query: str) -> InferenceResult:
        """관련 법률 유추"""
        logger.info(f"법률 유추 시작: '{query}'")
        
        # 1. 쿼리 분석
        query_analysis = self._analyze_query(query)
        
        # 2. 후보 생성
        candidates = self._generate_candidates(query, query_analysis)
        
        # 3. 검증 및 점수화
        verified_candidates = self._verify_candidates(query, candidates)
        
        # 4. 최종 선별
        final_candidates = self._select_final_candidates(verified_candidates)
        
        # 5. 신뢰도 계산
        overall_confidence = self._calculate_overall_confidence(final_candidates)
        
        # 6. 명확화 질문 생성
        needs_clarification, clarification_questions = self._generate_clarification(
            query, query_analysis, final_candidates
        )
        
        result = InferenceResult(
            query=query,
            candidates=final_candidates,
            confidence=overall_confidence,
            needs_clarification=needs_clarification,
            clarification_questions=clarification_questions,
            metadata={
                'query_analysis': query_analysis,
                'total_candidates_generated': len(candidates),
                'candidates_after_verification': len(verified_candidates)
            }
        )
        
        logger.info(f"법률 유추 완료: {len(final_candidates)}개 후보 "
                   f"(신뢰도: {overall_confidence:.3f})")
        
        return result
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """쿼리 분석"""
        analysis = {
            'normalized_query': '',
            'domains': [],
            'query_type': 'general',
            'entities': [],
            'legal_references': {},
            'keywords': []
        }
        
        # 정규화
        normalized_result = self.normalizer.normalize_legal_text(query)
        analysis['normalized_query'] = normalized_result['normalized_text']
        analysis['legal_references'] = normalized_result['references']
        
        # 도메인 분류 (normalizer 사용)
        analysis['domains'] = self.normalizer.classify_legal_domain(query)
        
        # 쿼리 타입 분류
        analysis['query_type'] = self._classify_query_type(query)
        
        # 엔티티 추출
        analysis['entities'] = self._extract_entities(query)
        
        # 키워드 추출
        analysis['keywords'] = self._extract_query_keywords(query)
        
        return analysis
    
    def _classify_query_domains(self, query: str) -> List[str]:
        """쿼리 도메인 분류 (normalizer 위임)"""
        return self.normalizer.classify_legal_domain(query)
    
    def _classify_query_type(self, query: str) -> str:
        """쿼리 타입 분류"""
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return query_type
        return 'general'
    
    def _extract_entities(self, query: str) -> List[str]:
        """엔티티 추출 (간단한 규칙 기반)"""
        entities = []
        
        # 법률명 추출
        law_patterns = [
            r'([가-힣\s]+법)',
            r'([가-힣\s]+령)',
            r'([가-힣\s]+규칙)',
            r'([가-힣\s]+조례)'
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """쿼리에서 키워드 추출"""
        # 불용어 제거 후 중요 명사 추출
        stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '와', '과']
        
        words = re.findall(r'[가-힣]{2,}', query)
        keywords = [word for word in words if word not in stopwords]
        
        return keywords[:10]  # 상위 10개
    
    def _generate_candidates(self, query: str, query_analysis: Dict) -> List[LawCandidate]:
        """후보 법률 생성"""
        candidates = []
        
        # 1. 도메인 기반 후보
        domain_candidates = self._generate_domain_candidates(query_analysis['domains'])
        candidates.extend(domain_candidates)
        
        # 2. 키워드 기반 후보
        keyword_candidates = self._generate_keyword_candidates(query_analysis['keywords'])
        candidates.extend(keyword_candidates)
        
        # 3. 의미 유사도 기반 후보
        semantic_candidates = self._generate_semantic_candidates(query)
        candidates.extend(semantic_candidates)
        
        # 중복 제거
        unique_candidates = self._deduplicate_candidates(candidates)
        
        return unique_candidates[:self.config.max_candidates * 2]  # 검증 전 충분한 후보 확보
    
    def _generate_domain_candidates(self, domains: List[str]) -> List[LawCandidate]:
        """도메인 기반 후보 생성"""
        candidates = []
        
        # 도메인별 대표 법률 매핑
        domain_laws = {
            'labor': [('labor_standards_act', '근로기준법')],
            'privacy': [('privacy_protection_act', '개인정보 보호법')],
            'environment': [('environment_policy_act', '환경정책기본법')],
            'commerce': [('fair_trade_act', '독점규제 및 공정거래에 관한 법률')],
            'criminal': [('criminal_act', '형법')],
            'civil': [('civil_act', '민법')],
            'administrative': [('administrative_basic_act', '행정기본법')],
            'intellectual_property': [('patent_act', '특허법'), ('copyright_act', '저작권법')],
            'tax': [('framework_act_on_national_taxes', '국세기본법')],
            'construction': [('construction_industry_basic_act', '건설산업기본법')]
        }
        
        for domain in domains:
            if domain in domain_laws:
                for law_id, law_title in domain_laws[domain]:
                    candidate = LawCandidate(
                        law_id=law_id,
                        law_title=law_title,
                        confidence=0.6,  # 초기 신뢰도
                        reasoning=f"도메인 '{domain}' 매칭",
                        evidence_chunks=[],
                        domains=[domain],
                        keywords=[],
                        verification_score=0.0
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_keyword_candidates(self, keywords: List[str]) -> List[LawCandidate]:
        """키워드 기반 후보 생성"""
        candidates = []
        
        # 키워드 매칭으로 법률 검색
        for keyword in keywords[:3]:  # 상위 3개 키워드만
            try:
                search_results = self.retriever.search(
                    keyword, top_k=5, search_type="bm25"
                )
                
                for result in search_results:
                    chunk = result.chunk
                    law_id = chunk.metadata.law_id
                    law_title = chunk.metadata.law_title
                    
                    candidate = LawCandidate(
                        law_id=law_id,
                        law_title=law_title,
                        confidence=0.5,
                        reasoning=f"키워드 '{keyword}' 매칭",
                        evidence_chunks=[chunk],
                        domains=[],
                        keywords=[keyword],
                        verification_score=0.0
                    )
                    candidates.append(candidate)
            except Exception as e:
                logger.warning(f"키워드 '{keyword}' 검색 실패: {e}")
        
        return candidates
    
    def _generate_semantic_candidates(self, query: str) -> List[LawCandidate]:
        """의미 유사도 기반 후보 생성"""
        candidates = []
        
        try:
            search_results = self.retriever.search(
                query, top_k=10, search_type="vector"
            )
            
            for result in search_results:
                chunk = result.chunk
                law_id = chunk.metadata.law_id
                law_title = chunk.metadata.law_title
                
                candidate = LawCandidate(
                    law_id=law_id,
                    law_title=law_title,
                    confidence=min(result.score, 0.8),  # 의미 유사도 기반 신뢰도
                    reasoning=f"의미 유사도 매칭 (점수: {result.score:.3f})",
                    evidence_chunks=[chunk],
                    domains=[],
                    keywords=[],
                    verification_score=0.0
                )
                candidates.append(candidate)
        except Exception as e:
            logger.warning(f"의미 유사도 검색 실패: {e}")
        
        return candidates
    
    def _deduplicate_candidates(self, candidates: List[LawCandidate]) -> List[LawCandidate]:
        """후보 중복 제거"""
        seen_laws = {}
        unique_candidates = []
        
        for candidate in candidates:
            if candidate.law_id not in seen_laws:
                seen_laws[candidate.law_id] = candidate
                unique_candidates.append(candidate)
            else:
                # 기존 후보와 병합 (더 높은 신뢰도, 더 많은 근거 유지)
                existing = seen_laws[candidate.law_id]
                if candidate.confidence > existing.confidence:
                    existing.confidence = candidate.confidence
                    existing.reasoning += f"; {candidate.reasoning}"
                
                # 근거 청크 병합
                existing.evidence_chunks.extend(candidate.evidence_chunks)
                existing.evidence_chunks = list(set(existing.evidence_chunks))
                
                # 키워드 병합
                existing.keywords.extend(candidate.keywords)
                existing.keywords = list(set(existing.keywords))
        
        return unique_candidates
    
    def _verify_candidates(self, query: str, candidates: List[LawCandidate]) -> List[LawCandidate]:
        """후보 검증"""
        verified_candidates = []
        
        for candidate in candidates:
            try:
                # 해당 법률 내에서 재검색
                verification_results = self.retriever.search_by_law(
                    candidate.law_id, query, top_k=5
                )
                
                if verification_results:
                    # 재순위화 수행 (가능한 경우)
                    if self.reranker:
                        rerank_results = self.reranker.rerank(query, verification_results, top_k=3)
                        if rerank_results:
                            candidate.verification_score = rerank_results[0].calibrated_score
                            candidate.evidence_chunks = [r.chunk for r in rerank_results[:3]]
                        else:
                            candidate.verification_score = 0.3
                    else:
                        candidate.verification_score = verification_results[0].score
                        candidate.evidence_chunks = [r.chunk for r in verification_results[:3]]
                    
                    # 신뢰도 업데이트
                    candidate.confidence = (
                        candidate.confidence * 0.5 + 
                        candidate.verification_score * 0.5
                    )
                    
                    verified_candidates.append(candidate)
                else:
                    # 검증 실패
                    candidate.verification_score = 0.1
                    candidate.confidence *= 0.5
                    if candidate.confidence >= self.config.min_confidence:
                        verified_candidates.append(candidate)
            
            except Exception as e:
                logger.warning(f"후보 '{candidate.law_title}' 검증 실패: {e}")
                candidate.verification_score = 0.2
                candidate.confidence *= 0.3
                if candidate.confidence >= self.config.min_confidence:
                    verified_candidates.append(candidate)
        
        return verified_candidates
    
    def _select_final_candidates(self, verified_candidates: List[LawCandidate]) -> List[LawCandidate]:
        """최종 후보 선별"""
        # 신뢰도 순으로 정렬
        verified_candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        # 최소 신뢰도 이상인 후보만 선택
        final_candidates = [
            c for c in verified_candidates 
            if c.confidence >= self.config.min_confidence
        ]
        
        return final_candidates[:self.config.max_candidates]
    
    def _calculate_overall_confidence(self, candidates: List[LawCandidate]) -> float:
        """전체 신뢰도 계산"""
        if not candidates:
            return 0.0
        
        # 최고 신뢰도와 평균 신뢰도의 가중 평균
        max_confidence = max(c.confidence for c in candidates)
        avg_confidence = sum(c.confidence for c in candidates) / len(candidates)
        
        overall_confidence = max_confidence * 0.7 + avg_confidence * 0.3
        
        return min(overall_confidence, 1.0)
    
    def _generate_clarification(self, 
                               query: str, 
                               query_analysis: Dict, 
                               candidates: List[LawCandidate]) -> Tuple[bool, List[str]]:
        """명확화 질문 생성"""
        clarification_questions = []
        needs_clarification = False
        
        # 1. 신뢰도가 낮은 경우
        if not candidates or max(c.confidence for c in candidates) < 0.7:
            needs_clarification = True
            
            # 도메인 관련 질문
            if query_analysis['domains']:
                domain_names = {
                    'labor': '근로/노동',
                    'privacy': '개인정보보호',
                    'environment': '환경',
                    'commerce': '상거래/공정거래',
                    'criminal': '형사',
                    'civil': '민사',
                    'administrative': '행정'
                }
                
                domain_list = [domain_names.get(d, d) for d in query_analysis['domains'][:3]]
                clarification_questions.append(
                    f"다음 중 어느 분야와 관련된 질문인가요? {', '.join(domain_list)}"
                )
        
        # 2. 여러 후보가 비슷한 신뢰도를 가진 경우
        if len(candidates) > 1:
            top_candidates = candidates[:3]
            confidence_diff = top_candidates[0].confidence - top_candidates[-1].confidence
            
            if confidence_diff < 0.2:
                needs_clarification = True
                law_titles = [c.law_title for c in top_candidates]
                clarification_questions.append(
                    f"다음 중 어느 법률과 관련된 질문인가요? {', '.join(law_titles)}"
                )
        
        # 3. 쿼리가 너무 일반적인 경우
        if len(query_analysis['keywords']) < 2:
            needs_clarification = True
            clarification_questions.append(
                "더 구체적인 상황이나 키워드를 알려주시면 정확한 답변을 드릴 수 있습니다."
            )
        
        return needs_clarification, clarification_questions[:2]  # 최대 2개 질문
