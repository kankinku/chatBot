"""
법률 라우터 모듈

법률 모드 엔트리 포인트로, 임계값·필터·결과 형식을 제어하고
기존 쿼리 라우터와 통합됩니다.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .legal_schema import LegalDocument, LegalChunk, LegalNormalizer
from .legal_chunker import LegalChunker, ChunkingConfig
from .legal_indexer import LegalIndexer
from .legal_retriever import LegalRetriever, SearchConfig, SearchResult
from .legal_reranker import LegalReranker, RerankConfig, RerankResult
from .law_inference import LawInference, InferenceConfig, InferenceResult, LawCandidate

import logging
logger = logging.getLogger(__name__)

class LegalMode(Enum):
    """법률 검색 모드"""
    ACCURACY = "accuracy"      # 정확도 우선 (보수적)
    EXPLORATION = "exploration"  # 탐색 모드 (폭넓은 결과)
    INFERENCE = "inference"    # 법률 유추 모드

@dataclass
class LegalRouterConfig:
    """법률 라우터 설정"""
    default_mode: LegalMode = LegalMode.ACCURACY
    accuracy_threshold: float = 0.7      # 정확도 모드 임계값
    exploration_threshold: float = 0.4   # 탐색 모드 임계값
    max_results_accuracy: int = 3        # 정확도 모드 최대 결과
    max_results_exploration: int = 10    # 탐색 모드 최대 결과
    enable_reranking: bool = True        # 재순위화 사용 여부
    enable_inference: bool = True        # 법률 유추 사용 여부
    conservative_response: bool = True   # 보수적 응답 모드

@dataclass
class LegalResponse:
    """법률 검색 응답"""
    mode: LegalMode                      # 사용된 모드
    query: str                          # 원본 질의
    results: List[RerankResult]         # 검색 결과
    inference_result: Optional[InferenceResult] = None  # 유추 결과
    confidence: float = 0.0             # 전체 신뢰도
    source_citations: List[str] = None  # 근거 문헌 표시(법령명/조문 등)
    reasoning: str = ""                 # 응답 근거
    needs_clarification: bool = False   # 추가 질문 필요
    clarification_questions: List[str] = None  # 명확화 질문
    metadata: Dict[str, Any] = None     # 메타데이터
    
    def __post_init__(self):
        if self.source_citations is None:
            self.source_citations = []
        if self.clarification_questions is None:
            self.clarification_questions = []
        if self.metadata is None:
            self.metadata = {}

class LegalRouter:
    """법률 라우터 클래스"""
    
    def __init__(self, 
                 config: Optional[LegalRouterConfig] = None,
                 embedding_model: str = "jhgan/ko-sroberta-multitask"):
        """라우터 초기화"""
        self.config = config or LegalRouterConfig()
        self.embedding_model_name = embedding_model
        self.normalizer = LegalNormalizer()
        
        # 컴포넌트 초기화 (지연 로딩)
        self.indexer: Optional[LegalIndexer] = None
        self.retriever: Optional[LegalRetriever] = None
        self.reranker: Optional[LegalReranker] = None
        self.inference: Optional[LawInference] = None
        
        # 초기화 상태
        self.is_initialized = False
        
        # 통계 정보
        self.stats = {
            'total_queries': 0,
            'mode_usage': {mode.value: 0 for mode in LegalMode},
            'avg_confidence': 0.0,
            'threshold_passes': 0
        }
        
        logger.info(f"법률 라우터 초기화 완료 (기본 모드: {self.config.default_mode.value})")
    
    def initialize_components(self, 
                            legal_documents: Optional[List[LegalDocument]] = None,
                            force_reindex: bool = False) -> None:
        """컴포넌트 초기화 및 인덱싱"""
        logger.info("법률 라우터 컴포넌트 초기화 시작...")
        
        # 1. 인덱서 초기화
        self.indexer = LegalIndexer(
            embedding_model=self.embedding_model_name,
            vector_store_path="vector_store/legal"
        )
        
        # 2. 기존 인덱스 로드 시도
        if not force_reindex:
            try:
                self.indexer.load_index()
                logger.info("기존 법률 인덱스 로드 완료")
            except Exception as e:
                logger.warning(f"기존 인덱스 로드 실패, 새로 생성: {e}")
                force_reindex = True
        
        # 3. 새 문서가 있거나 강제 재인덱싱인 경우
        if legal_documents and (force_reindex or not self.indexer.get_index_stats()['bm25_documents']):
            logger.info(f"법률 문서 인덱싱: {len(legal_documents)}개 문서")
            self.indexer.index_legal_documents(legal_documents)
            self.indexer.save_index()
        
        # 4. 검색기 초기화
        search_config = SearchConfig(
            dense_k=200,
            bm25_k=200,
            rrf_k=200,
            mmr_k=80,
            mmr_lambda=0.7
        )
        self.retriever = LegalRetriever(
            indexer=self.indexer,
            config=search_config
        )
        
        # 5. 재순위화기 초기화 (선택적)
        if self.config.enable_reranking:
            rerank_config = RerankConfig(
                threshold=self.config.accuracy_threshold,
                calibration_enabled=True
            )
            self.reranker = LegalReranker(config=rerank_config)
        
        # 6. 법률 유추기 초기화 (선택적)
        if self.config.enable_inference:
            inference_config = InferenceConfig(
                min_confidence=0.6,
                max_candidates=5
            )
            self.inference = LawInference(
                retriever=self.retriever,
                reranker=self.reranker,
                config=inference_config
            )
        
        self.is_initialized = True
        logger.info("법률 라우터 컴포넌트 초기화 완료")
    
    def route_legal_query(self, 
                         query: str,
                         mode: Optional[LegalMode] = None,
                         filters: Optional[Dict] = None) -> LegalResponse:
        """법률 질의 라우팅"""
        if not self.is_initialized:
            raise RuntimeError("법률 라우터가 초기화되지 않았습니다. initialize_components()를 먼저 호출하세요.")
        
        mode = mode or self.config.default_mode
        self.stats['total_queries'] += 1
        self.stats['mode_usage'][mode.value] += 1
        
        logger.info(f"법률 질의 라우팅: '{query}' (모드: {mode.value})")
        
        if mode == LegalMode.ACCURACY:
            return self._handle_accuracy_mode(query, filters)
        elif mode == LegalMode.EXPLORATION:
            return self._handle_exploration_mode(query, filters)
        elif mode == LegalMode.INFERENCE:
            return self._handle_inference_mode(query, filters)
        else:
            raise ValueError(f"지원하지 않는 모드: {mode}")
    
    def _handle_accuracy_mode(self, query: str, filters: Optional[Dict]) -> LegalResponse:
        """정확도 우선 모드"""
        # 1. 하이브리드 검색
        search_results = self.retriever.search(
            query, 
            top_k=20,  # 충분한 후보 확보
            search_type="hybrid",
            filters=filters
        )
        
        # 2. 재순위화 (필수)
        if self.reranker:
            rerank_results = self.reranker.rerank(query, search_results, top_k=10)
            
            # 3. 보수적 필터링
            if self.config.conservative_response:
                high_confidence_results = self.reranker.get_conservative_results(
                    rerank_results,
                    min_confidence=0.7,
                    max_results=self.config.max_results_accuracy
                )
            else:
                high_confidence_results = self.reranker.filter_by_threshold(
                    rerank_results, self.config.accuracy_threshold
                )[:self.config.max_results_accuracy]
        else:
            # 재순위화 없이 상위 결과만
            high_confidence_results = [
                RerankResult(
                    chunk=r.chunk,
                    original_score=r.score,
                    rerank_score=r.score,
                    calibrated_score=r.score,
                    confidence=min(r.score, 1.0),
                    rank=i+1,
                    passed_threshold=r.score >= self.config.accuracy_threshold
                )
                for i, r in enumerate(search_results[:self.config.max_results_accuracy])
            ]
        
        # 4. 응답 생성
        confidence = self._calculate_response_confidence(high_confidence_results)
        source_citations = self._generate_citations(high_confidence_results)
        reasoning = self._generate_reasoning(high_confidence_results, "accuracy")
        
        # 5. 불확실한 경우 명확화 질문
        needs_clarification = confidence < 0.6 or len(high_confidence_results) == 0
        clarification_questions = []
        
        if needs_clarification and self.inference:
            inference_result = self.inference.infer_relevant_laws(query)
            clarification_questions = inference_result.clarification_questions
        
        return LegalResponse(
            mode=LegalMode.ACCURACY,
            query=query,
            results=high_confidence_results,
            confidence=confidence,
            source_citations=source_citations,
            reasoning=reasoning,
            needs_clarification=needs_clarification,
            clarification_questions=clarification_questions,
            metadata={
                'search_results_count': len(search_results),
                'threshold_passed': len(high_confidence_results)
            }
        )
    
    def _handle_exploration_mode(self, query: str, filters: Optional[Dict]) -> LegalResponse:
        """탐색 모드"""
        # 1. 폭넓은 검색
        search_results = self.retriever.search(
            query,
            top_k=30,
            search_type="hybrid",
            filters=filters
        )
        
        # 2. 재순위화 (선택적)
        if self.reranker:
            rerank_results = self.reranker.rerank(query, search_results, top_k=20)
            
            # 낮은 임계값 적용
            filtered_results = self.reranker.filter_by_threshold(
                rerank_results, self.config.exploration_threshold
            )[:self.config.max_results_exploration]
        else:
            # 재순위화 없이 상위 결과
            filtered_results = [
                RerankResult(
                    chunk=r.chunk,
                    original_score=r.score,
                    rerank_score=r.score,
                    calibrated_score=r.score,
                    confidence=min(r.score, 1.0),
                    rank=i+1,
                    passed_threshold=r.score >= self.config.exploration_threshold
                )
                for i, r in enumerate(search_results[:self.config.max_results_exploration])
            ]
        
        # 3. 응답 생성
        confidence = self._calculate_response_confidence(filtered_results)
        source_citations = self._generate_citations(filtered_results)
        reasoning = self._generate_reasoning(filtered_results, "exploration")
        
        return LegalResponse(
            mode=LegalMode.EXPLORATION,
            query=query,
            results=filtered_results,
            confidence=confidence,
            source_citations=source_citations,
            reasoning=reasoning,
            metadata={
                'search_results_count': len(search_results),
                'exploration_results': len(filtered_results)
            }
        )
    
    def _handle_inference_mode(self, query: str, filters: Optional[Dict]) -> LegalResponse:
        """법률 유추 모드"""
        if not self.inference:
            logger.warning("법률 유추기가 비활성화됨. 정확도 모드로 대체")
            return self._handle_accuracy_mode(query, filters)
        
        # 1. 법률 유추 수행
        inference_result = self.inference.infer_relevant_laws(query)
        
        # 2. 유추된 법률 기반 검색
        search_results = []
        if inference_result.candidates:
            for candidate in inference_result.candidates[:3]:  # 상위 3개 후보
                law_results = self.retriever.search_by_law(
                    candidate.law_id, query, top_k=5
                )
                search_results.extend(law_results)
        
        # 일반 검색도 병행
        general_results = self.retriever.search(query, top_k=10, search_type="hybrid")
        search_results.extend(general_results)
        
        # 중복 제거
        unique_results = self._deduplicate_search_results(search_results)
        
        # 3. 재순위화
        if self.reranker and unique_results:
            rerank_results = self.reranker.rerank(query, unique_results, top_k=8)
            final_results = rerank_results[:5]
        else:
            final_results = [
                RerankResult(
                    chunk=r.chunk,
                    original_score=r.score,
                    rerank_score=r.score,
                    calibrated_score=r.score,
                    confidence=min(r.score, 1.0),
                    rank=i+1,
                    passed_threshold=r.score >= 0.5
                )
                for i, r in enumerate(unique_results[:5])
            ]
        
        # 4. 응답 생성
        confidence = max(inference_result.confidence, self._calculate_response_confidence(final_results))
        source_citations = self._generate_citations(final_results)
        reasoning = self._generate_inference_reasoning(inference_result, final_results)
        
        return LegalResponse(
            mode=LegalMode.INFERENCE,
            query=query,
            results=final_results,
            inference_result=inference_result,
            confidence=confidence,
            source_citations=source_citations,
            reasoning=reasoning,
            needs_clarification=inference_result.needs_clarification,
            clarification_questions=inference_result.clarification_questions,
            metadata={
                'inferred_laws': len(inference_result.candidates),
                'search_results_count': len(unique_results)
            }
        )
    
    def _calculate_response_confidence(self, results: List[RerankResult]) -> float:
        """응답 신뢰도 계산"""
        if not results:
            return 0.0
        
        # 상위 결과들의 가중 평균
        weights = [1.0, 0.7, 0.5, 0.3, 0.1]  # 상위 결과일수록 높은 가중치
        
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(results[:len(weights)]):
            weight = weights[i]
            total_weighted_confidence += result.confidence * weight
            total_weight += weight
        
        confidence = total_weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # 통계 업데이트
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (self.stats['total_queries'] - 1) + confidence) /
            self.stats['total_queries']
        )
        
        if confidence >= self.config.accuracy_threshold:
            self.stats['threshold_passes'] += 1
        
        return confidence
    
    def _generate_citations(self, results: List[RerankResult]) -> List[str]:
        """출처 인용 생성"""
        citations = []
        
        for result in results:
            metadata = result.chunk.metadata
            citation_parts = [metadata.law_title]
            
            if metadata.article_no:
                citation_parts.append(f"제{metadata.article_no}조")
            if metadata.clause_no:
                citation_parts.append(f"제{metadata.clause_no}항")
            if metadata.effective_date:
                citation_parts.append(f"({metadata.effective_date} 시행)")
            
            citation = " ".join(citation_parts)
            citations.append(citation)
        
        return citations
    
    def _generate_reasoning(self, results: List[RerankResult], mode: str) -> str:
        """응답 근거 생성"""
        if not results:
            return "관련 법률 조문을 찾을 수 없습니다."
        
        reasoning_parts = []
        
        if mode == "accuracy":
            reasoning_parts.append(f"정확도 우선 검색으로 {len(results)}개의 관련 조문을 찾았습니다.")
        elif mode == "exploration":
            reasoning_parts.append(f"탐색 모드로 {len(results)}개의 관련 조문을 폭넓게 검색했습니다.")
        
        # 상위 결과 요약
        if results:
            top_result = results[0]
            reasoning_parts.append(
                f"가장 관련성이 높은 조문은 {top_result.chunk.metadata.law_title} "
                f"제{top_result.chunk.metadata.article_no}조입니다 "
                f"(신뢰도: {top_result.confidence:.2f})."
            )
        
        return " ".join(reasoning_parts)
    
    def _generate_inference_reasoning(self, 
                                    inference_result: InferenceResult,
                                    search_results: List[RerankResult]) -> str:
        """유추 모드 근거 생성"""
        reasoning_parts = []
        
        if inference_result.candidates:
            law_names = [c.law_title for c in inference_result.candidates[:3]]
            reasoning_parts.append(
                f"질의 분석 결과 다음 법률들과 관련될 가능성이 높습니다: {', '.join(law_names)}"
            )
        
        if search_results:
            reasoning_parts.append(
                f"관련 법률에서 {len(search_results)}개의 조문을 검색했습니다."
            )
        
        reasoning_parts.append(f"전체 유추 신뢰도: {inference_result.confidence:.2f}")
        
        return " ".join(reasoning_parts)
    
    def _deduplicate_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """검색 결과 중복 제거"""
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            chunk_id = result.chunk.chunk_id
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        return unique_results
    
    def get_router_stats(self) -> Dict[str, Any]:
        """라우터 통계 정보"""
        stats = self.stats.copy()
        
        # 추가 통계 계산
        if stats['total_queries'] > 0:
            stats['threshold_pass_rate'] = stats['threshold_passes'] / stats['total_queries']
        else:
            stats['threshold_pass_rate'] = 0.0
        
        # 컴포넌트 통계
        if self.retriever:
            stats['retriever_stats'] = self.retriever.get_search_stats()
        
        if self.reranker:
            stats['reranker_stats'] = self.reranker.get_stats()
        
        if self.indexer:
            stats['indexer_stats'] = self.indexer.get_index_stats()
        
        return stats
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'total_queries': 0,
            'mode_usage': {mode.value: 0 for mode in LegalMode},
            'avg_confidence': 0.0,
            'threshold_passes': 0
        }
        
        if self.reranker:
            self.reranker.reset_stats()
        
        logger.info("법률 라우터 통계 초기화 완료")
