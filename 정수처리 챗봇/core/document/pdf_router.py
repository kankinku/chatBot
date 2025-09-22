"""
PDF 파이프라인 통합 라우터

법률 파이프라인의 최적화 기법을 적용한 PDF RAG 시스템:
- 하이브리드 검색 (벡터 + BM25 + RRF)
- 크로스엔코더 재순위화
- 멀티뷰 인덱싱
- 성능 최적화
"""

import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .pdf_retriever import PDFRetriever, PDFSearchConfig, PDFSearchResult
from .pdf_reranker import PDFReranker, PDFRerankConfig, PDFRerankResult
from .pdf_processor import TextChunk
from .vector_store import HybridVectorStore

import logging
logger = logging.getLogger(__name__)

class PDFMode(Enum):
    """PDF 검색 모드"""
    ACCURACY = "accuracy"      # 정확도 우선 (재순위화 사용)
    SPEED = "speed"           # 속도 우선 (재순위화 생략)
    BALANCED = "balanced"     # 균형 모드

@dataclass
class PDFResponse:
    """PDF 검색 응답"""
    results: List[PDFRerankResult]
    confidence: float
    search_time: float
    rerank_time: float
    total_time: float
    mode: PDFMode
    metadata: Dict[str, Any]

@dataclass
class PDFRouterConfig:
    """PDF 라우터 설정"""
    # 검색 설정
    default_mode: PDFMode = PDFMode.ACCURACY
    accuracy_threshold: float = 0.5  # 낮춤
    speed_threshold: float = 0.3     # 낮춤
    # 무응답(abstain) 정책
    min_evidence: int = 3            # 최소 근거 수 상향
    confidence_threshold: float = 0.7 # 응답 허용 최소 신뢰도 상향
    abstain_on_low_confidence: bool = True
    
    # 결과 수 제한
    max_results_accuracy: int = 8    # 늘림
    max_results_speed: int = 10      # 늘림
    max_results_balanced: int = 8    # 늘림
    
    # 재순위화 설정
    enable_reranking: bool = True
    rerank_threshold: float = 0.35   # 낮춤
    
    # 성능 설정
    enable_multiview: bool = True
    similarity_threshold: float = 0.15  # 낮춤

class PDFRouter:
    """PDF 파이프라인 통합 라우터"""
    
    def __init__(self, 
                 embedding_model: str = "jhgan/ko-sroberta-multitask",
                 config: Optional[PDFRouterConfig] = None,
                 vector_store: Optional[HybridVectorStore] = None):
        """PDF 라우터 초기화"""
        self.config = config or PDFRouterConfig()
        self.embedding_model = embedding_model
        
        # 검색기 초기화
        search_config = PDFSearchConfig(
            multiview_enabled=self.config.enable_multiview,
            similarity_threshold=self.config.similarity_threshold
        )
        self.retriever = PDFRetriever(
            embedding_model=embedding_model,
            config=search_config,
            vector_store=vector_store
        )
        
        # 재순위화기 초기화
        rerank_config = PDFRerankConfig(
            threshold=self.config.rerank_threshold
        )
        self.reranker = PDFReranker(config=rerank_config) if self.config.enable_reranking else None
        
        logger.info(f"PDF 라우터 초기화 완료 (모드: {self.config.default_mode.value})")
    
    def index_pdf_chunks(self, chunks: List[TextChunk]) -> None:
        """PDF 청크들을 인덱싱"""
        logger.info(f"PDF 청크 인덱싱 시작: {len(chunks)}개")
        self.retriever.index_pdf_chunks(chunks)
        logger.info("PDF 청크 인덱싱 완료")
    
    def search_pdf(self, 
                   query: str, 
                   mode: Optional[PDFMode] = None,
                   top_k: Optional[int] = None) -> PDFResponse:
        """PDF 검색 수행"""
        search_mode = mode or self.config.default_mode
        start_time = time.time()
        
        logger.info(f"PDF 검색 시작: '{query}' (모드: {search_mode.value})")
        
        # 모드별 설정
        if search_mode == PDFMode.ACCURACY:
            search_top_k = top_k or self.config.max_results_accuracy
            use_reranking = True
        elif search_mode == PDFMode.SPEED:
            search_top_k = top_k or self.config.max_results_speed
            use_reranking = False
        else:  # BALANCED
            search_top_k = top_k or self.config.max_results_balanced
            use_reranking = self.config.enable_reranking
        
        # 1. 하이브리드 검색
        search_start = time.time()
        search_results = self.retriever.search(
            query, 
            top_k=search_top_k * 2,  # 재순위화를 위해 더 많은 후보 검색
            search_type="hybrid"
        )
        search_time = time.time() - search_start
        
        # 2. 재순위화 (조건부)
        rerank_start = time.time()
        if use_reranking and self.reranker and search_results:
            rerank_results = self.reranker.rerank(
                query, 
                search_results, 
                top_k=search_top_k
            )
            rerank_time = time.time() - rerank_start
            
            # 임계값 통과 결과만 필터링
            final_results = [
                result for result in rerank_results 
                if result.passed_threshold
            ]
            
            # 임계값 통과 결과가 부족하면 상위 결과로 보완
            if len(final_results) < search_top_k:
                additional_needed = search_top_k - len(final_results)
                additional_results = [
                    result for result in rerank_results 
                    if not result.passed_threshold
                ][:additional_needed]
                final_results.extend(additional_results)
            
            final_results = final_results[:search_top_k]
            
        else:
            # 재순위화 없이 직접 변환
            final_results = []
            for i, result in enumerate(search_results[:search_top_k]):
                final_results.append(PDFRerankResult(
                    chunk=result.chunk,
                    original_score=result.score,
                    rerank_score=result.score,
                    calibrated_score=result.score,
                    confidence=min(result.score, 1.0),
                    rank=i + 1,
                    passed_threshold=result.score >= self.config.rerank_threshold
                ))
            rerank_time = 0.0
        
        total_time = time.time() - start_time
        
        # 3. 신뢰도 계산
        confidence = self._calculate_confidence(final_results, search_mode)

        # 3.5 무응답(abstain) 정책 적용
        passed_count = sum(1 for r in final_results if r.passed_threshold)
        should_abstain = False
        if self.config.abstain_on_low_confidence:
            if passed_count < self.config.min_evidence:
                should_abstain = True
            if confidence < self.config.confidence_threshold:
                should_abstain = True
        if should_abstain:
            logger.info(
                f"무응답 결정: passed={passed_count}, conf={confidence:.3f}, "
                f"min_evidence={self.config.min_evidence}, conf_th={self.config.confidence_threshold}"
            )
            final_results = []
            confidence = 0.0
        
        # 4. 메타데이터 생성
        metadata = {
            'search_mode': search_mode.value,
            'total_candidates': len(search_results),
            'reranking_used': use_reranking,
            'threshold_passed': sum(1 for r in final_results if r.passed_threshold),
            'retriever_stats': self.retriever.get_search_stats(),
            'reranker_stats': self.reranker.get_stats() if self.reranker else None,
            'abstained': should_abstain if 'should_abstain' in locals() else False
        }
        
        response = PDFResponse(
            results=final_results,
            confidence=confidence,
            search_time=search_time,
            rerank_time=rerank_time,
            total_time=total_time,
            mode=search_mode,
            metadata=metadata
        )
        
        logger.info(f"PDF 검색 완료: {len(final_results)}개 결과, 신뢰도: {confidence:.3f}, 총 시간: {total_time:.3f}초")
        
        return response
    
    def _calculate_confidence(self, 
                            results: List[PDFRerankResult], 
                            mode: PDFMode) -> float:
        """검색 결과의 신뢰도 계산"""
        if not results:
            return 0.0
        
        # 상위 결과들의 평균 신뢰도
        top_results = results[:3]  # 상위 3개 결과
        avg_confidence = sum(r.confidence for r in top_results) / len(top_results)
        
        # 임계값 통과 비율
        threshold_ratio = sum(1 for r in results if r.passed_threshold) / len(results)
        
        # 모드별 가중치
        if mode == PDFMode.ACCURACY:
            # 정확도 모드: 신뢰도와 임계값 비율 모두 고려
            confidence = (avg_confidence * 0.7) + (threshold_ratio * 0.3)
        elif mode == PDFMode.SPEED:
            # 속도 모드: 신뢰도 위주
            confidence = avg_confidence * 0.9 + (threshold_ratio * 0.1)
        else:  # BALANCED
            # 균형 모드: 중간 가중치
            confidence = (avg_confidence * 0.6) + (threshold_ratio * 0.4)
        
        return min(confidence, 1.0)
    
    def get_router_stats(self) -> Dict[str, Any]:
        """라우터 통계 정보"""
        return {
            'config': {
                'default_mode': self.config.default_mode.value,
                'enable_reranking': self.config.enable_reranking,
                'enable_multiview': self.config.enable_multiview,
                'similarity_threshold': self.config.similarity_threshold
            },
            'retriever_stats': self.retriever.get_search_stats(),
            'reranker_stats': self.reranker.get_stats() if self.reranker else None
        }
    
    def update_config(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"설정 업데이트: {key} = {value}")
            else:
                logger.warning(f"알 수 없는 설정: {key}")
    
    def reset_stats(self):
        """통계 정보 초기화"""
        if self.reranker:
            self.reranker.reset_stats()
        logger.info("PDF 라우터 통계 초기화 완료")

# 편의 함수들
def create_default_pdf_router(embedding_model: str = "jhgan/ko-sroberta-multitask") -> PDFRouter:
    """기본 설정으로 PDF 라우터 생성"""
    config = PDFRouterConfig(
        default_mode=PDFMode.ACCURACY,
        enable_reranking=True,
        enable_multiview=True
    )
    return PDFRouter(embedding_model=embedding_model, config=config)

def create_optimized_pdf_router(embedding_model: str = "jhgan/ko-sroberta-multitask") -> PDFRouter:
    """최적화된 설정으로 PDF 라우터 생성"""
    config = PDFRouterConfig(
        default_mode=PDFMode.ACCURACY,
        accuracy_threshold=0.75,
        rerank_threshold=0.55,
        max_results_accuracy=5,
        enable_reranking=True,
        enable_multiview=True,
        similarity_threshold=0.18
    )
    return PDFRouter(embedding_model=embedding_model, config=config)

def create_speed_pdf_router(embedding_model: str = "jhgan/ko-sroberta-multitask") -> PDFRouter:
    """속도 최적화 PDF 라우터 생성"""
    config = PDFRouterConfig(
        default_mode=PDFMode.SPEED,
        enable_reranking=True,
        enable_multiview=False,
        max_results_speed=8,
        similarity_threshold=0.25
    )
    return PDFRouter(embedding_model=embedding_model, config=config)
