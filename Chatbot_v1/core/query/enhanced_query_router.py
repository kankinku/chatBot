"""
향상된 쿼리 라우터

기존 쿼리 라우터에 향상된 PDF 파이프라인을 통합:
- 하이브리드 검색 (벡터 + BM25 + RRF)
- 크로스엔코더 재순위화
- 멀티뷰 인덱싱
- 성능 최적화
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .query_router import QueryRouter, QueryRoute, RouteResult
from .enhanced_pdf_handler import EnhancedPDFHandler, create_enhanced_pdf_handler
from core.document.pdf_processor import TextChunk

import logging
logger = logging.getLogger(__name__)

@dataclass
class EnhancedRouteResult(RouteResult):
    """향상된 라우팅 결과"""
    handler: Optional[Any] = None
    pipeline_version: str = "2.0"

class EnhancedQueryRouter:
    """향상된 쿼리 라우터"""
    
    def __init__(self, 
                 embedding_model: str = "jhgan/ko-sroberta-multitask",
                 llm_model: str = "qwen2:1.5b-instruct-q4_K_M",
                 enable_enhanced_pdf: bool = True):
        """향상된 라우터 초기화"""
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.enable_enhanced_pdf = enable_enhanced_pdf
        
        # 기존 라우터 초기화
        self.base_router = QueryRouter(embedding_model=embedding_model)
        
        # 향상된 PDF 처리기 초기화
        if enable_enhanced_pdf:
            self.enhanced_pdf_handler = create_enhanced_pdf_handler(
                embedding_model=embedding_model,
                llm_model=llm_model
            )
        else:
            self.enhanced_pdf_handler = None
        
        logger.info(f"향상된 쿼리 라우터 초기화 완료 (향상된 PDF: {enable_enhanced_pdf})")
    
    def initialize_with_chunks(self, chunks: List[TextChunk]) -> None:
        """기존 청크들로 초기화"""
        if self.enhanced_pdf_handler:
            self.enhanced_pdf_handler.initialize_with_existing_chunks(chunks)
            logger.info(f"향상된 PDF 처리기 초기화 완료: {len(chunks)}개 청크")
    
    def route_query(self, question: str) -> EnhancedRouteResult:
        """질문을 적절한 파이프라인으로 라우팅"""
        start_time = time.time()
        
        # 기존 라우터로 기본 라우팅
        base_result = self.base_router.route_query(question)
        
        # 향상된 결과 생성
        enhanced_result = EnhancedRouteResult(
            route=base_result.route,
            confidence=base_result.confidence,
            reasoning=base_result.reasoning,
            metadata=base_result.metadata,
            pipeline_version="2.0"
        )
        
        # PDF 검색인 경우 향상된 처리기 할당
        if base_result.route == QueryRoute.PDF_SEARCH and self.enhanced_pdf_handler:
            enhanced_result.handler = self.enhanced_pdf_handler
            enhanced_result.reasoning += " (향상된 PDF 파이프라인 사용)"
        
        routing_time = time.time() - start_time
        enhanced_result.metadata = enhanced_result.metadata or {}
        enhanced_result.metadata['routing_time'] = routing_time
        enhanced_result.metadata['enhanced_features'] = {
            'hybrid_search': True,
            'reranking': True,
            'multiview_indexing': True,
            'performance_optimization': True
        }
        
        logger.info(f"향상된 라우팅 완료: {base_result.route.value} (신뢰도: {base_result.confidence:.3f}, 시간: {routing_time:.3f}초)")
        
        return enhanced_result
    
    def handle_query(self, question: str) -> Dict[str, Any]:
        """질문 처리 (라우팅 + 실행)"""
        start_time = time.time()
        
        try:
            # 1. 라우팅
            route_result = self.route_query(question)
            
            # 2. 처리기 실행
            if route_result.route == QueryRoute.PDF_SEARCH and route_result.handler:
                # 향상된 PDF 처리기 사용
                result = route_result.handler.handle_pdf_query(question)
                result['routing_info'] = {
                    'route': route_result.route.value,
                    'confidence': route_result.confidence,
                    'reasoning': route_result.reasoning,
                    'pipeline_version': route_result.pipeline_version
                }
            else:
                # 기존 처리기 사용 (인사말 등)
                result = {
                    'query': question,
                    'answer': "안녕하세요! 정수처리 관련 질문을 해주세요.",
                    'confidence': 0.8,
                    'sources': [],
                    'routing_info': {
                        'route': route_result.route.value,
                        'confidence': route_result.confidence,
                        'reasoning': route_result.reasoning,
                        'pipeline_version': route_result.pipeline_version
                    }
                }
            
            total_time = time.time() - start_time
            result['total_processing_time'] = total_time
            
            logger.info(f"질문 처리 완료: {total_time:.3f}초")
            
            return result
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            return {
                'query': question,
                'answer': f"죄송합니다. 질문 처리 중 오류가 발생했습니다: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'error': str(e),
                'total_processing_time': time.time() - start_time
            }
    
    def get_router_stats(self) -> Dict[str, Any]:
        """라우터 통계 정보"""
        stats = {
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'enhanced_pdf_enabled': self.enable_enhanced_pdf,
            'pipeline_version': '2.0'
        }
        
        if self.enhanced_pdf_handler:
            stats['enhanced_pdf_stats'] = self.enhanced_pdf_handler.get_handler_stats()
        
        return stats
    
    def update_config(self, **kwargs):
        """설정 업데이트"""
        if 'enable_enhanced_pdf' in kwargs:
            self.enable_enhanced_pdf = kwargs['enable_enhanced_pdf']
            logger.info(f"향상된 PDF 활성화: {self.enable_enhanced_pdf}")
        
        if self.enhanced_pdf_handler:
            self.enhanced_pdf_handler.update_config(**kwargs)
    
    def reset_stats(self):
        """통계 정보 초기화"""
        if self.enhanced_pdf_handler:
            self.enhanced_pdf_handler.reset_stats()
        logger.info("향상된 쿼리 라우터 통계 초기화 완료")

# 편의 함수들
def create_enhanced_query_router(embedding_model: str = "jhgan/ko-sroberta-multitask",
                                llm_model: str = "qwen2:1.5b-instruct-q4_K_M") -> EnhancedQueryRouter:
    """향상된 쿼리 라우터 생성"""
    return EnhancedQueryRouter(
        embedding_model=embedding_model,
        llm_model=llm_model,
        enable_enhanced_pdf=True
    )

def create_fast_query_router(embedding_model: str = "jhgan/ko-sroberta-multitask",
                            llm_model: str = "qwen2:1.5b-instruct-q4_K_M") -> EnhancedQueryRouter:
    """속도 최적화 쿼리 라우터 생성"""
    return EnhancedQueryRouter(
        embedding_model=embedding_model,
        llm_model=llm_model,
        enable_enhanced_pdf=True
    )

