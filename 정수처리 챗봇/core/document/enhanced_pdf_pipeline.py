"""
향상된 PDF 파이프라인 통합 모듈

기존 PDF 파이프라인에 법률 파이프라인의 최적화 기법을 적용:
- 하이브리드 검색 (벡터 + BM25 + RRF)
- 크로스엔코더 재순위화
- 멀티뷰 인덱싱
- 성능 최적화
"""

import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .pdf_processor import TextChunk, PDFProcessor
from .pdf_router import PDFRouter, PDFMode, PDFResponse, create_optimized_pdf_router
from .vector_store import HybridVectorStore
from core.llm.answer_generator import AnswerGenerator, Answer

import logging
logger = logging.getLogger(__name__)

@dataclass
class EnhancedPDFConfig:
    """향상된 PDF 파이프라인 설정"""
    # 검색 설정
    embedding_model: str = "jhgan/ko-sroberta-multitask"
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    enable_multiview: bool = True
    
    # 성능 설정
    similarity_threshold: float = 0.3
    rerank_threshold: float = 0.55
    max_results: int = 5
    
    # LLM 설정
    llm_model: str = "qwen2:1.5b-instruct-q4_K_M"
    max_context_chunks: int = 3
    max_tokens: int = 256

class EnhancedPDFPipeline:
    """향상된 PDF 파이프라인"""
    
    def __init__(self, config: Optional[EnhancedPDFConfig] = None):
        """파이프라인 초기화"""
        self.config = config or EnhancedPDFConfig()
        
        # PDF 처리기
        self.pdf_processor = PDFProcessor(
            embedding_model=self.config.embedding_model,
            chunk_size=512,
            chunk_overlap=50,
            enable_keyword_extraction=True
        )
        
        # PDF 라우터 (최적화된 설정)
        self.pdf_router = create_optimized_pdf_router(
            embedding_model=self.config.embedding_model
        )
        
        # 답변 생성기
        self.answer_generator = AnswerGenerator(
            model_name=self.config.llm_model
        )
        
        # 벡터 저장소
        self.vector_store = HybridVectorStore(
            embedding_models=[self.config.embedding_model],
            primary_model=self.config.embedding_model
        )
        
        logger.info("향상된 PDF 파이프라인 초기화 완료")
    
    def process_pdf_file(self, pdf_path: str) -> List[TextChunk]:
        """PDF 파일 처리 및 청크 생성"""
        logger.info(f"PDF 파일 처리 시작: {pdf_path}")
        
        try:
            # PDF에서 텍스트 추출 및 청킹
            chunks = self.pdf_processor.process_pdf(pdf_path)
            
            # 벡터 저장소에 추가
            self.vector_store.add_chunks(chunks)
            
            # PDF 라우터에 인덱싱
            self.pdf_router.index_pdf_chunks(chunks)
            
            logger.info(f"PDF 파일 처리 완료: {len(chunks)}개 청크 생성")
            return chunks
            
        except Exception as e:
            logger.error(f"PDF 파일 처리 실패: {e}")
            raise
    
    def process_pdf_chunks(self, chunks: List[TextChunk]) -> None:
        """기존 청크들을 파이프라인에 추가"""
        logger.info(f"PDF 청크 처리 시작: {len(chunks)}개")
        
        try:
            # 벡터 저장소에 추가
            self.vector_store.add_chunks(chunks)
            
            # PDF 라우터에 인덱싱
            self.pdf_router.index_pdf_chunks(chunks)
            
            logger.info("PDF 청크 처리 완료")
            
        except Exception as e:
            logger.error(f"PDF 청크 처리 실패: {e}")
            raise
    
    def search_and_answer(self, 
                         query: str, 
                         mode: PDFMode = PDFMode.ACCURACY,
                         max_results: Optional[int] = None) -> Dict[str, Any]:
        """검색 및 답변 생성"""
        start_time = time.time()
        
        logger.info(f"PDF 검색 및 답변 생성 시작: '{query}' (모드: {mode.value})")
        
        try:
            # 1. PDF 검색
            search_results = self.pdf_router.search_pdf(
                query=query,
                mode=mode,
                top_k=max_results or self.config.max_results
            )
            
            # 2. 컨텍스트 준비
            context_chunks = self._prepare_context(search_results.results)
            
            # 3. 답변 생성
            if context_chunks:
                answer = self.answer_generator.generate_context_answer(
                    question=query,
                    context_chunks=context_chunks
                )
            else:
                answer = self.answer_generator.generate_direct_answer(query)
            
            total_time = time.time() - start_time
            
            # 4. 결과 구성
            result = {
                'query': query,
                'answer': answer.content,
                'confidence': answer.confidence,
                'sources': self._format_sources(search_results.results),
                'search_results': {
                    'total_found': len(search_results.results),
                    'threshold_passed': sum(1 for r in search_results.results if r.passed_threshold),
                    'search_time': search_results.search_time,
                    'rerank_time': search_results.rerank_time,
                    'total_search_time': search_results.total_time
                },
                'generation_time': answer.generation_time,
                'total_time': total_time,
                'mode': mode.value,
                'metadata': {
                    'context_chunks_used': len(context_chunks),
                    'max_context_chunks': self.config.max_context_chunks,
                    'llm_model': self.config.llm_model
                }
            }
            
            logger.info(f"PDF 검색 및 답변 생성 완료: 총 시간 {total_time:.3f}초, 신뢰도 {answer.confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"PDF 검색 및 답변 생성 실패: {e}")
            return {
                'query': query,
                'answer': f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'error': str(e),
                'total_time': time.time() - start_time
            }
    
    def _prepare_context(self, search_results: List) -> List[TextChunk]:
        """검색 결과에서 컨텍스트 준비"""
        context_chunks = []
        
        # 상위 결과들을 컨텍스트로 사용
        for result in search_results[:self.config.max_context_chunks]:
            if hasattr(result, 'chunk'):
                chunk = result.chunk
            else:
                chunk = result
            
            # TextChunk 객체로 변환
            if isinstance(chunk, TextChunk):
                context_chunks.append(chunk)
            else:
                # 다른 형태의 청크를 TextChunk로 변환
                text_chunk = TextChunk(
                    content=getattr(chunk, 'content', str(chunk)),
                    page_number=getattr(chunk, 'page_number', 1),
                    chunk_id=getattr(chunk, 'chunk_id', f"chunk_{len(context_chunks)}"),
                    metadata=getattr(chunk, 'metadata', {})
                )
                context_chunks.append(text_chunk)
        
        return context_chunks
    
    def _format_sources(self, search_results: List) -> List[Dict[str, Any]]:
        """검색 결과를 소스 형태로 포맷팅"""
        sources = []
        
        for result in search_results:
            if hasattr(result, 'chunk'):
                chunk = result.chunk
                confidence = getattr(result, 'confidence', 0.0)
            else:
                chunk = result
                confidence = 0.0
            
            source = {
                'content': getattr(chunk, 'content', str(chunk))[:200] + "...",  # 200자로 제한
                'page_number': getattr(chunk, 'page_number', 1),
                'confidence': confidence,
                'metadata': getattr(chunk, 'metadata', {})
            }
            sources.append(source)
        
        return sources
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """파이프라인 통계 정보"""
        return {
            'config': {
                'embedding_model': self.config.embedding_model,
                'enable_hybrid_search': self.config.enable_hybrid_search,
                'enable_reranking': self.config.enable_reranking,
                'enable_multiview': self.config.enable_multiview,
                'similarity_threshold': self.config.similarity_threshold,
                'max_results': self.config.max_results
            },
            'router_stats': self.pdf_router.get_router_stats(),
            'vector_store_stats': {
                'total_chunks': len(self.vector_store.faiss_store.chunks) if hasattr(self.vector_store, 'faiss_store') else 0
            }
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
        self.pdf_router.reset_stats()
        logger.info("향상된 PDF 파이프라인 통계 초기화 완료")

# 편의 함수들
def create_enhanced_pdf_pipeline(embedding_model: str = "jhgan/ko-sroberta-multitask",
                                llm_model: str = "qwen2:1.5b-instruct-q4_K_M") -> EnhancedPDFPipeline:
    """향상된 PDF 파이프라인 생성"""
    config = EnhancedPDFConfig(
        embedding_model=embedding_model,
        llm_model=llm_model,
        enable_hybrid_search=True,
        enable_reranking=True,
        enable_multiview=True
    )
    return EnhancedPDFPipeline(config=config)

def create_fast_pdf_pipeline(embedding_model: str = "jhgan/ko-sroberta-multitask",
                            llm_model: str = "qwen2:1.5b-instruct-q4_K_M") -> EnhancedPDFPipeline:
    """속도 최적화 PDF 파이프라인 생성"""
    config = EnhancedPDFConfig(
        embedding_model=embedding_model,
        llm_model=llm_model,
        enable_hybrid_search=True,
        enable_reranking=False,  # 재순위화 비활성화
        enable_multiview=False,  # 멀티뷰 비활성화
        similarity_threshold=0.25,
        max_results=8
    )
    return EnhancedPDFPipeline(config=config)

