"""
향상된 PDF 파이프라인 통합 모듈

기존 PDF 파이프라인에 법률 파이프라인의 최적화 기법을 적용:
- 하이브리드 검색 (벡터 + BM25 + RRF)
- 크로스엔코더 재순위화
- 멀티뷰 인덱싱
- 성능 최적화
"""

import time
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .pdf_processor import TextChunk, PDFProcessor
from .pdf_router import PDFRouter, PDFMode, PDFResponse, create_optimized_pdf_router
from .vector_store import HybridVectorStore
from .enhanced_vector_search import EnhancedVectorSearcher, create_enhanced_searcher
from .accuracy_validator import AccuracyValidator
from .enhanced_search_filter import EnhancedSearchFilter, create_enhanced_search_filter
from .wastewater_reranker import WastewaterReranker, create_wastewater_reranker
from core.query.llm_query_expander import LLMQueryExpander, create_llm_query_expander
from core.llm.answer_generator import AnswerGenerator, Answer
from core.llm.hallucination_prevention import create_hallucination_prevention

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
    max_context_chunks: int = 6  # 근거 커버리지 확대(6~8 권장)
    max_tokens: int = 256
    window_expand: int = 1  # 인접 윈도우 확장 (같은 페이지 인접 청크 우선)
    
    # 검색 필터 설정
    enable_enhanced_filtering: bool = True
    filter_confidence_threshold: float = 0.25
    
    # 쿼리 확장 설정
    enable_query_expansion: bool = True
    max_query_expansions: int = 3
    
    # 재순위화 설정
    enable_wastewater_reranking: bool = True
    domain_rerank_weight: float = 0.4
    
    # 환상 방지 설정
    enable_hallucination_prevention: bool = True
    strict_mode: bool = True

class EnhancedPDFPipeline:
    """향상된 PDF 파이프라인"""
    
    def __init__(self, config: Optional[EnhancedPDFConfig] = None):
        """파이프라인 초기화"""
        self.config = config or EnhancedPDFConfig()
        
        # PDF 처리기 (정수처리 특화 청킹 활성화)
        self.pdf_processor = PDFProcessor(
            embedding_model=self.config.embedding_model,
            chunk_size=256,
            chunk_overlap=30,
            enable_keyword_extraction=True,
            enable_wastewater_chunking=True,
            wastewater_chunk_size=384,
            wastewater_overlap_ratio=0.25
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
        
        # 향상된 벡터 검색기
        self.enhanced_searcher = create_enhanced_searcher(
            vector_store=self.vector_store,
            primary_model=self.config.embedding_model
        )
        
        # 정확도 검증기
        self.accuracy_validator = AccuracyValidator(
            embedding_model=self.config.embedding_model
        )
        
        # 향상된 검색 필터
        if self.config.enable_enhanced_filtering:
            self.search_filter = create_enhanced_search_filter(
                max_chunks=self.config.max_context_chunks,
                confidence_threshold=self.config.filter_confidence_threshold
            )
        else:
            self.search_filter = None
        
        # LLM 쿼리 확장기
        if self.config.enable_query_expansion:
            self.query_expander = create_llm_query_expander(
                llm_model=self.config.llm_model,
                max_expansions=self.config.max_query_expansions
            )
        else:
            self.query_expander = None
        
        # 정수처리 도메인 특화 재순위화기
        if self.config.enable_wastewater_reranking:
            self.wastewater_reranker = create_wastewater_reranker(
                domain_weight=self.config.domain_rerank_weight
            )
        else:
            self.wastewater_reranker = None
        
        # 환상 방지 시스템
        if self.config.enable_hallucination_prevention:
            self.hallucination_prevention = create_hallucination_prevention()
        else:
            self.hallucination_prevention = None
        
        logger.info("향상된 PDF 파이프라인 초기화 완료")
    
    def process_pdf_file(self, pdf_path: str) -> List[TextChunk]:
        """PDF 파일 처리 및 청크 생성"""
        logger.info(f"PDF 파일 처리 시작: {pdf_path}")
        
        try:
            # PDF에서 텍스트 추출 및 청킹
            chunks, metadata = self.pdf_processor.process_pdf(pdf_path)
            
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
                         max_results: Optional[int] = None,
                         expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """검색 및 답변 생성"""
        start_time = time.time()
        
        logger.info(f"PDF 검색 및 답변 생성 시작: '{query}' (모드: {mode.value})")
        
        try:
            # 1. 쿼리 확장 (선택적)
            expanded_queries = []
            if self.query_expander:
                try:
                    expansion_result = self.query_expander.expand_query(query)
                    expanded_queries = expansion_result.expanded_queries
                    logger.info(f"쿼리 확장 완료: {len(expanded_queries)}개 확장 쿼리 생성")
                except Exception as e:
                    logger.warning(f"쿼리 확장 실패, 원본 쿼리 사용: {e}")
            
            # 검색할 쿼리 목록 (원본 + 확장)
            search_queries = [query] + expanded_queries
            
            # 2. 향상된 벡터 검색 수행 (다중 쿼리)
            all_search_results = []
            for search_query in search_queries:
                try:
                    results = self.enhanced_searcher.search(
                        query=search_query,
                        top_k=max_results or self.config.max_results,
                        context=expected_answer
                    )
                    all_search_results.extend(results)
                except Exception as e:
                    logger.warning(f"검색 실패 (쿼리: '{search_query}'): {e}")
            
            # 중복 제거 및 상위 결과 선택
            unique_results = []
            seen_chunk_ids = set()
            for result in all_search_results:
                chunk_id = getattr(result.chunk if hasattr(result, 'chunk') else result, 'chunk_id', None)
                if chunk_id and chunk_id not in seen_chunk_ids:
                    unique_results.append(result)
                    seen_chunk_ids.add(chunk_id)
                elif not chunk_id:  # chunk_id가 없는 경우도 포함
                    unique_results.append(result)
            
            # 상위 결과만 선택 (원본 검색 결과 형태로 유지)
            enhanced_search_results = unique_results[:max_results or self.config.max_results * 2]
            
            # 3. PDF 라우터로 추가 검색 (기존 방식과 병행)
            search_results = self.pdf_router.search_pdf(
                query=query,
                mode=mode,
                top_k=max_results or self.config.max_results
            )
            
            # 4. 정수처리 도메인 특화 재순위화 적용
            if self.wastewater_reranker and search_results.results:
                try:
                    # 청크 추출
                    chunks_to_rerank = []
                    for result in search_results.results:
                        if hasattr(result, 'chunk'):
                            chunks_to_rerank.append(result.chunk)
                        else:
                            # 직접 청크인 경우 TextChunk로 변환
                            if not isinstance(result, TextChunk):
                                result = TextChunk(
                                    content=getattr(result, 'content', str(result)),
                                    page_number=getattr(result, 'page_number', 1),
                                    chunk_id=getattr(result, 'chunk_id', f"chunk_{len(chunks_to_rerank)}"),
                                    metadata=getattr(result, 'metadata', {})
                                )
                            chunks_to_rerank.append(result)
                    
                    # 재순위화 수행
                    reranked_results = self.wastewater_reranker.rerank(
                        query=query,
                        chunks=chunks_to_rerank,
                        top_k=len(chunks_to_rerank)
                    )
                    
                    # 재순위화된 결과로 업데이트
                    search_results.results = [chunk for chunk, score in reranked_results]
                    logger.info(f"정수처리 재순위화 완료: {len(reranked_results)}개 청크")
                    
                except Exception as e:
                    logger.warning(f"정수처리 재순위화 실패, 원본 결과 사용: {e}")
            
            # 5. 검색 결과 정확도 검증
            validation_result = self.accuracy_validator.validate_search_results(
                query=query,
                search_results=enhanced_search_results,
                expected_answer=expected_answer
            )
            
            # 6. 향상된 검색 필터링 적용
            if self.search_filter and search_results.results:
                filtered_results = self.search_filter.filter_search_results(
                    search_results=(enhanced_search_results + (search_results.results or [])),
                    query=query,
                    expected_answer=expected_answer
                )
                # SearchResult를 다시 청크로 변환
                filtered_chunks = [result.chunk for result in filtered_results]
            else:
                filtered_chunks = search_results.results
            
            # 7. 컨텍스트 준비 (인접 윈도우 확장 포함)
            context_chunks = self._prepare_context(filtered_chunks)
            
            # 8. 답변 생성 (PDF-only 정책: 컨텍스트 없으면 무응답)
            # 7.5 무응답 완충: 컨텍스트가 비면 임계 완화 재시도 → 단일 스팬
            fallback_info = {}
            if not context_chunks:
                if self.search_filter and (search_results.results or enhanced_search_results):
                    try:
                        original_th = self.search_filter.config.confidence_threshold
                        self.search_filter.config.confidence_threshold = max(0.0, original_th * 0.8)
                        merged_candidates = (enhanced_search_results + (search_results.results or []))
                        filtered_results = self.search_filter.filter_search_results(
                            search_results=merged_candidates,
                            query=query,
                            expected_answer=expected_answer
                        )
                        filtered_chunks = [result.chunk for result in filtered_results]
                        context_chunks = self._prepare_context(filtered_chunks)
                        fallback_info['low_confidence_retry'] = True
                    finally:
                        self.search_filter.config.confidence_threshold = original_th
                if not context_chunks and (search_results.results or enhanced_search_results):
                    # 최후 수단: 상위 1개 스팬만 투입
                    merged_candidates = (enhanced_search_results + (search_results.results or []))
                    top_candidate = None
                    best_score = -1.0
                    for r in merged_candidates:
                        score = 0.0
                        for attr in ('confidence', 'score', 'rerank_score', 'relevance_score'):
                            if hasattr(r, attr):
                                try:
                                    score = float(getattr(r, attr))
                                    break
                                except Exception:
                                    pass
                        if score > best_score:
                            best_score = score
                            top_candidate = r
                    if top_candidate is not None:
                        ch = top_candidate.chunk if hasattr(top_candidate, 'chunk') else top_candidate
                        context_chunks = [ch]
                        filtered_chunks = [ch]
                        fallback_info['single_span_fallback'] = True

            if context_chunks:
                answer = self.answer_generator.generate_context_answer(
                    question=query,
                    context_chunks=context_chunks
                )
            else:
                # PDF 기반 컨텍스트가 없으므로 즉시 무응답
                total_time = time.time() - start_time
                return {
                    'query': query,
                    'answer': "제공된 PDF 문서에서 답변에 해당하는 정보를 찾을 수 없습니다.",
                    'confidence': 0.0,
                    'sources': [],
                    'search_results': {
                        'total_found': 0,
                        'threshold_passed': 0,
                        'search_time': search_results.search_time,
                        'rerank_time': search_results.rerank_time,
                        'total_search_time': search_results.total_time
                    },
                    'generation_time': 0.0,
                    'total_time': total_time,
                    'mode': mode.value,
                    'metadata': {
                        'context_chunks_used': 0,
                        'max_context_chunks': self.config.max_context_chunks,
                        'llm_model': self.config.llm_model,
                        'pdf_only_abstain': True
                    }
                }
            
            total_time = time.time() - start_time
            
            # 9. 결과 구성
            result = {
                'query': query,
                'answer': answer.content,
                'confidence': answer.confidence,
                'sources': self._format_sources(filtered_chunks),
                'filtering_stats': self.search_filter.get_filter_stats(filtered_results) if self.search_filter and 'filtered_results' in locals() else {},
                'query_expansion': {
                    'original_query': query,
                    'expanded_queries': expanded_queries,
                    'total_search_queries': len(search_queries),
                    'expansion_enabled': self.config.enable_query_expansion
                },
                'reranking_stats': {
                    'wastewater_reranking_enabled': self.config.enable_wastewater_reranking,
                    'domain_weight': self.config.domain_rerank_weight
                },
                'hallucination_prevention': {
                    'enabled': self.config.enable_hallucination_prevention,
                    'strict_mode': self.config.strict_mode
                },
                'search_results': {
                    'total_found': len(search_results.results),
                    'threshold_passed': sum(1 for r in search_results.results if r.passed_threshold),
                    'search_time': search_results.search_time,
                    'rerank_time': search_results.rerank_time,
                    'total_search_time': search_results.total_time
                },
                'enhanced_search': {
                    'total_candidates': len(enhanced_search_results),
                    'avg_confidence': np.mean([r.confidence for r in enhanced_search_results]) if enhanced_search_results else 0.0,
                    'model_scores': {model: np.mean([r.model_scores.get(model, 0.0) for r in enhanced_search_results]) 
                                   for model in self.enhanced_searcher.embedding_models.keys()} if enhanced_search_results else {}
                },
                'fallback': fallback_info,
                'validation': {
                    'is_accurate': validation_result.is_accurate,
                    'confidence_score': validation_result.confidence_score,
                    'accuracy_issues': validation_result.accuracy_issues,
                    'improvement_suggestions': validation_result.improvement_suggestions
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
        """검색 결과에서 컨텍스트 준비 (인접 윈도우 확장)"""
        if not search_results:
            return []

        # 1) 기본 상위 결과 선택
        selected: List[TextChunk] = []
        for result in search_results:
            chunk = result.chunk if hasattr(result, 'chunk') else result
            if isinstance(chunk, TextChunk):
                selected.append(chunk)
            else:
                selected.append(TextChunk(
                    content=getattr(chunk, 'content', str(chunk)),
                    page_number=getattr(chunk, 'page_number', 1),
                    chunk_id=getattr(chunk, 'chunk_id', f"chunk_{len(selected)}"),
                    metadata=getattr(chunk, 'metadata', {})
                ))
            if len(selected) >= self.config.max_context_chunks:
                break

        # 2) 인접 윈도우 확장: 같은 페이지 또는 인접 페이지의 후보 우선 추가
        if self.config.window_expand > 0 and len(selected) < self.config.max_context_chunks:
            base_pages = {c.page_number for c in selected}
            extra: List[TextChunk] = []
            for result in search_results:
                chunk = result.chunk if hasattr(result, 'chunk') else result
                if not isinstance(chunk, TextChunk):
                    chunk = TextChunk(
                        content=getattr(chunk, 'content', str(chunk)),
                        page_number=getattr(chunk, 'page_number', 1),
                        chunk_id=getattr(chunk, 'chunk_id', f"chunk_ex_{len(extra)}"),
                        metadata=getattr(chunk, 'metadata', {})
                    )
                # 같은 페이지 또는 ±window 페이지
                if any(abs(chunk.page_number - p) <= self.config.window_expand for p in base_pages):
                    if all(chunk.chunk_id != c.chunk_id for c in selected) and all(chunk.chunk_id != e.chunk_id for e in extra):
                        extra.append(chunk)
                        if len(selected) + len(extra) >= self.config.max_context_chunks:
                            break
            selected.extend(extra[: max(0, self.config.max_context_chunks - len(selected))])

        return selected
    
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
    """향상된 PDF 파이프라인 생성 (정수처리 특화 기능 포함)"""
    config = EnhancedPDFConfig(
        embedding_model=embedding_model,
        llm_model=llm_model,
        enable_hybrid_search=True,
        enable_reranking=True,
        enable_multiview=True,
        # 정수처리 특화 설정
        enable_enhanced_filtering=True,
        enable_query_expansion=True,
        enable_wastewater_reranking=True,
        max_context_chunks=3,  # 2-3개로 제한
        filter_confidence_threshold=0.25,
        max_query_expansions=3,
        domain_rerank_weight=0.4
    )
    return EnhancedPDFPipeline(config=config)

def create_fast_pdf_pipeline(embedding_model: str = "jhgan/ko-sroberta-multitask",
                            llm_model: str = "qwen2:1.5b-instruct-q4_K_M") -> EnhancedPDFPipeline:
    """속도 최적화 PDF 파이프라인 생성"""
    config = EnhancedPDFConfig(
        embedding_model=embedding_model,
        llm_model=llm_model,
        enable_hybrid_search=True,
        enable_reranking=True,
        enable_multiview=False,  # 멀티뷰 비활성화
        similarity_threshold=0.25,
        max_results=8,
        # 속도 최적화를 위한 설정
        enable_enhanced_filtering=True,
        enable_query_expansion=False,  # 쿼리 확장 비활성화
        enable_wastewater_reranking=True,
        max_context_chunks=2,  # 더 적은 청크
        max_query_expansions=1
    )
    return EnhancedPDFPipeline(config=config)

def create_accuracy_focused_pipeline(embedding_model: str = "jhgan/ko-sroberta-multitask",
                                    llm_model: str = "qwen2:1.5b-instruct-q4_K_M") -> EnhancedPDFPipeline:
    """정확도 중심 PDF 파이프라인 생성 (모든 고급 기능 활성화)"""
    config = EnhancedPDFConfig(
        embedding_model=embedding_model,
        llm_model=llm_model,
        enable_hybrid_search=True,
        enable_reranking=True,
        enable_multiview=True,
        # 정확도 최적화 설정
        enable_enhanced_filtering=True,
        enable_query_expansion=True,
        enable_wastewater_reranking=True,
        max_context_chunks=3,
        filter_confidence_threshold=0.5,  # 더 높은 임계값
        max_query_expansions=5,  # 더 많은 확장
        domain_rerank_weight=0.6,  # 도메인 가중치 증가
        similarity_threshold=0.4,
        rerank_threshold=0.6
    )
    return EnhancedPDFPipeline(config=config)

