"""
향상된 PDF 질의 처리기

기존 PDF 파이프라인을 향상된 버전으로 업그레이드:
- 하이브리드 검색 (벡터 + BM25 + RRF)
- 크로스엔코더 재순위화
- 멀티뷰 인덱싱
- 성능 최적화
"""

import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from core.document.enhanced_water_treatment_pipeline import EnhancedWaterTreatmentPipeline, EnhancedWaterTreatmentConfig
from core.document.text_chunk import TextChunk
from core.document.pdf_router import PDFMode
from core.llm.answer_generator import Answer

import logging
logger = logging.getLogger(__name__)

@dataclass
class EnhancedPDFHandlerConfig:
    """향상된 PDF 처리기 설정"""
    # 파이프라인 설정
    embedding_model: str = "jhgan/ko-sroberta-multitask"
    llm_model: str = "qwen2:1.5b-instruct-q4_K_M"
    
    # 성능 설정
    default_mode: PDFMode = PDFMode.ACCURACY
    enable_reranking: bool = True
    enable_multiview: bool = True
    
    # 결과 설정
    max_results: int = 5
    max_context_chunks: int = 5

class EnhancedPDFHandler:
    """향상된 정수처리 도메인 특화 PDF 질의 처리기"""
    
    def __init__(self, config: Optional[EnhancedPDFHandlerConfig] = None):
        """처리기 초기화"""
        self.config = config or EnhancedPDFHandlerConfig()
        
        # 향상된 정수처리 파이프라인 초기화
        self.pdf_pipeline = self._create_enhanced_water_treatment_pipeline()
        
        # 기존 청크들 (이미 로드된 경우)
        self.existing_chunks = []
        
        logger.info("향상된 정수처리 도메인 특화 PDF 처리기 초기화 완료")
    
    def _create_enhanced_water_treatment_pipeline(self):
        """향상된 정수처리 파이프라인 생성"""
        try:
            from core.document.vector_store import HybridVectorStore
            from core.llm.answer_generator import AnswerGenerator
            from core.llm.answer_generator import OllamaLLMInterface
            
            # 기존 컴포넌트들 초기화
            vector_store = HybridVectorStore()
            answer_generator = AnswerGenerator()
            ollama_interface = OllamaLLMInterface()
            
            # 향상된 설정
            enhanced_config = EnhancedWaterTreatmentConfig(
                chunking_strategy="hybrid",
                enable_query_expansion=True,
                enable_reranking=True,
                enable_context_optimization=True,
                initial_search_k=20,
                final_context_k=3,
                similarity_threshold=0.25
            )
            
            # 향상된 파이프라인 생성
            pipeline = EnhancedWaterTreatmentPipeline(
                vector_store=vector_store,
                answer_generator=answer_generator,
                ollama_interface=ollama_interface,
                config=enhanced_config
            )
            
            logger.info("향상된 정수처리 파이프라인 생성 완료")
            return pipeline
            
        except Exception as e:
            logger.error(f"향상된 파이프라인 생성 실패: {e}")
            # 폴백: 기본 파이프라인 사용
            from core.document.enhanced_pdf_pipeline import create_enhanced_pdf_pipeline
            return create_enhanced_pdf_pipeline(
                embedding_model=self.config.embedding_model,
                llm_model=self.config.llm_model
            )
    
    def initialize_with_existing_chunks(self, chunks: List[TextChunk]) -> None:
        """기존 청크들로 초기화"""
        logger.info(f"기존 청크로 초기화: {len(chunks)}개")
        
        try:
            self.existing_chunks = chunks
            self.pdf_pipeline.process_pdf_chunks(chunks)
            logger.info("기존 청크 초기화 완료")
        except Exception as e:
            logger.error(f"기존 청크 초기화 실패: {e}")
            # 에러 로그에 기록
            try:
                from utils.error_logger import log_error
                log_error(e, "기존 청크 초기화", {"file": "enhanced_pdf_handler.py"})
            except:
                pass
            raise
    
    def process_pdf_file(self, pdf_path: str) -> List[TextChunk]:
        """PDF 파일 처리"""
        return self.pdf_pipeline.process_pdf_file(pdf_path)
    
    def handle_pdf_query(self, 
                        question: str, 
                        mode: Optional[PDFMode] = None,
                        max_results: Optional[int] = None) -> Dict[str, Any]:
        """PDF 질의 처리"""
        start_time = time.time()
        
        logger.info(f"PDF 질의 처리 시작: '{question}'")
        
        try:
            # 모드 결정
            search_mode = mode or self.config.default_mode
            
            # 향상된 파이프라인으로 검색 및 답변 생성
            result = self.pdf_pipeline.search_and_answer(
                query=question,
                mode=search_mode,
                max_results=max_results or self.config.max_results
            )
            
            # 추가 메타데이터
            result['handler_type'] = 'enhanced_pdf'
            result['pipeline_version'] = '2.0'
            result['optimizations'] = {
                'hybrid_search': True,
                'reranking': self.config.enable_reranking,
                'multiview': self.config.enable_multiview
            }
            
            total_time = time.time() - start_time
            result['handler_time'] = total_time
            
            logger.info(f"PDF 질의 처리 완료: {total_time:.3f}초")
            
            return result
            
        except Exception as e:
            logger.error(f"PDF 질의 처리 실패: {e}")
            return {
                'query': question,
                'answer': f"죄송합니다. PDF 검색 중 오류가 발생했습니다: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'error': str(e),
                'handler_type': 'enhanced_pdf',
                'handler_time': time.time() - start_time
            }
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """처리기 통계 정보"""
        return {
            'config': {
                'embedding_model': self.config.embedding_model,
                'llm_model': self.config.llm_model,
                'default_mode': self.config.default_mode.value,
                'enable_reranking': self.config.enable_reranking,
                'enable_multiview': self.config.enable_multiview,
                'max_results': self.config.max_results
            },
            'pipeline_stats': self.pdf_pipeline.get_pipeline_stats(),
            'existing_chunks_count': len(self.existing_chunks)
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
        self.pdf_pipeline.reset_stats()
        logger.info("향상된 PDF 처리기 통계 초기화 완료")

# 편의 함수들
def create_enhanced_pdf_handler(embedding_model: str = "jhgan/ko-sroberta-multitask",
                               llm_model: str = "qwen2:1.5b-instruct-q4_K_M") -> EnhancedPDFHandler:
    """향상된 PDF 처리기 생성"""
    config = EnhancedPDFHandlerConfig(
        embedding_model=embedding_model,
        llm_model=llm_model,
        default_mode=PDFMode.ACCURACY,
        enable_reranking=True,
        enable_multiview=True
    )
    return EnhancedPDFHandler(config=config)

def create_fast_pdf_handler(embedding_model: str = "jhgan/ko-sroberta-multitask",
                           llm_model: str = "qwen2:1.5b-instruct-q4_K_M") -> EnhancedPDFHandler:
    """속도 최적화 PDF 처리기 생성"""
    config = EnhancedPDFHandlerConfig(
        embedding_model=embedding_model,
        llm_model=llm_model,
        default_mode=PDFMode.SPEED,
        enable_reranking=True,
        enable_multiview=False,
        max_results=8
    )
    return EnhancedPDFHandler(config=config)

