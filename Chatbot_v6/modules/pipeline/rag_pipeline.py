"""
RAG Pipeline - RAG 파이프라인

모든 모듈을 통합한 완전한 RAG 파이프라인 (단일 책임: 조율).
"""

from __future__ import annotations

import time
from typing import List, Optional

from config.pipeline_config import PipelineConfig
from config.model_config import ModelConfig
from modules.core.types import Chunk, Answer
from modules.core.logger import get_logger
from modules.core.exceptions import PipelineInitError, PipelineExecutionError
from modules.embedding.factory import create_embedder
from modules.retrieval.hybrid_retriever import HybridRetriever
from modules.generation.answer_generator import AnswerGenerator
from modules.analysis.question_analyzer import QuestionAnalyzer
from modules.filtering.context_filter import ContextFilter
from modules.filtering.deduplicator import Deduplicator
from modules.filtering.guardrail import GuardrailChecker
from modules.reranking.reranker import Reranker

logger = get_logger(__name__)


class RAGPipeline:
    """
    RAG 파이프라인
    
    단일 책임: 모든 모듈을 조율하여 end-to-end RAG 수행
    """
    
    def __init__(
        self,
        chunks: List[Chunk],
        pipeline_config: Optional[PipelineConfig] = None,
        model_config: Optional[ModelConfig] = None,
        evaluation_mode: bool = False,
    ):
        """
        Args:
            chunks: 청크 리스트
            pipeline_config: 파이프라인 설정
            model_config: 모델 설정
            evaluation_mode: 평가 모드 (True시 평가 최적화 프롬프트 사용)
        """
        self.chunks = chunks
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.model_config = model_config or ModelConfig()
        self.evaluation_mode = evaluation_mode
        
        logger.info("RAGPipeline initializing",
                   num_chunks=len(chunks),
                   config_hash=self.pipeline_config.config_hash(),
                   evaluation_mode=evaluation_mode)
        
        try:
            # 질문 분석기
            self.question_analyzer = QuestionAnalyzer(
                domain_dict_path=self.pipeline_config.domain.domain_dict_path
            )
            
            # 임베더 생성
            self.embedder = create_embedder(self.model_config.embedding)
            
            # 검색기 생성
            self.retriever = HybridRetriever(
                chunks=chunks,
                embedder=self.embedder,
                vector_weight=self.pipeline_config.rrf.vector_weight,
                bm25_weight=self.pipeline_config.rrf.bm25_weight,
                index_dir=self.pipeline_config.vector_store_dir,
            )
            
            # 필터링 및 리랭킹
            self.deduplicator = Deduplicator(self.pipeline_config.deduplication)
            self.context_filter = ContextFilter(self.pipeline_config)
            self.guardrail = GuardrailChecker(self.pipeline_config)
            self.reranker = Reranker(self.pipeline_config)
            
            # 답변 생성기 생성
            from modules.generation.llm_client import OllamaClient
            llm_client = OllamaClient(self.model_config.llm)
            
            # 평가 모드에 따라 PromptBuilder 생성
            from modules.generation.prompt_builder import PromptBuilder
            prompt_builder = PromptBuilder(
                domain_dict_path=self.pipeline_config.domain.domain_dict_path,
                evaluation_mode=self.evaluation_mode,
            )
            
            self.generator = AnswerGenerator(
                llm_client=llm_client,
                prompt_builder=prompt_builder,
                max_retries=self.pipeline_config.llm_retries,
                retry_backoff_ms=self.pipeline_config.llm_retry_backoff_ms,
            )
            
            logger.info("RAGPipeline initialized successfully")
        
        except Exception as e:
            raise PipelineInitError(
                pipeline_name="RAGPipeline",
                cause=e,
            ) from e
    
    def ask(
        self,
        question: str,
        top_k: int = 50,
    ) -> Answer:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 질문
            top_k: 검색할 상위 결과 수
            
        Returns:
            Answer 객체
        """
        t0 = time.time()
        
        logger.info(f"Processing question", question=question)
        
        try:
            # 1. 질문 분석
            logger.debug("Step 1: Question Analysis")
            analysis = self.question_analyzer.analyze(question)
            
            # 2. 검색 (분석 결과 반영 + 조건부 쿼리 확장)
            logger.debug("Step 2: Retrieval")
            # 조건부 쿼리 확장 (성능 최적화)
            if analysis.expanded_query and analysis.expanded_query != question:
                # 확장된 쿼리가 원본과 다를 때만 사용
                search_query = analysis.expanded_query
                logger.debug(f"Using expanded query: {search_query}")
            else:
                # 원본 쿼리 사용 (성능 우선)
                search_query = question
                logger.debug(f"Using original query: {search_query}")
            
            spans, retrieval_metrics = self.retriever.search(
                query=search_query,
                top_k=top_k,
                vector_weight=analysis.rrf_vector_weight,
                bm25_weight=analysis.rrf_bm25_weight,
            )
            
            if not spans:
                logger.warning("No contexts found")
                return Answer(
                    text="문서에서 관련 정보를 찾을 수 없습니다.",
                    confidence=0.0,
                    sources=[],
                    metrics={
                        **retrieval_metrics,
                        "total_time_ms": int((time.time() - t0) * 1000),
                        "no_results": 1,
                        "question_type": analysis.qtype,
                    },
                    fallback_used="no_results",
                )
            
            # 3. 중복 제거
            logger.debug("Step 3: Deduplication")
            spans = self.deduplicator.deduplicate(spans)
            
            # 4. 필터링 및 캘리브레이션
            logger.debug("Step 4: Filtering")
            threshold = self.pipeline_config.thresholds.confidence_threshold
            if analysis.qtype == "numeric":
                threshold = min(threshold, self.pipeline_config.thresholds.confidence_threshold_numeric)
            
            spans, filter_stats = self.context_filter.filter_and_calibrate(
                spans, question, threshold
            )
            
            # 5. 리랭킹 (accuracy 모드일 때만)
            if self.pipeline_config.flags.mode == "accuracy":
                logger.debug("Step 5: Reranking")
                spans, rerank_time = self.reranker.rerank(question, spans)
            else:
                rerank_time = 0
            
            # 6. Context 선택
            k = self._choose_k(analysis)
            contexts = spans[:k]
            
            # 7. Guardrail 체크
            logger.debug("Step 6: Guardrail")
            guard_result = self.guardrail.check(question, contexts)
            
            # 8. Fallback 처리
            fallback_used = "none"
            if guard_result["hard_blocked"] or not contexts:
                # Threshold 완화 후 재시도
                relaxed_threshold = max(0.15, threshold * 0.7)
                contexts_retry, _ = self.context_filter.filter_and_calibrate(
                    spans, question, relaxed_threshold
                )
                if contexts_retry:
                    contexts = contexts_retry[:k]
                    fallback_used = "low_conf_retry"
                elif spans:
                    contexts = spans[:min(2, len(spans))]
                    fallback_used = "single_span"
            
            # 9. 답변 생성
            logger.debug(f"Step 7: Generation with {len(contexts)} contexts")
            
            gen_start = time.time()
            answer_text = self.generator.generate(
                question=question,
                contexts=contexts,
                question_type=analysis.qtype,
            )
            gen_time_ms = int((time.time() - gen_start) * 1000)
            
            # 10. 신뢰도 계산
            if contexts:
                calibrated_confs = [
                    c.calibrated_conf for c in contexts[:3]
                    if c.calibrated_conf is not None
                ]
                ctx_conf = sum(calibrated_confs) / len(calibrated_confs) if calibrated_confs else 0.6
            else:
                ctx_conf = 0.6
            
            guard_overlap = float(guard_result.get("overlap_ratio", 0.0))
            confidence = max(0.0, min(1.0, 0.7 * ctx_conf + 0.3 * guard_overlap))
            
            # 11. 메트릭 수집
            total_time_ms = int((time.time() - t0) * 1000)
            
            metrics = {
                **retrieval_metrics,
                **filter_stats,
                **guard_result,
                "generation_time_ms": gen_time_ms,
                "rerank_time_ms": rerank_time,
                "total_time_ms": total_time_ms,
                "num_contexts_used": len(contexts),
                "question_type": analysis.qtype,
                "config_hash": self.pipeline_config.config_hash(),
            }
            
            logger.info("Question processed successfully",
                       total_time_ms=total_time_ms,
                       confidence=confidence,
                       question_type=analysis.qtype)
            
            return Answer(
                text=answer_text,
                confidence=confidence,
                sources=contexts,
                metrics=metrics,
                fallback_used=fallback_used,
            )
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise PipelineExecutionError(
                stage="ask",
                cause=e,
            ) from e
    
    def _choose_k(self, analysis) -> int:
        """질문 분석 결과에 따라 k 선택"""
        if analysis.qtype == "numeric":
            return self.pipeline_config.context.k_numeric
        elif analysis.qtype in ["definition", "technical_spec"]:
            return self.pipeline_config.context.k_definition_max
        else:
            return self.pipeline_config.context.k_default

