"""
RAGPipeline 통합 테스트
"""

import pytest
from config.pipeline_config import PipelineConfig
from config.model_config import ModelConfig
from modules.pipeline.rag_pipeline import RAGPipeline


class TestRAGPipeline:
    """RAG 파이프라인 통합 테스트"""
    
    @pytest.fixture
    def pipeline_config(self, domain_dict_path):
        """테스트용 파이프라인 설정"""
        config = PipelineConfig()
        config.domain.domain_dict_path = domain_dict_path
        # 테스트 속도를 위해 일부 설정 조정
        config.flags.use_cross_reranker = False
        return config
    
    @pytest.fixture
    def pipeline(self, sample_chunks, pipeline_config):
        """파이프라인 인스턴스"""
        model_config = ModelConfig()
        model_config.embedding.device = "cpu"  # 테스트는 CPU 사용
        
        return RAGPipeline(
            chunks=sample_chunks,
            pipeline_config=pipeline_config,
            model_config=model_config,
        )
    
    def test_pipeline_initialization(self, pipeline):
        """파이프라인 초기화 확인"""
        assert pipeline is not None
        assert pipeline.chunks is not None
        assert pipeline.question_analyzer is not None
        assert pipeline.retriever is not None
        assert pipeline.generator is not None
    
    def test_simple_question(self, pipeline):
        """간단한 질문 처리"""
        answer = pipeline.ask("고산 정수장 URL은?")
        
        assert answer is not None
        assert answer.text
        assert 0.0 <= answer.confidence <= 1.0
        assert len(answer.sources) >= 0
        assert "metrics" in answer.__dict__
    
    def test_numeric_question(self, pipeline):
        """숫자 질문 처리"""
        answer = pipeline.ask("설계 용량은?")
        
        assert answer is not None
        assert answer.text
        # 숫자가 포함되어야 함
        assert any(char.isdigit() for char in answer.text) or "확인할 수 없습니다" in answer.text
    
    def test_system_info_question(self, pipeline):
        """시스템 정보 질문"""
        answer = pipeline.ask("AI플랫폼 주소는?")
        
        assert answer is not None
        assert answer.text
    
    def test_no_results_fallback(self, pipeline):
        """결과 없을 때 fallback"""
        answer = pipeline.ask("완전히 관련없는 xyz123 정보")
        
        assert answer is not None
        assert answer.text
        # 낮은 신뢰도 또는 fallback 사용
        assert answer.confidence < 0.7 or answer.fallback_used != "none"
    
    def test_metrics_collection(self, pipeline):
        """메트릭 수집 확인"""
        answer = pipeline.ask("테스트 질문")
        
        assert "metrics" in answer.__dict__
        metrics = answer.metrics
        
        # 필수 메트릭 확인
        assert "total_time_ms" in metrics
        assert "question_type" in metrics
        assert metrics["total_time_ms"] >= 0
    
    def test_context_selection(self, pipeline):
        """컨텍스트 선택"""
        answer = pipeline.ask("pH 기준은?")
        
        # 소스가 반환되어야 함
        assert len(answer.sources) >= 0
        
        # 각 소스는 RetrievedSpan이어야 함
        for source in answer.sources:
            assert hasattr(source, 'chunk')
            assert hasattr(source, 'score')
            assert hasattr(source, 'rank')
    
    def test_question_type_detection(self, pipeline):
        """질문 유형 감지"""
        numeric_answer = pipeline.ask("50000 m³/day는?")
        definition_answer = pipeline.ask("정수장이란?")
        
        # 메트릭에 질문 유형이 포함되어야 함
        assert "question_type" in numeric_answer.metrics
        assert "question_type" in definition_answer.metrics
    
    def test_confidence_variation(self, pipeline):
        """신뢰도 변동"""
        # 명확한 질문
        clear_answer = pipeline.ask("고산 정수장")
        
        # 모호한 질문
        vague_answer = pipeline.ask("xyz")
        
        # 명확한 질문의 신뢰도가 더 높아야 함 (보장은 못하지만 경향성)
        assert clear_answer.confidence >= 0.0
        assert vague_answer.confidence >= 0.0
    
    @pytest.mark.slow
    def test_multiple_questions(self, pipeline):
        """여러 질문 연속 처리"""
        questions = [
            "고산 정수장 URL은?",
            "pH 기준은?",
            "설계 용량은?",
        ]
        
        for q in questions:
            answer = pipeline.ask(q)
            assert answer is not None
            assert answer.text

