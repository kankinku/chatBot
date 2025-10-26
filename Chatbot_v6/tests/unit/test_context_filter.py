"""
ContextFilter 단위 테스트
"""

import pytest
from config.pipeline_config import PipelineConfig
from modules.filtering.context_filter import ContextFilter


class TestContextFilter:
    """컨텍스트 필터 테스트"""
    
    @pytest.fixture
    def filter(self):
        """필터 인스턴스"""
        config = PipelineConfig()
        return ContextFilter(config)
    
    def test_filter_empty_spans(self, filter):
        """빈 span 리스트 처리"""
        filtered, stats = filter.filter_and_calibrate([], "test question")
        
        assert len(filtered) == 0
        assert stats["filter_in"] == 0
        assert stats["filter_out"] == 0
    
    def test_filter_by_threshold(self, filter, sample_retrieved_spans):
        """임계값 기준 필터링"""
        question = "고산 정수장 AI플랫폼"
        
        # 높은 임계값 (많이 필터링)
        filtered_high, _ = filter.filter_and_calibrate(
            sample_retrieved_spans, 
            question, 
            threshold_override=0.8
        )
        
        # 낮은 임계값 (적게 필터링)
        filtered_low, _ = filter.filter_and_calibrate(
            sample_retrieved_spans,
            question,
            threshold_override=0.3
        )
        
        assert len(filtered_high) <= len(filtered_low)
    
    def test_calibration(self, filter, sample_retrieved_spans):
        """점수 캘리브레이션 확인"""
        question = "테스트 질문"
        filtered, _ = filter.filter_and_calibrate(
            sample_retrieved_spans,
            question,
            threshold_override=0.0  # 모든 span 통과
        )
        
        # 모든 span에 calibrated_conf가 설정되어야 함
        for span in filtered:
            assert span.calibrated_conf is not None
            assert 0.0 <= span.calibrated_conf <= 1.0
    
    def test_overlap_calculation(self, filter):
        """오버랩 계산"""
        query = "고산 정수장 AI플랫폼"
        context = "고산 정수장의 AI플랫폼 시스템입니다"
        
        overlap = filter._calculate_overlap(query, context)
        
        assert 0.0 <= overlap <= 1.0
        assert overlap > 0.5  # 높은 오버랩
    
    def test_keyword_filter(self, filter, sample_retrieved_spans):
        """키워드 필터링"""
        # 키워드가 없는 질문
        question = "xyz123 테스트"
        
        filtered, stats = filter.filter_and_calibrate(
            sample_retrieved_spans,
            question,
            threshold_override=0.0
        )
        
        # 키워드가 맞지 않으면 일부 필터링됨
        assert len(filtered) <= len(sample_retrieved_spans)
    
    def test_diversification(self, filter, sample_chunks):
        """다양성 확보 (중복 제거)"""
        from modules.core.types import RetrievedSpan
        
        # 같은 청크를 여러 번 포함
        spans = [
            RetrievedSpan(chunk=sample_chunks[0], source="test", score=0.9, rank=1),
            RetrievedSpan(chunk=sample_chunks[0], source="test", score=0.8, rank=2),  # 중복
            RetrievedSpan(chunk=sample_chunks[1], source="test", score=0.7, rank=3),
        ]
        
        diversified = filter._diversify(spans)
        
        # 중복이 제거되어야 함
        assert len(diversified) == 2
        assert diversified[0].chunk == sample_chunks[0]
        assert diversified[1].chunk == sample_chunks[1]
    
    def test_stats_collection(self, filter, sample_retrieved_spans):
        """통계 수집"""
        filtered, stats = filter.filter_and_calibrate(
            sample_retrieved_spans,
            "테스트",
            threshold_override=0.5
        )
        
        assert "filter_in" in stats
        assert "filter_out" in stats
        assert "filter_pass_rate" in stats
        assert stats["filter_in"] >= stats["filter_out"]
        assert 0.0 <= stats["filter_pass_rate"] <= 1.0

