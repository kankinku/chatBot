"""
QuestionAnalyzer 단위 테스트
"""

import pytest
from modules.analysis.question_analyzer import QuestionAnalyzer


class TestQuestionAnalyzer:
    """질문 분석기 테스트"""
    
    @pytest.fixture
    def analyzer(self, domain_dict_path):
        """분석기 인스턴스"""
        return QuestionAnalyzer(domain_dict_path=domain_dict_path)
    
    def test_numeric_question_detection(self, analyzer):
        """숫자가 포함된 질문을 numeric으로 분류"""
        result = analyzer.analyze("pH가 7.5일 때 처리방법은?")
        
        assert result.qtype == "numeric"
        assert result.has_number is True
    
    def test_system_info_question_detection(self, analyzer):
        """시스템 정보 질문을 올바르게 분류"""
        result = analyzer.analyze("AI 플랫폼 URL은?")
        
        assert result.qtype == "system_info"
    
    def test_definition_question_detection(self, analyzer):
        """정의 질문 감지"""
        result = analyzer.analyze("고산 정수장이란 무엇인가?")
        
        assert result.qtype == "definition"
    
    def test_procedural_question_detection(self, analyzer):
        """절차 질문 감지"""
        result = analyzer.analyze("로그인하는 방법은?")
        
        assert result.qtype == "procedural"
    
    def test_weights_for_numeric_questions(self, analyzer):
        """숫자 질문은 BM25 가중치가 더 높아야 함"""
        result = analyzer.analyze("온도가 25℃일 때?")
        
        assert result.qtype == "numeric"
        assert result.rrf_bm25_weight > result.rrf_vector_weight
    
    def test_weights_for_semantic_questions(self, analyzer):
        """의미론적 질문은 Vector 가중치가 더 높아야 함"""
        result = analyzer.analyze("정수 처리의 목적은?")
        
        assert result.qtype in ["definition", "general"]
        assert result.rrf_vector_weight >= result.rrf_bm25_weight
    
    def test_cache_effectiveness(self, analyzer):
        """캐싱이 정상 작동하는지 확인"""
        question = "테스트 질문입니다"
        
        # 첫 호출
        result1 = analyzer.analyze(question)
        
        # 캐시된 호출 (동일한 결과)
        result2 = analyzer.analyze(question)
        
        assert result1.qtype == result2.qtype
        assert result1.length == result2.length
        assert result1.rrf_vector_weight == result2.rrf_vector_weight
    
    def test_unit_detection(self, analyzer):
        """단위 포함 감지"""
        result = analyzer.analyze("PAC 주입률은 20 mg/L입니다")
        
        assert result.has_unit is True
        assert result.has_number is True
    
    def test_domain_keyword_detection(self, analyzer):
        """도메인 키워드 감지"""
        result = analyzer.analyze("AI 플랫폼 성능은?")
        
        assert result.has_domain_keyword is True
    
    def test_threshold_adjustment(self, analyzer):
        """질문 유형별 임계값 조정"""
        system_result = analyzer.analyze("시스템 URL은?")
        general_result = analyzer.analyze("일반적인 질문")
        
        # system_info는 임계값이 더 낮아야 함
        assert system_result.threshold_adj < general_result.threshold_adj
    
    @pytest.mark.parametrize("question,expected_type", [
        ("고산 정수장이란?", "definition"),
        ("처리 절차는?", "procedural"),
        ("A와 B 중 어느 것이 더 좋은가?", "comparative"),
        ("오류가 발생했습니다", "problem"),
        ("운영 현황은?", "operational"),
    ])
    def test_various_question_types(self, analyzer, question, expected_type):
        """다양한 질문 유형 분류"""
        result = analyzer.analyze(question)
        assert result.qtype == expected_type

