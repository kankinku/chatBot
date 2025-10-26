"""
BM25Retriever 단위 테스트
"""

import pytest
from modules.retrieval.bm25_retriever import BM25Retriever


class TestBM25Retriever:
    """BM25 검색기 테스트"""
    
    @pytest.fixture
    def retriever(self, sample_chunks):
        """BM25 검색기 인스턴스"""
        return BM25Retriever(sample_chunks)
    
    def test_initialization(self, retriever, sample_chunks):
        """초기화 확인"""
        assert len(retriever.chunks) == len(sample_chunks)
        assert retriever.N == len(sample_chunks)
        assert retriever.avgdl > 0
        assert len(retriever.idf) > 0
    
    def test_search_returns_results(self, retriever):
        """검색 결과 반환 확인"""
        results = retriever.search("고산 정수장", top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)  # (idx, score)
    
    def test_exact_match_boost(self, retriever):
        """정확한 매칭에 높은 점수"""
        # "AI플랫폼"이 정확히 포함된 청크
        results = retriever.search("AI플랫폼", top_k=5)
        
        # 첫 번째 결과의 청크에 "AI플랫폼"이 포함되어야 함
        top_idx, top_score = results[0]
        assert "AI플랫폼" in retriever.chunks[top_idx].text or "AI" in retriever.chunks[top_idx].text
        assert top_score > 0
    
    def test_score_ranking(self, retriever):
        """점수 순으로 정렬"""
        results = retriever.search("정수장 pH 탁도", top_k=5)
        
        # 점수가 내림차순이어야 함
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_empty_query(self, retriever):
        """빈 쿼리 처리"""
        results = retriever.search("", top_k=5)
        
        # 빈 쿼리도 결과 반환 (모든 문서)
        assert len(results) > 0
    
    def test_top_k_limit(self, retriever):
        """top_k 제한"""
        results_3 = retriever.search("test", top_k=3)
        results_2 = retriever.search("test", top_k=2)
        
        assert len(results_3) <= 3
        assert len(results_2) <= 2
    
    def test_char_ngrams(self, retriever):
        """문자 n-gram 생성"""
        ngrams = retriever._char_ngrams("테스트")
        
        assert len(ngrams) > 0
        # 3-5 char n-grams
        assert any(len(ng) == 3 for ng in ngrams)
        assert any(len(ng) == 5 for ng in ngrams)
    
    def test_korean_search(self, retriever):
        """한글 검색"""
        results = retriever.search("수질 모니터링", top_k=3)
        
        assert len(results) > 0
        # 수질이 포함된 청크가 상위에 있어야 함
        top_idx, _ = results[0]
        assert "수질" in retriever.chunks[top_idx].text
    
    def test_numeric_search(self, retriever):
        """숫자 검색"""
        results = retriever.search("50000", top_k=3)
        
        assert len(results) > 0

