"""
법률 검색 최적화 모듈

하이브리드 검색(벡터 + BM25)과 MMR 다양화를 통해 
법률 문서에서 관련 조문을 정확하게 검색합니다.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from .legal_schema import LegalChunk, LegalMetadata, LegalNormalizer
from .legal_indexer import LegalIndexer

import logging
logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    """검색 설정"""
    dense_k: int = 120          # Dense 검색 후보 수(축소)
    bm25_k: int = 120           # BM25 검색 후보 수(축소)
    rrf_k: int = 120            # RRF 병합 후 후보 수(축소)
    mmr_k: int = 0              # MMR 비활성화(다양성 비용 제거)
    mmr_lambda: float = 0.7     # 유지(활성 시 사용)
    rrf_constant: int = 60      # RRF 상수 (일반적으로 60 사용)
    multiview_enabled: bool = False  # 멀티뷰 검색 비활성화(속도 최적화)

@dataclass
class SearchResult:
    """검색 결과"""
    chunk: LegalChunk
    score: float
    rank: int
    search_type: str  # 'hybrid', 'vector', 'bm25'
    metadata: Optional[Dict] = None

class LegalRetriever:
    """법률 검색 최적화 클래스"""
    
    def __init__(self, 
                 indexer: LegalIndexer,
                 config: Optional[SearchConfig] = None,
                 embedding_model: Optional[SentenceTransformer] = None):
        """검색기 초기화"""
        self.indexer = indexer
        self.config = config or SearchConfig()
        self.normalizer = LegalNormalizer()
        
        # 임베딩 모델 (인덱서와 동일한 모델 사용)
        self.embedding_model = embedding_model or indexer.embedding_model
        
        logger.info(f"법률 검색기 초기화 완료 (MMR λ={self.config.mmr_lambda})")
    
    def search(self, query: str, top_k: int = 10, 
               search_type: str = "hybrid",
               filters: Optional[Dict] = None) -> List[SearchResult]:
        """법률 검색 수행"""
        # 쿼리 정규화
        normalized_query_result = self.normalizer.normalize_legal_text(query)
        normalized_query = normalized_query_result['normalized_text']
        
        logger.info(f"검색 쿼리: '{query}' -> '{normalized_query}' (타입: {search_type})")
        
        if search_type == "hybrid":
            results = self._hybrid_search(normalized_query, top_k, filters)
        elif search_type == "vector":
            results = self._vector_search(normalized_query, top_k, filters)
        elif search_type == "bm25":
            results = self._bm25_search(normalized_query, top_k, filters)
        else:
            raise ValueError(f"지원하지 않는 검색 타입: {search_type}")
        
        logger.info(f"검색 완료: {len(results)}개 결과 반환")
        return results
    
    def _hybrid_search(self, query: str, top_k: int, 
                      filters: Optional[Dict] = None) -> List[SearchResult]:
        """하이브리드 검색 (벡터 + BM25 + RRF + MMR)"""
        # 1. 벡터 검색
        if self.config.multiview_enabled:
            vector_results = self.indexer.search_multiview(query, self.config.dense_k)
        else:
            # 기본 벡터 검색 (단일 뷰)
            vector_results = self._single_view_vector_search(query, self.config.dense_k)
        
        # 2. BM25 검색
        bm25_results = self.indexer.search_bm25(query, self.config.bm25_k)
        
        # 3. RRF로 결과 병합
        rrf_results = self._reciprocal_rank_fusion(
            vector_results, bm25_results, self.config.rrf_k
        )
        
        # 4. 필터 적용
        if filters:
            rrf_results = self._apply_filters(rrf_results, filters)
        
        # 5. MMR(다양성) 비활성화 시 바로 반환, 활성화 시에만 적용
        if self.config.mmr_k and self.config.mmr_k > 0:
            final_results = self._maximal_marginal_relevance(
                query, rrf_results, min(self.config.mmr_k, len(rrf_results))
            )
        else:
            final_results = rrf_results
        
        # 6. SearchResult 객체로 변환
        search_results = []
        for i, (chunk, score) in enumerate(final_results[:top_k]):
            search_results.append(SearchResult(
                chunk=chunk,
                score=score,
                rank=i + 1,
                search_type="hybrid",
                metadata={
                    'vector_score': self._get_original_score(chunk, vector_results),
                    'bm25_score': self._get_original_score(chunk, bm25_results),
                    'rrf_score': score
                }
            ))
        
        return search_results
    
    def _vector_search(self, query: str, top_k: int,
                      filters: Optional[Dict] = None) -> List[SearchResult]:
        """벡터 검색만 수행"""
        if self.config.multiview_enabled:
            results = self.indexer.search_multiview(query, top_k * 2)
        else:
            results = self._single_view_vector_search(query, top_k * 2)
        
        # 필터 적용
        if filters:
            results = self._apply_filters(results, filters)
        
        # SearchResult 객체로 변환
        search_results = []
        for i, (chunk, score) in enumerate(results[:top_k]):
            search_results.append(SearchResult(
                chunk=chunk,
                score=score,
                rank=i + 1,
                search_type="vector"
            ))
        
        return search_results
    
    def _bm25_search(self, query: str, top_k: int,
                    filters: Optional[Dict] = None) -> List[SearchResult]:
        """BM25 검색만 수행"""
        results = self.indexer.search_bm25(query, top_k * 2)
        
        # 필터 적용
        if filters:
            results = self._apply_filters(results, filters)
        
        # SearchResult 객체로 변환
        search_results = []
        for i, (chunk, score) in enumerate(results[:top_k]):
            search_results.append(SearchResult(
                chunk=chunk,
                score=score,
                rank=i + 1,
                search_type="bm25"
            ))
        
        return search_results
    
    def _single_view_vector_search(self, query: str, top_k: int) -> List[Tuple[LegalChunk, float]]:
        """단일 뷰 벡터 검색 (fallback)"""
        # 기본적으로 content 뷰 사용
        content_index = self.indexer.multiview_indices.get('content')
        if content_index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        
        content_index.hnsw.efSearch = self.indexer.index_config['hnsw_params']['ef_search']
        scores, indices = content_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            top_k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.indexer.bm25_chunks):
                results.append((self.indexer.bm25_chunks[idx], float(score)))
        
        return results
    
    def _reciprocal_rank_fusion(self, 
                               vector_results: List[Tuple[LegalChunk, float]],
                               bm25_results: List[Tuple[LegalChunk, float]],
                               top_k: int) -> List[Tuple[LegalChunk, float]]:
        """Reciprocal Rank Fusion으로 결과 병합"""
        chunk_scores = {}
        
        # 벡터 검색 결과 처리
        for rank, (chunk, score) in enumerate(vector_results):
            chunk_id = chunk.chunk_id
            rrf_score = 1.0 / (self.config.rrf_constant + rank + 1)
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    'chunk': chunk,
                    'total_score': 0.0,
                    'vector_rank': rank + 1,
                    'bm25_rank': None
                }
            chunk_scores[chunk_id]['total_score'] += rrf_score * self.indexer.index_config.get('vector_weight', 0.7)
        
        # BM25 검색 결과 처리
        for rank, (chunk, score) in enumerate(bm25_results):
            chunk_id = chunk.chunk_id
            rrf_score = 1.0 / (self.config.rrf_constant + rank + 1)
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    'chunk': chunk,
                    'total_score': 0.0,
                    'vector_rank': None,
                    'bm25_rank': rank + 1
                }
            else:
                chunk_scores[chunk_id]['bm25_rank'] = rank + 1
            
            chunk_scores[chunk_id]['total_score'] += rrf_score * self.indexer.index_config.get('bm25_weight', 0.3)
        
        # 점수 순으로 정렬
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x['total_score'],
            reverse=True
        )
        
        # 상위 결과 반환
        return [(item['chunk'], item['total_score']) for item in sorted_results[:top_k]]
    
    def _maximal_marginal_relevance(self, 
                                   query: str,
                                   candidates: List[Tuple[LegalChunk, float]], 
                                   top_k: int) -> List[Tuple[LegalChunk, float]]:
        """Maximal Marginal Relevance로 다양화"""
        if not candidates:
            return []
        
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        
        # 후보 임베딩 수집
        candidate_embeddings = []
        for chunk, _ in candidates:
            if hasattr(chunk, 'embedding') and chunk.embedding:
                # 멀티뷰 임베딩이 있는 경우 content 임베딩 사용
                if isinstance(chunk.embedding, dict):
                    embedding = chunk.embedding.get('content', chunk.embedding.get('structure', None))
                else:
                    embedding = chunk.embedding
            else:
                # 임베딩이 없으면 실시간 생성
                embedding = self.embedding_model.encode([chunk.text], normalize_embeddings=True)[0]
            
            candidate_embeddings.append(embedding)
        
        candidate_embeddings = np.array(candidate_embeddings)
        
        # MMR 알고리즘
        selected = []
        remaining_indices = list(range(len(candidates)))
        
        while len(selected) < top_k and remaining_indices:
            mmr_scores = []
            
            for i in remaining_indices:
                chunk, relevance_score = candidates[i]
                
                # 관련성 점수 (쿼리와의 유사도)
                relevance = float(np.dot(query_embedding, candidate_embeddings[i]))
                
                # 다양성 점수 (이미 선택된 문서들과의 최대 유사도)
                if selected:
                    selected_embeddings = candidate_embeddings[[idx for idx, _ in selected]]
                    diversity = float(np.max(np.dot(selected_embeddings, candidate_embeddings[i])))
                else:
                    diversity = 0.0
                
                # MMR 점수 계산
                mmr_score = (self.config.mmr_lambda * relevance - 
                           (1 - self.config.mmr_lambda) * diversity)
                mmr_scores.append((i, mmr_score))
            
            # 최고 점수 선택
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected.append((best_idx, best_score))
            remaining_indices.remove(best_idx)
        
        # 결과 반환
        return [(candidates[idx][0], score) for idx, score in selected]
    
    def _apply_filters(self, 
                      results: List[Tuple[LegalChunk, float]],
                      filters: Dict) -> List[Tuple[LegalChunk, float]]:
        """필터 적용"""
        filtered_results = []
        
        for chunk, score in results:
            metadata = chunk.metadata
            include = True
            
            # 법률 ID 필터
            if 'law_ids' in filters:
                if metadata.law_id not in filters['law_ids']:
                    include = False
            
            # 법률명 필터
            if 'law_titles' in filters:
                if not any(title.lower() in metadata.law_title.lower() 
                          for title in filters['law_titles']):
                    include = False
            
            # 조문 번호 필터
            if 'article_nos' in filters:
                if metadata.article_no not in filters['article_nos']:
                    include = False
            
            # 시행일 필터
            if 'effective_date_after' in filters:
                if (metadata.effective_date and 
                    metadata.effective_date < filters['effective_date_after']):
                    include = False
            
            # 도메인 필터
            if 'domains' in filters:
                chunk_domains = self.normalizer.classify_legal_domain(chunk.text)
                if not any(domain in chunk_domains for domain in filters['domains']):
                    include = False
            
            if include:
                filtered_results.append((chunk, score))
        
        return filtered_results
    
    def _get_original_score(self, 
                           target_chunk: LegalChunk,
                           results: List[Tuple[LegalChunk, float]]) -> Optional[float]:
        """원본 점수 조회"""
        for chunk, score in results:
            if chunk.chunk_id == target_chunk.chunk_id:
                return score
        return None
    
    def search_similar_articles(self, 
                               article_id: str, 
                               top_k: int = 10) -> List[SearchResult]:
        """유사한 조문 검색"""
        # 메타데이터 인덱스에서 해당 조문 찾기
        article_chunks = self.indexer.metadata_index.get('by_article', {}).get(article_id, [])
        
        if not article_chunks:
            logger.warning(f"조문을 찾을 수 없습니다: {article_id}")
            return []
        
        # 첫 번째 청크의 텍스트를 쿼리로 사용
        query_chunk = article_chunks[0]['chunk']
        query_text = query_chunk.text
        
        # 유사 검색 수행
        results = self.search(query_text, top_k * 2, search_type="vector")
        
        # 자기 자신 제외
        filtered_results = [
            result for result in results 
            if not result.chunk.chunk_id.startswith(article_id)
        ]
        
        return filtered_results[:top_k]
    
    def search_by_law(self, law_id: str, query: str, top_k: int = 10) -> List[SearchResult]:
        """특정 법률 내에서 검색"""
        filters = {'law_ids': [law_id]}
        return self.search(query, top_k, search_type="hybrid", filters=filters)
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 정보"""
        indexer_stats = self.indexer.get_index_stats()
        
        return {
            'config': {
                'dense_k': self.config.dense_k,
                'bm25_k': self.config.bm25_k,
                'mmr_lambda': self.config.mmr_lambda,
                'multiview_enabled': self.config.multiview_enabled
            },
            'indexer_stats': indexer_stats
        }
