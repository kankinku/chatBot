"""
PDF 전용 하이브리드 검색 모듈

법률 파이프라인의 최적화 기법을 PDF RAG에 적용:
- 하이브리드 검색 (벡터 + BM25 + RRF)
- 멀티뷰 인덱싱 (본문, 제목, 키워드, 구조)
- 성능 최적화 설정
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .pdf_processor import TextChunk
from .vector_store import HybridVectorStore

import logging
logger = logging.getLogger(__name__)

@dataclass
class PDFSearchConfig:
    """PDF 검색 설정"""
    dense_k: int = 120          # 벡터 검색 후보 수
    bm25_k: int = 120           # BM25 검색 후보 수
    rrf_k: int = 120            # RRF 병합 후 후보 수
    mmr_k: int = 0              # MMR 비활성화 (정확도 우선)
    rrf_constant: int = 60      # RRF 상수
    multiview_enabled: bool = True  # 멀티뷰 검색 사용
    similarity_threshold: float = 0.3  # 유사도 임계값

@dataclass
class PDFSearchResult:
    """PDF 검색 결과"""
    chunk: TextChunk
    score: float
    rank: int
    search_type: str  # 'hybrid', 'vector', 'bm25'
    metadata: Optional[Dict] = None

class PDFRetriever:
    """PDF 전용 하이브리드 검색기"""
    
    def __init__(self, 
                 embedding_model: str = "jhgan/ko-sroberta-multitask",
                 config: Optional[PDFSearchConfig] = None,
                 vector_store: Optional[HybridVectorStore] = None):
        """PDF 검색기 초기화"""
        self.config = config or PDFSearchConfig()
        self.embedding_model_name = embedding_model
        
        # 임베딩 모델 (지연 로딩)
        self.embedding_model = None
        self._embedding_model_loaded = False
        
        # 벡터 저장소
        self.vector_store = vector_store or HybridVectorStore(
            embedding_models=[embedding_model],
            primary_model=embedding_model
        )
        
        # BM25 인덱스
        self.bm25_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=None,
            ngram_range=(1, 3),  # PDF 용어 특성 고려
            analyzer='char',     # 한국어 최적화
            min_df=2,
            max_df=0.95
        )
        self.bm25_matrix = None
        self.bm25_chunks = []
        
        # 멀티뷰 인덱스 (PDF 특화)
        self.multiview_indices = {
            'content': None,     # 본문 내용
            'title': None,       # 제목/헤더
            'keywords': None,    # 키워드
            'structure': None    # 구조 정보 (페이지, 섹션)
        }
        
        # 멀티뷰 가중치 (PDF 특화)
        self.multiview_weights = {
            'content': 0.5,      # 본문
            'title': 0.3,        # 제목/헤더
            'keywords': 0.15,    # 키워드
            'structure': 0.05    # 구조 정보
        }
        
        logger.info(f"PDF 검색기 초기화 완료 (멀티뷰: {self.config.multiview_enabled})")
    
    def _ensure_embedding_model_loaded(self):
        """임베딩 모델 지연 로딩"""
        if not self._embedding_model_loaded:
            try:
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    cache_folder="./models"
                )
                self._embedding_model_loaded = True
                logger.info(f"PDF 검색용 임베딩 모델 로드: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"임베딩 모델 로드 실패: {e}")
                raise
    
    def index_pdf_chunks(self, chunks: List[TextChunk]) -> None:
        """PDF 청크들을 인덱싱"""
        if not chunks:
            logger.warning("인덱싱할 청크가 없습니다.")
            return
        
        logger.info(f"PDF 청크 인덱싱 시작: {len(chunks)}개")
        
        # 1. 멀티뷰 임베딩 생성
        if self.config.multiview_enabled:
            self._create_multiview_embeddings(chunks)
        
        # 2. BM25 인덱스 구축
        self._build_bm25_index(chunks)
        
        # 3. 벡터 저장소에 추가
        self.vector_store.add_chunks(chunks)
        
        logger.info("PDF 청크 인덱싱 완료")
    
    def _create_multiview_embeddings(self, chunks: List[TextChunk]) -> None:
        """PDF 멀티뷰 임베딩 생성"""
        self._ensure_embedding_model_loaded()
        logger.info("PDF 멀티뷰 임베딩 생성 시작...")
        
        # 각 뷰별 텍스트 준비
        content_texts = []
        title_texts = []
        keyword_texts = []
        structure_texts = []
        
        for chunk in chunks:
            # 본문 텍스트
            content_texts.append(chunk.content)
            
            # 제목 텍스트 (메타데이터에서 추출)
            title = chunk.metadata.get('title', '') if chunk.metadata else ''
            if not title:
                # 첫 문장을 제목으로 사용
                title = chunk.content.split('.')[0][:100]
            title_texts.append(title)
            
            # 키워드 텍스트
            keywords = chunk.metadata.get('keywords', []) if chunk.metadata else []
            if not keywords:
                keywords = chunk.keywords if hasattr(chunk, 'keywords') else []
            keyword_texts.append(' '.join(keywords) if keywords else chunk.content[:100])
            
            # 구조 텍스트 (페이지, 섹션 정보)
            structure_parts = []
            if hasattr(chunk, 'page_number'):
                structure_parts.append(f"페이지 {chunk.page_number}")
            if chunk.metadata:
                if 'section' in chunk.metadata:
                    structure_parts.append(chunk.metadata['section'])
                if 'subsection' in chunk.metadata:
                    structure_parts.append(chunk.metadata['subsection'])
            structure_texts.append(' '.join(structure_parts) if structure_parts else f"페이지 {getattr(chunk, 'page_number', 1)}")
        
        # 임베딩 생성 (배치 처리)
        batch_size = 32
        
        logger.info("본문 임베딩 생성...")
        content_embeddings = self.embedding_model.encode(
            content_texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        logger.info("제목 임베딩 생성...")
        title_embeddings = self.embedding_model.encode(
            title_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        logger.debug("키워드 임베딩 생성...")
        keyword_embeddings = self.embedding_model.encode(
            keyword_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        logger.info("구조 임베딩 생성...")
        structure_embeddings = self.embedding_model.encode(
            structure_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # FAISS 인덱스 생성
        dimension = content_embeddings.shape[1]
        
        import faiss
        
        # 각 뷰별 인덱스 생성
        for view_name, embeddings in [
            ('content', content_embeddings),
            ('title', title_embeddings),
            ('keywords', keyword_embeddings),
            ('structure', structure_embeddings)
        ]:
            index = faiss.IndexFlatIP(dimension)  # 내적 기반 인덱스
            index.add(embeddings.astype('float32'))
            self.multiview_indices[view_name] = index
        
        # 청크에 임베딩 저장
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'multiview_embeddings'):
                chunk.multiview_embeddings = {}
            chunk.multiview_embeddings = {
                'content': content_embeddings[i],
                'title': title_embeddings[i],
                'keywords': keyword_embeddings[i],
                'structure': structure_embeddings[i]
            }
        
        logger.info("PDF 멀티뷰 임베딩 생성 완료")
    
    def _build_bm25_index(self, chunks: List[TextChunk]) -> None:
        """BM25 인덱스 구축"""
        logger.info("PDF BM25 인덱스 구축 시작...")
        
        # 텍스트 수집 (본문 + 제목 + 키워드 결합)
        texts = []
        for chunk in chunks:
            text_parts = [chunk.content]
            
            # 제목 추가
            if chunk.metadata and 'title' in chunk.metadata:
                text_parts.append(chunk.metadata['title'])
            
            # 키워드 추가
            keywords = chunk.metadata.get('keywords', []) if chunk.metadata else []
            if not keywords and hasattr(chunk, 'keywords'):
                keywords = chunk.keywords
            if keywords:
                text_parts.extend(keywords)
            
            texts.append(' '.join(text_parts))
        
        # TF-IDF 행렬 생성
        self.bm25_matrix = self.bm25_vectorizer.fit_transform(texts)
        self.bm25_chunks = chunks.copy()
        
        logger.info(f"PDF BM25 인덱스 구축 완료: {len(texts)}개 문서, {self.bm25_matrix.shape[1]}개 특성")
    
    def search(self, query: str, top_k: int = 10, 
               search_type: str = "hybrid") -> List[PDFSearchResult]:
        """PDF 검색 수행"""
        logger.info(f"PDF 검색: '{query}' (타입: {search_type}, top_k: {top_k})")
        
        if search_type == "hybrid":
            results = self._hybrid_search(query, top_k)
        elif search_type == "vector":
            results = self._vector_search(query, top_k)
        elif search_type == "bm25":
            results = self._bm25_search(query, top_k)
        else:
            raise ValueError(f"지원하지 않는 검색 타입: {search_type}")
        
        logger.info(f"PDF 검색 완료: {len(results)}개 결과 반환")
        return results
    
    def _hybrid_search(self, query: str, top_k: int) -> List[PDFSearchResult]:
        """하이브리드 검색 (벡터 + BM25 + RRF)"""
        search_start = time.time()
        
        # 1. 벡터 검색
        if self.config.multiview_enabled:
            vector_results = self._multiview_search(query, self.config.dense_k)
        else:
            vector_results = self._single_view_vector_search(query, self.config.dense_k)
        
        # 2. BM25 검색
        bm25_results = self._bm25_search_internal(query, self.config.bm25_k)
        
        # 3. RRF로 결과 병합
        rrf_results = self._reciprocal_rank_fusion(
            vector_results, bm25_results, self.config.rrf_k
        )
        
        # 4. 유사도 임계값 필터링
        filtered_results = [
            (chunk, score) for chunk, score in rrf_results 
            if score >= self.config.similarity_threshold
        ]
        
        # 5. SearchResult 객체로 변환
        search_results = []
        for i, (chunk, score) in enumerate(filtered_results[:top_k]):
            search_results.append(PDFSearchResult(
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
        
        search_time = time.time() - search_start
        logger.info(f"하이브리드 검색 완료: {search_time:.3f}초")
        
        return search_results
    
    def _multiview_search(self, query: str, top_k: int) -> List[Tuple[TextChunk, float]]:
        """멀티뷰 검색"""
        self._ensure_embedding_model_loaded()
        
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        
        results = {}
        
        # 각 뷰에서 검색
        for view_name, index in self.multiview_indices.items():
            if index is None:
                continue
            
            # 검색 수행
            scores, indices = index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                top_k * 2  # 충분한 후보 확보
            )
            
            # 결과 가중 적용
            weight = self.multiview_weights[view_name]
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # 유효하지 않은 인덱스
                    continue
                
                if idx not in results:
                    results[idx] = 0.0
                results[idx] += float(score) * weight
        
        # 상위 결과 선별
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # TextChunk와 점수 반환
        final_results = []
        for idx, score in sorted_results:
            if idx < len(self.bm25_chunks):
                final_results.append((self.bm25_chunks[idx], score))
        
        return final_results
    
    def _single_view_vector_search(self, query: str, top_k: int) -> List[Tuple[TextChunk, float]]:
        """단일 뷰 벡터 검색 (fallback)"""
        # 벡터 저장소를 통한 검색
        self._ensure_embedding_model_loaded()
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        
        results = self.vector_store.search(
            query_embedding, 
            top_k,
            similarity_threshold=self.config.similarity_threshold
        )
        
        return results
    
    def _bm25_search_internal(self, query: str, top_k: int) -> List[Tuple[TextChunk, float]]:
        """BM25 검색 (내부용)"""
        if self.bm25_matrix is None:
            return []
        
        # 쿼리 벡터화
        query_vec = self.bm25_vectorizer.transform([query])
        
        # 코사인 유사도 계산
        scores = (self.bm25_matrix * query_vec.T).toarray().flatten()
        
        # 상위 결과 선별
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 0점 제외
                results.append((self.bm25_chunks[idx], float(scores[idx])))
        
        return results
    
    def _bm25_search(self, query: str, top_k: int) -> List[PDFSearchResult]:
        """BM25 검색 (외부 인터페이스)"""
        results = self._bm25_search_internal(query, top_k)
        
        search_results = []
        for i, (chunk, score) in enumerate(results[:top_k]):
            search_results.append(PDFSearchResult(
                chunk=chunk,
                score=score,
                rank=i + 1,
                search_type="bm25"
            ))
        
        return search_results
    
    def _vector_search(self, query: str, top_k: int) -> List[PDFSearchResult]:
        """벡터 검색 (외부 인터페이스)"""
        if self.config.multiview_enabled:
            results = self._multiview_search(query, top_k * 2)
        else:
            results = self._single_view_vector_search(query, top_k * 2)
        
        search_results = []
        for i, (chunk, score) in enumerate(results[:top_k]):
            search_results.append(PDFSearchResult(
                chunk=chunk,
                score=score,
                rank=i + 1,
                search_type="vector"
            ))
        
        return search_results
    
    def _reciprocal_rank_fusion(self, 
                               vector_results: List[Tuple[TextChunk, float]],
                               bm25_results: List[Tuple[TextChunk, float]],
                               top_k: int) -> List[Tuple[TextChunk, float]]:
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
            chunk_scores[chunk_id]['total_score'] += rrf_score * 0.6  # 벡터 가중치
        
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
            
            chunk_scores[chunk_id]['total_score'] += rrf_score * 0.4  # BM25 가중치
        
        # 점수 순으로 정렬
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x['total_score'],
            reverse=True
        )
        
        # 상위 결과 반환
        return [(item['chunk'], item['total_score']) for item in sorted_results[:top_k]]
    
    def _get_original_score(self, 
                           target_chunk: TextChunk,
                           results: List[Tuple[TextChunk, float]]) -> Optional[float]:
        """원본 점수 조회"""
        for chunk, score in results:
            if chunk.chunk_id == target_chunk.chunk_id:
                return score
        return None
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 정보"""
        return {
            'config': {
                'dense_k': self.config.dense_k,
                'bm25_k': self.config.bm25_k,
                'multiview_enabled': self.config.multiview_enabled,
                'similarity_threshold': self.config.similarity_threshold
            },
            'bm25_documents': len(self.bm25_chunks) if self.bm25_chunks else 0,
            'multiview_indices': {
                view_name: index.ntotal if index is not None else 0
                for view_name, index in self.multiview_indices.items()
            }
        }

