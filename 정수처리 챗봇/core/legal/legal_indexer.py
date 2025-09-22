"""
법률 인덱싱 모듈

하이브리드 인덱싱(BM25 + 벡터)을 구축하고 관리합니다.
멀티뷰 인덱싱을 통해 본문, 제목/조문번호, 키프레이즈를 별도 필드로 저장합니다.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb
from chromadb.config import Settings
import faiss

from .legal_schema import LegalDocument, LegalChunk, LegalMetadata
from core.document.vector_store import HybridVectorStore

import logging
logger = logging.getLogger(__name__)

class LegalIndexer:
    """법률 문서 인덱싱 클래스"""
    
    def __init__(self, 
                 embedding_model: str = "jhgan/ko-sroberta-multitask",
                 vector_store_path: str = "vector_store/legal",
                 index_config: Optional[Dict] = None):
        """인덱서 초기화"""
        self.embedding_model_name = embedding_model
        self.vector_store_path = vector_store_path
        self.index_config = index_config or self._get_default_config()
        
        # 임베딩 모델 초기화 (지연 로딩)
        self.embedding_model = None
        self._embedding_model_loaded = False
        
        # 벡터 저장소 초기화
        self.vector_store = HybridVectorStore(
            embedding_models=[embedding_model],
            primary_model=embedding_model
        )
        
        # BM25 인덱스 초기화
        self.bm25_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=None,
            ngram_range=(1, 3),  # 법률 용어는 복합어가 많음
            analyzer='char',     # 한국어 특성 고려
            min_df=2,
            max_df=0.95
        )
        self.bm25_matrix = None
        self.bm25_chunks = []
        
        # 멀티뷰 인덱스
        self.multiview_indices = {
            'content': None,      # 본문 임베딩
            'structure': None,    # 제목/조문번호 임베딩
            'keywords': None      # 키프레이즈 임베딩
        }
        
        # 메타데이터 인덱스
        self.metadata_index = {}
        self.chunk_id_to_metadata = {}
        
        logger.info(f"법률 인덱서 초기화 완료 (모델: {embedding_model})")
    
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            'hnsw_params': {
                'M': 48,
                'ef_construction': 200,
                'ef_search': 128
            },
            'multiview_weights': {
                'content': 0.6,
                'structure': 0.25,
                'keywords': 0.15
            },
            'bm25_weight': 0.4,
            'vector_weight': 0.6
        }
    
    def _ensure_embedding_model_loaded(self):
        """임베딩 모델 지연 로딩"""
        if not self._embedding_model_loaded:
            try:
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    cache_folder="./models"
                )
                self._embedding_model_loaded = True
                logger.info(f"임베딩 모델 로드 완료: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"임베딩 모델 로드 실패: {e}")
                raise
    
    def index_legal_documents(self, documents: List[LegalDocument]) -> None:
        """법률 문서들을 인덱싱"""
        all_chunks = []
        
        for document in documents:
            all_chunks.extend(document.chunks)
            logger.info(f"문서 '{document.title}' 청크 수집: {len(document.chunks)}개")
        
        if not all_chunks:
            logger.warning("인덱싱할 청크가 없습니다.")
            return
        
        # 멀티뷰 임베딩 생성 및 인덱싱
        self._create_multiview_embeddings(all_chunks)
        
        # BM25 인덱스 구축
        self._build_bm25_index(all_chunks)
        
        # 메타데이터 인덱스 구축
        self._build_metadata_index(all_chunks)
        
        # 벡터 저장소에 추가
        self._add_to_vector_store(all_chunks)
        
        logger.info(f"법률 문서 인덱싱 완료: {len(all_chunks)}개 청크")
    
    def _create_multiview_embeddings(self, chunks: List[LegalChunk]) -> None:
        """멀티뷰 임베딩 생성"""
        self._ensure_embedding_model_loaded()
        logger.info("멀티뷰 임베딩 생성 시작...")
        
        # 각 뷰별 텍스트 준비
        content_texts = []
        structure_texts = []
        keyword_texts = []
        
        for chunk in chunks:
            # 본문 텍스트
            content_texts.append(chunk.text)
            
            # 구조 텍스트 (법률명 + 조문번호 + 계층)
            structure_parts = [chunk.metadata.law_title]
            if chunk.metadata.article_no:
                structure_parts.append(f"제{chunk.metadata.article_no}조")
            if chunk.metadata.clause_no:
                structure_parts.append(f"제{chunk.metadata.clause_no}항")
            if chunk.metadata.section_hierarchy:
                structure_parts.append(chunk.metadata.section_hierarchy)
            structure_texts.append(" ".join(structure_parts))
            
            # 키워드 텍스트
            keyword_texts.append(" ".join(chunk.keywords) if chunk.keywords else chunk.text[:100])
        
        # 임베딩 생성
        logger.info("본문 임베딩 생성...")
        content_embeddings = self.embedding_model.encode(
            content_texts, 
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        logger.info("구조 임베딩 생성...")
        structure_embeddings = self.embedding_model.encode(
            structure_texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        logger.debug("키워드 임베딩 생성...")
        keyword_embeddings = self.embedding_model.encode(
            keyword_texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        # FAISS 인덱스 생성
        dimension = content_embeddings.shape[1]
        
        # 본문 인덱스
        content_index = faiss.IndexHNSWFlat(dimension, self.index_config['hnsw_params']['M'])
        content_index.hnsw.efConstruction = self.index_config['hnsw_params']['ef_construction']
        content_index.add(content_embeddings.astype('float32'))
        self.multiview_indices['content'] = content_index
        
        # 구조 인덱스
        structure_index = faiss.IndexHNSWFlat(dimension, self.index_config['hnsw_params']['M'])
        structure_index.hnsw.efConstruction = self.index_config['hnsw_params']['ef_construction']
        structure_index.add(structure_embeddings.astype('float32'))
        self.multiview_indices['structure'] = structure_index
        
        # 키워드 인덱스
        keyword_index = faiss.IndexHNSWFlat(dimension, self.index_config['hnsw_params']['M'])
        keyword_index.hnsw.efConstruction = self.index_config['hnsw_params']['ef_construction']
        keyword_index.add(keyword_embeddings.astype('float32'))
        self.multiview_indices['keywords'] = keyword_index
        
        # 청크에 임베딩 저장
        for i, chunk in enumerate(chunks):
            chunk.embedding = {
                'content': content_embeddings[i],
                'structure': structure_embeddings[i],
                'keywords': keyword_embeddings[i]
            }
        
        logger.info("멀티뷰 임베딩 생성 완료")
    
    def _build_bm25_index(self, chunks: List[LegalChunk]) -> None:
        """BM25 인덱스 구축"""
        logger.info("BM25 인덱스 구축 시작...")
        
        # 텍스트 수집
        texts = [chunk.text for chunk in chunks]
        
        # TF-IDF 행렬 생성
        self.bm25_matrix = self.bm25_vectorizer.fit_transform(texts)
        self.bm25_chunks = chunks.copy()
        
        logger.info(f"BM25 인덱스 구축 완료: {len(texts)}개 문서, {self.bm25_matrix.shape[1]}개 특성")
    
    def _build_metadata_index(self, chunks: List[LegalChunk]) -> None:
        """메타데이터 인덱스 구축"""
        logger.info("메타데이터 인덱스 구축 시작...")
        
        # 법률별 인덱스
        law_index = {}
        # 조문별 인덱스
        article_index = {}
        # 도메인별 인덱스
        domain_index = {}
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            chunk_info = {
                'chunk_idx': i,
                'chunk': chunk
            }
            
            # 법률별 그룹화
            law_key = metadata.law_id
            if law_key not in law_index:
                law_index[law_key] = []
            law_index[law_key].append(chunk_info)
            
            # 조문별 그룹화
            if metadata.article_no:
                article_key = f"{metadata.law_id}_{metadata.article_no}"
                if article_key not in article_index:
                    article_index[article_key] = []
                article_index[article_key].append(chunk_info)
            
            # 청크 ID 매핑
            self.chunk_id_to_metadata[chunk.chunk_id] = metadata
        
        self.metadata_index = {
            'by_law': law_index,
            'by_article': article_index,
            'by_domain': domain_index
        }
        
        logger.info(f"메타데이터 인덱스 구축 완료: {len(law_index)}개 법률, {len(article_index)}개 조문")
    
    def _add_to_vector_store(self, chunks: List[LegalChunk]) -> None:
        """벡터 저장소에 청크 추가"""
        logger.info("벡터 저장소에 청크 추가...")
        
        # TextChunk 형태로 변환 (기존 벡터 저장소와 호환성)
        from core.document.pdf_processor import TextChunk
        
        text_chunks = []
        for chunk in chunks:
            text_chunk = TextChunk(
                text=chunk.text,
                metadata={
                    'law_id': chunk.metadata.law_id,
                    'law_title': chunk.metadata.law_title,
                    'article_no': chunk.metadata.article_no,
                    'clause_no': chunk.metadata.clause_no,
                    'chunk_id': chunk.chunk_id,
                    'keywords': chunk.keywords
                },
                chunk_id=chunk.chunk_id,
                source_file=chunk.metadata.source_path or '',
                page_number=0,
                keywords=chunk.keywords
            )
            text_chunks.append(text_chunk)
        
        # 벡터 저장소에 추가
        self.vector_store.add_chunks(text_chunks)
        
        logger.info(f"벡터 저장소 추가 완료: {len(text_chunks)}개 청크")
    
    def search_multiview(self, query: str, top_k: int = 50) -> List[Tuple[LegalChunk, float]]:
        """멀티뷰 검색"""
        self._ensure_embedding_model_loaded()
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        
        results = {}
        weights = self.index_config['multiview_weights']
        
        # 각 뷰에서 검색
        for view_name, index in self.multiview_indices.items():
            if index is None:
                continue
            
            # 검색 수행
            index.hnsw.efSearch = self.index_config['hnsw_params']['ef_search']
            scores, indices = index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                top_k * 2  # 충분한 후보 확보
            )
            
            # 결과 가중 적용
            weight = weights[view_name]
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # 유효하지 않은 인덱스
                    continue
                
                if idx not in results:
                    results[idx] = 0.0
                results[idx] += float(score) * weight
        
        # 상위 결과 선별
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # LegalChunk와 점수 반환
        final_results = []
        for idx, score in sorted_results:
            if idx < len(self.bm25_chunks):
                final_results.append((self.bm25_chunks[idx], score))
        
        return final_results
    
    def search_bm25(self, query: str, top_k: int = 50) -> List[Tuple[LegalChunk, float]]:
        """BM25 검색"""
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
    
    def save_index(self, path: Optional[str] = None) -> None:
        """인덱스 저장"""
        save_path = path or self.vector_store_path
        os.makedirs(save_path, exist_ok=True)
        
        # 멀티뷰 인덱스 저장
        for view_name, index in self.multiview_indices.items():
            if index is not None:
                index_path = os.path.join(save_path, f"{view_name}_index.faiss")
                faiss.write_index(index, index_path)
        
        # 메타데이터 저장
        metadata_path = os.path.join(save_path, "metadata_index.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            # 직렬화 가능한 형태로 변환
            serializable_metadata = {}
            for key, value in self.metadata_index.items():
                if key == 'by_law' or key == 'by_article':
                    serializable_metadata[key] = {}
                    for sub_key, chunks_info in value.items():
                        serializable_metadata[key][sub_key] = [
                            {
                                'chunk_idx': info['chunk_idx'],
                                'chunk_id': info['chunk'].chunk_id,
                                'metadata': asdict(info['chunk'].metadata)
                            }
                            for info in chunks_info
                        ]
            json.dump(serializable_metadata, f, ensure_ascii=False, indent=2)
        
        # 설정 저장
        config_path = os.path.join(save_path, "index_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.index_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"인덱스 저장 완료: {save_path}")
    
    def load_index(self, path: Optional[str] = None) -> None:
        """인덱스 로드"""
        load_path = path or self.vector_store_path
        
        if not os.path.exists(load_path):
            logger.warning(f"인덱스 경로가 존재하지 않습니다: {load_path}")
            return
        
        # 멀티뷰 인덱스 로드
        for view_name in self.multiview_indices.keys():
            index_path = os.path.join(load_path, f"{view_name}_index.faiss")
            if os.path.exists(index_path):
                self.multiview_indices[view_name] = faiss.read_index(index_path)
                logger.info(f"{view_name} 인덱스 로드 완료")
        
        # 메타데이터 로드
        metadata_path = os.path.join(load_path, "metadata_index.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata_index = json.load(f)
            logger.info("메타데이터 인덱스 로드 완료")
        
        # 설정 로드
        config_path = os.path.join(load_path, "index_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.index_config = json.load(f)
            logger.info("인덱스 설정 로드 완료")
        
        logger.info(f"인덱스 로드 완료: {load_path}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보 반환"""
        stats = {
            'multiview_indices': {},
            'bm25_documents': len(self.bm25_chunks) if self.bm25_chunks else 0,
            'metadata_index': {
                'laws': len(self.metadata_index.get('by_law', {})),
                'articles': len(self.metadata_index.get('by_article', {}))
            }
        }
        
        for view_name, index in self.multiview_indices.items():
            if index is not None:
                stats['multiview_indices'][view_name] = {
                    'total_vectors': index.ntotal,
                    'dimension': index.d
                }
        
        return stats
