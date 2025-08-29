"""
고속 벡터 저장소 모듈

전처리된 데이터를 사용하여 매우 빠른 검색을 제공합니다.
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Tuple, Optional
import logging

from core.pdf_processor import TextChunk
from core.pdf_preprocessor import PDFDatabase

logger = logging.getLogger(__name__)

class FastVectorStore:
    """고속 벡터 검색을 위한 최적화된 저장소"""
    
    def __init__(self, embedding_dimension: int = 768):
        """
        벡터 저장소 초기화
        
        Args:
            embedding_dimension: 임베딩 차원
        """
        self.embedding_dimension = embedding_dimension
        self.faiss_index = None
        self.chunks = []
        self.is_loaded = False
        
        # FAISS 인덱스 생성 (빠른 검색을 위한 IVF + PQ)
        self._init_faiss_index()
        
        logger.info(f"FastVectorStore 초기화 (차원: {embedding_dimension})")
    
    def _init_faiss_index(self):
        """FAISS 인덱스 초기화"""
        # 빠른 검색을 위한 IVF (Inverted File) 인덱스 사용
        # 클러스터 수는 데이터 크기에 따라 조정
        nlist = 100  # 클러스터 수
        
        # L2 거리 기반 인덱스
        quantizer = faiss.IndexFlatL2(self.embedding_dimension)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
        
        # GPU 사용 가능한 경우 GPU 활용
        if faiss.get_num_gpus() > 0:
            try:
                self.faiss_index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 0, self.faiss_index
                )
                logger.info("FAISS GPU 가속 활성화")
            except Exception as e:
                logger.warning(f"FAISS GPU 활성화 실패, CPU 사용: {e}")
    
    def load_from_database(self, database: PDFDatabase) -> bool:
        """
        데이터베이스에서 전처리된 데이터 로드
        
        Args:
            database: PDF 데이터베이스
            
        Returns:
            로드 성공 여부
        """
        try:
            logger.info("데이터베이스에서 벡터 데이터 로드 시작...")
            
            # 모든 청크 로드
            self.chunks = database.load_all_chunks()
            
            if not self.chunks:
                logger.warning("로드할 벡터 데이터가 없습니다")
                return False
            
            # 임베딩 벡터 추출
            embeddings = []
            valid_chunks = []
            
            for chunk in self.chunks:
                if chunk.embedding is not None:
                    embeddings.append(chunk.embedding)
                    valid_chunks.append(chunk)
            
            if not embeddings:
                logger.warning("유효한 임베딩이 없습니다")
                return False
            
            self.chunks = valid_chunks
            embeddings_array = np.vstack(embeddings).astype('float32')
            
            logger.info(f"임베딩 로드 완료: {len(embeddings)}개")
            
            # FAISS 인덱스 훈련 및 추가
            if not self.faiss_index.is_trained:
                logger.info("FAISS 인덱스 훈련 중...")
                self.faiss_index.train(embeddings_array)
            
            self.faiss_index.add(embeddings_array)
            self.is_loaded = True
            
            logger.info(f"벡터 저장소 로드 완료: {len(self.chunks)}개 청크")
            return True
            
        except Exception as e:
            logger.error(f"벡터 저장소 로드 실패: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 5, 
               score_threshold: float = 0.0) -> List[Tuple[TextChunk, float]]:
        """
        벡터 검색
        
        Args:
            query_embedding: 질문 임베딩
            top_k: 반환할 결과 수
            score_threshold: 최소 점수 임계값
            
        Returns:
            검색 결과 (청크, 점수) 리스트
        """
        if not self.is_loaded or len(self.chunks) == 0:
            logger.warning("벡터 저장소가 로드되지 않았습니다")
            return []
        
        try:
            # 쿼리 임베딩 전처리
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            
            # FAISS 검색
            scores, indices = self.faiss_index.search(query_vector, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks):
                    # FAISS는 L2 거리를 반환하므로 코사인 유사도로 변환
                    # L2 거리가 작을수록 유사함 (0에 가까울수록 유사)
                    # 정규화된 벡터에서 L2 거리 -> 코사인 유사도 변환
                    # similarity = 1 - (L2_distance^2 / 2)
                    # 하지만 실제로는 더 간단한 공식 사용
                    similarity = max(0.0, 1.0 - (score / 4.0))  # 점수를 0-1 범위로 정규화
                    
                    if similarity >= score_threshold:
                        chunk = self.chunks[idx]
                        results.append((chunk, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []
    
    def get_statistics(self) -> dict:
        """저장소 통계 정보"""
        return {
            "total_chunks": len(self.chunks),
            "is_loaded": self.is_loaded,
            "index_trained": self.faiss_index.is_trained if self.faiss_index else False,
            "embedding_dimension": self.embedding_dimension
        }
    
    def save_cache(self, cache_path: str = "./data/vector_cache"):
        """벡터 캐시 저장"""
        try:
            os.makedirs(cache_path, exist_ok=True)
            
            # FAISS 인덱스 저장
            faiss_path = os.path.join(cache_path, "faiss_index.bin")
            faiss.write_index(self.faiss_index, faiss_path)
            
            # 청크 정보 저장
            chunks_path = os.path.join(cache_path, "chunks.pkl")
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            logger.info(f"벡터 캐시 저장 완료: {cache_path}")
            
        except Exception as e:
            logger.error(f"벡터 캐시 저장 실패: {e}")
    
    def load_cache(self, cache_path: str = "./data/vector_cache") -> bool:
        """벡터 캐시 로드"""
        try:
            faiss_path = os.path.join(cache_path, "faiss_index.bin")
            chunks_path = os.path.join(cache_path, "chunks.pkl")
            
            if not (os.path.exists(faiss_path) and os.path.exists(chunks_path)):
                return False
            
            # FAISS 인덱스 로드
            self.faiss_index = faiss.read_index(faiss_path)
            
            # 청크 로드
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            self.is_loaded = True
            logger.info(f"벡터 캐시 로드 완료: {len(self.chunks)}개 청크")
            return True
            
        except Exception as e:
            logger.error(f"벡터 캐시 로드 실패: {e}")
            return False
