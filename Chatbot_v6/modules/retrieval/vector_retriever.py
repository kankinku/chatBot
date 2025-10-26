"""
Vector Retriever - 벡터 검색기

임베딩 벡터를 사용한 의미적 검색 (단일 책임).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from modules.core.types import Chunk
from modules.core.logger import get_logger
from modules.core.exceptions import VectorStoreNotFoundError, EmbeddingError
from modules.embedding.base_embedder import BaseEmbedder

logger = get_logger(__name__)


class VectorRetriever:
    """
    벡터 검색기
    
    단일 책임: 벡터 유사도 기반 검색만 수행
    """
    
    def __init__(
        self,
        chunks: List[Chunk],
        embedder: BaseEmbedder,
        index_dir: Optional[str] = None,
        backend: str = "faiss",  # "faiss" or "simple"
        use_gpu: bool = False,  # GPU 가속
    ):
        """
        Args:
            chunks: 청크 리스트
            embedder: 임베더
            index_dir: 인덱스 디렉토리 (FAISS 사용 시)
            backend: 백엔드 ("faiss" or "simple")
            use_gpu: GPU 가속 사용 여부
        """
        self.chunks = chunks
        self.embedder = embedder
        self.index_dir = index_dir
        self.backend = backend
        self.use_gpu = use_gpu
        
        logger.info("VectorRetriever initializing",
                   num_chunks=len(chunks),
                   backend=backend,
                   embedding_dim=embedder.dim)
        
        # 인덱스 구축
        self._build_index()
        
        logger.info("VectorRetriever initialized")
    
    def _build_index(self) -> None:
        """벡터 인덱스 구축"""
        if self.backend == "faiss":
            self._build_faiss_index()
        else:
            self._build_simple_index()
    
    def _build_simple_index(self) -> None:
        """간단한 numpy 기반 인덱스"""
        logger.info("Building simple numpy index")
        
        # 모든 청크 임베딩
        texts = [chunk.text for chunk in self.chunks]
        
        try:
            self.vectors = self.embedder.embed_texts(texts)
            
            # 🚀 최적화 1A: 벡터를 정규화하여 저장 (norm 계산 불필요!)
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self.vectors = self.vectors / (norms + 1e-9)
            # 이제 self.vectors는 이미 정규화되어 있음 (norm = 1)
            
            logger.info(f"Simple index built (normalized vectors)", shape=self.vectors.shape)
        
        except Exception as e:
            raise EmbeddingError(
                "Failed to build simple vector index",
                cause=e
            ) from e
    
    def _build_faiss_index(self) -> None:
        """FAISS 인덱스 구축/로드 (GPU 가속 지원)"""
        if not self.index_dir:
            # FAISS 없으면 simple로 fallback
            logger.warning("No index_dir provided, falling back to simple index")
            self._build_simple_index()
            return
        
        index_path = Path(self.index_dir) / "index.faiss"
        meta_path = Path(self.index_dir) / "meta.json"
        
        if not (index_path.exists() and meta_path.exists()):
            # 인덱스가 없으면 simple로 fallback
            logger.warning(f"FAISS index not found at {index_path}, falling back to simple")
            self._build_simple_index()
            return
        
        try:
            import faiss
            
            # GPU 가속 확인
            if self.use_gpu and faiss.get_num_gpus() > 0:
                logger.info("Using FAISS GPU acceleration")
                self._build_gpu_faiss_index()
            else:
                logger.info("Using FAISS CPU index")
                self._build_cpu_faiss_index()
            
            import json
            
            # FAISS 인덱스 로드
            self.index = faiss.read_index(str(index_path))
            
            # 메타 정보 로드
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.dim = meta.get("dim", self.embedder.dim)
            
            self.backend = "faiss"
            logger.info("FAISS index loaded", dim=self.dim)
        
        except ImportError:
            logger.warning("FAISS not available, falling back to simple index")
            self._build_simple_index()
        
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}", exc_info=True)
            self._build_simple_index()
    
    def _build_gpu_faiss_index(self) -> None:
        """GPU 가속 FAISS 인덱스 구축"""
        try:
            import faiss
            
            # GPU 리소스 생성
            self.gpu_res = faiss.StandardGpuResources()
            
            # CPU 인덱스를 GPU로 전환
            cpu_index = faiss.read_index(str(Path(self.index_dir) / "index.faiss"))
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
            
            logger.info("FAISS GPU index created successfully")
            
        except Exception as e:
            logger.warning(f"GPU FAISS failed, falling back to CPU: {e}")
            self._build_cpu_faiss_index()
    
    def _build_cpu_faiss_index(self) -> None:
        """CPU FAISS 인덱스 구축"""
        try:
            import faiss
            
            # CPU 인덱스 로드
            index_path = Path(self.index_dir) / "index.faiss"
            self.index = faiss.read_index(str(index_path))
            
            logger.info("FAISS CPU index loaded successfully")
            
        except Exception as e:
            logger.error(f"CPU FAISS failed: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """
        벡터 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 수
            
        Returns:
            [(청크 인덱스, 유사도), ...] 리스트
        """
        try:
            # 쿼리 임베딩
            query_vec = self.embedder.embed_query(query)
            
            if self.backend == "faiss" and hasattr(self, 'index'):
                return self._search_faiss(query_vec, top_k)
            else:
                return self._search_simple(query_vec, top_k)
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return []
    
    def _search_simple(
        self,
        query_vec: np.ndarray,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """간단한 numpy 기반 검색"""
        if not hasattr(self, 'vectors'):
            return []
        
        # 🚀 최적화 1B: 정규화된 벡터 사용 (dot product만으로 코사인 유사도!)
        query_normalized = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        similarities = np.dot(self.vectors, query_normalized)
        # 두 벡터 모두 정규화되어 있으면: dot(a, b) = cosine_similarity(a, b)
        
        # 상위 k개 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        result = [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
        ]
        
        logger.debug(f"Simple vector search completed",
                    results=len(result),
                    top_score=result[0][1] if result else 0.0)
        
        return result
    
    def _search_faiss(
        self,
        query_vec: np.ndarray,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """FAISS 기반 검색"""
        query_vec = query_vec.reshape(1, -1).astype('float32')
        
        # FAISS 검색
        D, I = self.index.search(query_vec, top_k)
        
        result = [
            (int(idx), float(score))
            for idx, score in zip(I[0], D[0])
            if idx >= 0 and idx < len(self.chunks)
        ]
        
        logger.debug(f"FAISS vector search completed",
                    results=len(result),
                    top_score=result[0][1] if result else 0.0)
        
        return result

