"""
벡터 저장소 및 검색 기능 모듈

이 모듈은 PDF에서 추출한 텍스트 청크들의 임베딩을 저장하고,
질문에 대해 의미적으로 유사한 텍스트를 검색하는 기능을 제공합니다.
"""

import os
import json
import pickle
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import asdict
import numpy as np
from abc import ABC, abstractmethod

import faiss
import chromadb
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity

from .pdf_processor import TextChunk

import logging
logger = logging.getLogger(__name__)

class VectorStoreInterface(ABC):
    """벡터 저장소 인터페이스"""
    
    @abstractmethod
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """텍스트 청크들을 저장소에 추가"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        """쿼리 임베딩과 유사한 청크들을 검색"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """저장소를 파일로 저장"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """파일에서 저장소 로드"""
        pass

class FAISSVectorStore(VectorStoreInterface):
    """
    FAISS를 사용한 벡터 저장소
    
    장점:
    - 매우 빠른 검색 속도
    - 메모리 효율적
    - 대용량 데이터 처리 가능
    
    단점:
    - 메타데이터 관리가 별도로 필요
    - 실시간 업데이트가 상대적으로 복잡
    """
    
    def __init__(self, embedding_dimension: int = 768):
        """
        FAISS 벡터 저장소 초기화
        
        Args:
            embedding_dimension: 임베딩 벡터의 차원
        """
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatIP(embedding_dimension)  # Inner Product (코사인 유사도)
        self.chunks: List[TextChunk] = []
        self.chunk_metadata: List[Dict] = []
        
        # 정규화를 위한 설정 (코사인 유사도 계산용)
        self.normalize_embeddings = True
        
        logger.info(f"FAISS 벡터 저장소 초기화 완료 (차원: {embedding_dimension})")
    
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        텍스트 청크들을 FAISS 인덱스에 추가
        
        Args:
            chunks: 임베딩이 포함된 TextChunk 리스트
        """
        if not chunks:
            return
        
        # 임베딩 벡터 추출 및 정규화
        embeddings = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"청크 {chunk.chunk_id}에 임베딩이 없습니다.")
            
            embedding = chunk.embedding.copy()
            if self.normalize_embeddings:
                # L2 정규화 (코사인 유사도 계산용)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            embeddings.append(embedding)
        
        # FAISS 인덱스에 추가
        embeddings_matrix = np.array(embeddings).astype('float32')
        self.index.add(embeddings_matrix)
        
        # 메타데이터 저장
        self.chunks.extend(chunks)
        for chunk in chunks:
            metadata = asdict(chunk)
            metadata.pop('embedding', None)  # 임베딩은 별도 저장
            self.chunk_metadata.append(metadata)
        
        logger.info(f"{len(chunks)}개 청크를 FAISS 인덱스에 추가 (총 {len(self.chunks)}개)")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        """
        쿼리 임베딩과 유사한 청크들을 검색
        
        Args:
            query_embedding: 쿼리의 임베딩 벡터
            top_k: 반환할 상위 결과 개수
            
        Returns:
            (TextChunk, 유사도_점수) 튜플 리스트
        """
        if self.index.ntotal == 0:
            return []
        
        # 쿼리 임베딩 정규화
        query_embedding = query_embedding.copy()
        if self.normalize_embeddings:
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        # FAISS 검색
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # 결과 구성
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS에서 유효하지 않은 인덱스
                continue
            
            chunk = self.chunks[idx]
            similarity_score = float(score)  # Inner product 점수
            results.append((chunk, similarity_score))
        
        return results
    
    def save(self, path: str) -> None:
        """
        FAISS 인덱스와 메타데이터를 파일로 저장
        
        Args:
            path: 저장할 디렉토리 경로
        """
        os.makedirs(path, exist_ok=True)
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))
        
        # 청크 메타데이터 저장
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        
        # 추가 메타데이터 저장
        metadata = {
            "embedding_dimension": self.embedding_dimension,
            "total_chunks": len(self.chunks),
            "normalize_embeddings": self.normalize_embeddings
        }
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"FAISS 벡터 저장소를 {path}에 저장 완료")
    
    def load(self, path: str) -> None:
        """
        파일에서 FAISS 인덱스와 메타데이터를 로드
        
        Args:
            path: 로드할 디렉토리 경로
        """
        # 메타데이터 로드
        with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        self.embedding_dimension = metadata["embedding_dimension"]
        self.normalize_embeddings = metadata.get("normalize_embeddings", True)
        
        # FAISS 인덱스 로드
        self.index = faiss.read_index(os.path.join(path, "faiss_index.bin"))
        
        # 청크 데이터 로드
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        
        logger.info(f"FAISS 벡터 저장소를 {path}에서 로드 완료 ({len(self.chunks)}개 청크)")

class ChromaDBVectorStore(VectorStoreInterface):
    """
    ChromaDB를 사용한 벡터 저장소
    
    장점:
    - 메타데이터 관리가 편리
    - 실시간 업데이트 지원
    - 필터링 기능 지원
    
    단점:
    - FAISS보다 검색 속도가 상대적으로 느림
    - 메모리 사용량이 상대적으로 많음
    """
    
    def __init__(self, collection_name: str = "pdf_chunks", persist_directory: str = "./chroma_db"):
        """
        ChromaDB 벡터 저장소 초기화
        
        Args:
            collection_name: 컬렉션 이름
            persist_directory: 데이터 지속성을 위한 디렉토리
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 컬렉션 생성 또는 가져오기
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"기존 ChromaDB 컬렉션 로드: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "PDF 텍스트 청크 저장소"}
            )
            logger.info(f"새 ChromaDB 컬렉션 생성: {collection_name}")
    
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        텍스트 청크들을 ChromaDB에 추가
        
        Args:
            chunks: 임베딩이 포함된 TextChunk 리스트
        """
        if not chunks:
            return
        
        # ChromaDB 형식으로 데이터 변환
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"청크 {chunk.chunk_id}에 임베딩이 없습니다.")
            
            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding.tolist())
            documents.append(chunk.content)
            
            metadata = {
                "page_number": chunk.page_number,
                "pdf_id": chunk.metadata.get("pdf_id") if chunk.metadata else "",
                "chunk_index": chunk.metadata.get("chunk_index") if chunk.metadata else 0
            }
            metadatas.append(metadata)
        
        # ChromaDB에 추가
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"{len(chunks)}개 청크를 ChromaDB에 추가")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               filter_metadata: Optional[Dict] = None) -> List[Tuple[TextChunk, float]]:
        """
        쿼리 임베딩과 유사한 청크들을 검색
        
        Args:
            query_embedding: 쿼리의 임베딩 벡터
            top_k: 반환할 상위 결과 개수
            filter_metadata: 메타데이터 필터 조건
            
        Returns:
            (TextChunk, 유사도_점수) 튜플 리스트
        """
        # ChromaDB 검색
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata
        )
        
        # TextChunk 객체로 변환
        chunks_with_scores = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                chunk_id = results['ids'][0][i]
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # 거리를 유사도로 변환 (ChromaDB는 거리 기반)
                similarity_score = 1.0 - distance
                
                chunk = TextChunk(
                    content=content,
                    page_number=metadata.get('page_number', 1),
                    chunk_id=chunk_id,
                    metadata=metadata
                )
                
                chunks_with_scores.append((chunk, similarity_score))
        
        return chunks_with_scores
    
    def save(self, path: str) -> None:
        """ChromaDB는 자동으로 지속성 관리"""
        logger.info("ChromaDB는 자동으로 데이터를 지속적으로 저장합니다.")
    
    def load(self, path: str) -> None:
        """ChromaDB는 초기화 시 자동으로 로드됨"""
        logger.info("ChromaDB는 초기화 시 자동으로 데이터를 로드합니다.")

class HybridVectorStore:
    """
    FAISS와 ChromaDB를 결합한 하이브리드 벡터 저장소
    
    - FAISS: 빠른 유사도 검색
    - ChromaDB: 메타데이터 관리 및 필터링
    """
    
    def __init__(self, embedding_dimension: int = 768, 
                 collection_name: str = "pdf_chunks",
                 persist_directory: str = "./vector_store"):
        """
        하이브리드 벡터 저장소 초기화
        
        Args:
            embedding_dimension: 임베딩 차원
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: 데이터 저장 디렉토리
        """
        self.faiss_store = FAISSVectorStore(embedding_dimension)
        self.chroma_store = ChromaDBVectorStore(
            collection_name, 
            os.path.join(persist_directory, "chroma")
        )
        self.persist_directory = persist_directory
        
        logger.info("하이브리드 벡터 저장소 초기화 완료")
    
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """두 저장소에 모두 청크 추가"""
        self.faiss_store.add_chunks(chunks)
        self.chroma_store.add_chunks(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               use_metadata_filter: bool = False,
               filter_metadata: Optional[Dict] = None) -> List[Tuple[TextChunk, float]]:
        """
        하이브리드 검색
        
        Args:
            query_embedding: 쿼리 임베딩
            top_k: 결과 개수
            use_metadata_filter: 메타데이터 필터 사용 여부
            filter_metadata: 필터 조건
            
        Returns:
            검색 결과
        """
        if use_metadata_filter and filter_metadata:
            # 메타데이터 필터가 필요한 경우 ChromaDB 사용
            return self.chroma_store.search(query_embedding, top_k, filter_metadata)
        else:
            # 빠른 검색이 필요한 경우 FAISS 사용
            return self.faiss_store.search(query_embedding, top_k)
    
    def save(self, path: Optional[str] = None) -> None:
        """두 저장소 모두 저장"""
        save_path = path or self.persist_directory
        
        faiss_path = os.path.join(save_path, "faiss")
        self.faiss_store.save(faiss_path)
        
        # ChromaDB는 자동 저장
        self.chroma_store.save("")
        
        logger.info(f"하이브리드 벡터 저장소 저장 완료: {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """저장된 데이터 로드"""
        load_path = path or self.persist_directory
        
        faiss_path = os.path.join(load_path, "faiss")
        if os.path.exists(faiss_path):
            self.faiss_store.load(faiss_path)
        
        # ChromaDB는 자동 로드
        logger.info(f"하이브리드 벡터 저장소 로드 완료: {load_path}")

# 유틸리티 함수들
def calculate_retrieval_metrics(relevant_chunks: List[str], 
                              retrieved_chunks: List[Tuple[TextChunk, float]],
                              k: int = 5) -> Dict[str, float]:
    """
    검색 성능 메트릭 계산
    
    Args:
        relevant_chunks: 실제 관련 있는 청크 ID 리스트
        retrieved_chunks: 검색된 청크들
        k: 상위 k개 결과에 대한 평가
        
    Returns:
        성능 메트릭 딕셔너리
    """
    if not relevant_chunks or not retrieved_chunks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "map": 0.0}
    
    # 상위 k개 결과만 고려
    top_k_chunks = retrieved_chunks[:k]
    retrieved_ids = [chunk.chunk_id for chunk, _ in top_k_chunks]
    
    # Precision@K
    relevant_retrieved = set(relevant_chunks) & set(retrieved_ids)
    precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
    
    # Recall@K  
    recall = len(relevant_retrieved) / len(relevant_chunks)
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Mean Average Precision (MAP)
    ap = 0.0
    relevant_count = 0
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_chunks:
            relevant_count += 1
            ap += relevant_count / (i + 1)
    
    map_score = ap / len(relevant_chunks) if relevant_chunks else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map": map_score
    }

if __name__ == "__main__":
    # 테스트 코드
    print("벡터 저장소 모듈이 정상적으로 로드되었습니다.")
    
    # 성능 비교 테스트 (실제 데이터가 있을 때)
    # embedding_dim = 768
    # faiss_store = FAISSVectorStore(embedding_dim)
    # chroma_store = ChromaDBVectorStore()
    # hybrid_store = HybridVectorStore(embedding_dim)
