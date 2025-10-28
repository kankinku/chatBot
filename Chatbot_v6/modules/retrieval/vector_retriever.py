"""
Vector Retriever - Chroma DB 기반

Chroma DB를 사용한 단순하고 효율적인 벡터 검색.
NumPy 의존성 제거, 복잡한 인덱스 관리 불필요.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import uuid

from modules.core.types import Chunk
from modules.core.logger import get_logger
from modules.core.exceptions import VectorStoreNotFoundError, EmbeddingError
from modules.embedding.base_embedder import BaseEmbedder

logger = get_logger(__name__)


class VectorRetriever:
    """Chroma DB 기반 벡터 검색"""
    
    def __init__(
        self,
        chunks: List[Chunk],
        embedder: BaseEmbedder,
        index_dir: Optional[str] = None,
        collection_name: str = "chatbot_chunks",
    ):
        """
        Args:
            chunks: 청크 리스트
            embedder: 임베더
            index_dir: Chroma DB 저장 디렉토리
            collection_name: Chroma 컬렉션 이름
        """
        self.chunks = chunks
        self.embedder = embedder
        self.index_dir = index_dir or "vector_store"
        self.collection_name = collection_name
        # 청크 인덱스를 Chroma ID에 매핑 (청크 인덱스 -> Chroma ID)
        self.chunk_index_to_id: List[str] = []
        
        logger.info("VectorRetriever initializing",
                   num_chunks=len(chunks),
                   collection_name=collection_name,
                   embedding_dim=embedder.dim)
        
        # Chroma DB 초기화
        self._init_chroma()
        
        logger.info("VectorRetriever initialized")
    
    def _init_chroma(self) -> None:
        """Chroma DB 초기화 및 인덱스 구축"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Chroma 클라이언트 초기화
            self.client = chromadb.PersistentClient(
                path=self.index_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 컬렉션 가져오기 또는 생성
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function()
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
                
                # 컬렉션 크기 확인
                count = self.collection.count()
                if count != len(self.chunks):
                    logger.warning(f"Collection size mismatch: {count} vs {len(self.chunks)}")
                    logger.info("Rebuilding collection...")
                    self._build_collection()
                else:
                    logger.info(f"Collection ready with {count} documents")
                    # 기존 컬렉션에서 ID 매핑 재구성
                    self._rebuild_id_mapping()
                    
            except Exception as e:
                # 컬렉션이 없거나 다른 문제 발생
                logger.info(f"Creating new collection: {self.collection_name}", error=str(e))
                self._build_collection()
                
        except ImportError as e:
            raise EmbeddingError(
                "Chroma DB not available. Please install: pip install chromadb",
                cause=e
            ) from e
        
        except Exception as e:
            raise EmbeddingError(
                "Failed to initialize Chroma DB",
                cause=e
            ) from e
    
    def _get_embedding_function(self):
        """Chroma용 임베딩 함수 생성 (Chroma 0.4.16+ 호환)"""
        try:
            from chromadb import EmbeddingFunction
            
            class CustomEmbeddingFunction(EmbeddingFunction):
                def __init__(self, embedder):
                    super().__init__()
                    self.embedder = embedder
                
                def __call__(self, input):
                    """Chroma 0.4.16+는 'input' 파라미터 사용"""
                    if not input:
                        return []
                    
                    # 단일 문자열인 경우 리스트로 변환
                    if isinstance(input, str):
                        input = [input]
                    
                    # 임베딩 생성
                    try:
                        embeddings = self.embedder.embed_texts(input)
                        
                        # NumPy 배열을 리스트로 변환
                        import numpy as np
                        if isinstance(embeddings, np.ndarray):
                            return [emb.tolist() for emb in embeddings]
                        else:
                            return [[float(x) for x in emb] for emb in embeddings]
                    except Exception as e:
                        logger.error(f"Embedding generation failed: {e}", exc_info=True)
                        return []
            
            return CustomEmbeddingFunction(self.embedder)
            
        except ImportError:
            # EmbeddingFunction 클래스가 없으면 간단한 함수로 대체
            def embedding_function(input):
                if not input:
                    return []
                if isinstance(input, str):
                    input = [input]
                
                embeddings = self.embedder.embed_texts(input)
                import numpy as np
                if isinstance(embeddings, np.ndarray):
                    return [emb.tolist() for emb in embeddings]
                else:
                    return [[float(x) for x in emb] for emb in embeddings]
            
            return embedding_function
    
    def _build_collection(self) -> None:
        """컬렉션 구축"""
        try:
            # 기존 컬렉션 삭제 (있다면)
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass  # 컬렉션이 없으면 무시
            
            # 새 컬렉션 생성
            embedding_func = self._get_embedding_function()
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_func,
                metadata={"description": "Chatbot chunks collection"}
            )
            
            # 청크 ID 매핑 초기화
            self.chunk_index_to_id = []
            
            # 청크들을 Chroma에 추가
            texts = []
            metadatas = []
            ids = []
            
            for idx, chunk in enumerate(self.chunks):
                # 고유 ID 생성 (인덱스 기반)
                chunk_id = f"chunk_{idx}_{uuid.uuid4().hex[:8]}"
                self.chunk_index_to_id.append(chunk_id)
                
                # 텍스트 추가
                texts.append(chunk.text)
                
                # 메타데이터 생성 (Chunk 타입에 맞게)
                metadata = {
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "page": str(chunk.page) if chunk.page is not None else "None",
                    "start_offset": str(chunk.start_offset),
                    "length": str(chunk.length),
                    "chunk_index": str(idx),  # 원본 인덱스 저장
                }
                
                # 추가 메타데이터가 있으면 포함
                if chunk.extra:
                    for key, value in chunk.extra.items():
                        # Chroma는 문자열만 지원하므로 변환
                        metadata[f"extra_{key}"] = str(value)
                
                metadatas.append(metadata)
                ids.append(chunk_id)
            
            # 배치로 추가 (Chroma가 자동으로 임베딩 생성)
            # 대량 데이터의 경우 배치 처리
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                logger.debug(f"Added batch {i//batch_size + 1}: {len(batch_texts)} documents")
            
            logger.info(f"Collection built successfully with {len(self.chunks)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build Chroma collection: {e}", exc_info=True)
            raise EmbeddingError(
                "Failed to build Chroma collection",
                cause=e
            ) from e
    
    def _rebuild_id_mapping(self) -> None:
        """기존 컬렉션에서 ID 매핑 재구성"""
        try:
            # 모든 문서 가져오기
            all_data = self.collection.get(include=["metadatas"])
            
            if not all_data or not all_data["ids"]:
                # 빈 컬렉션인 경우 새로 구축
                logger.warning("Collection is empty, rebuilding...")
                self._build_collection()
                return
            
            # 메타데이터에서 인덱스 추출하여 매핑 재구성
            self.chunk_index_to_id = [""] * len(self.chunks)
            
            for i, (id_val, metadata) in enumerate(zip(all_data["ids"], all_data["metadatas"] or [])):
                chunk_idx_str = metadata.get("chunk_index") if metadata else None
                if chunk_idx_str:
                    try:
                        chunk_idx = int(chunk_idx_str)
                        if 0 <= chunk_idx < len(self.chunks):
                            self.chunk_index_to_id[chunk_idx] = id_val
                    except (ValueError, IndexError):
                        pass
            
            logger.info(f"ID mapping rebuilt: {len([x for x in self.chunk_index_to_id if x])} mappings")
            
        except Exception as e:
            logger.warning(f"Failed to rebuild ID mapping: {e}", exc_info=True)
            # 매핑 재구성 실패 시 새로 구축
            logger.info("Rebuilding collection due to mapping failure...")
            self._build_collection()
    
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
            # Chroma 검색
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self.chunks)),  # 전체 청크 수를 초과하지 않도록
                include=["metadatas", "distances"]
            )
            
            # 결과 변환
            result = []
            if results and results.get("ids") and results["ids"][0]:
                for i, (id_val, metadata, distance) in enumerate(zip(
                    results["ids"][0],
                    results["metadatas"][0] if results["metadatas"] and results["metadatas"][0] else [],
                    results["distances"][0]
                )):
                    # 청크 인덱스 찾기
                    chunk_idx = self._find_chunk_index(id_val, metadata)
                    if chunk_idx is not None:
                        # 거리를 유사도로 변환 (Chroma는 거리 반환, 코사인 거리의 경우)
                        # 코사인 거리는 0~2 사이, 유사도는 1 - distance / 2로 변환
                        similarity = max(0.0, 1.0 - (distance / 2.0))
                        result.append((chunk_idx, similarity))
            
            logger.debug(f"Chroma vector search completed",
                        results=len(result),
                        top_score=result[0][1] if result else 0.0)
            
            return result
        
        except Exception as e:
            logger.error(f"Chroma search failed: {e}", exc_info=True)
            return []
    
    def _find_chunk_index(self, chroma_id: str, metadata: Optional[dict] = None) -> Optional[int]:
        """Chroma ID로부터 청크 인덱스 찾기"""
        try:
            # 방법 1: 메타데이터에서 직접 추출
            if metadata:
                chunk_idx_str = metadata.get("chunk_index")
                if chunk_idx_str:
                    try:
                        chunk_idx = int(chunk_idx_str)
                        if 0 <= chunk_idx < len(self.chunks):
                            return chunk_idx
                    except (ValueError, IndexError):
                        pass
            
            # 방법 2: ID 매핑에서 찾기
            if self.chunk_index_to_id:
                try:
                    chunk_idx = self.chunk_index_to_id.index(chroma_id)
                    if 0 <= chunk_idx < len(self.chunks):
                        return chunk_idx
                except ValueError:
                    pass
            
            # 방법 3: ID에서 인덱스 추출 시도 (fallback)
            # chunk_0_xxxxx 형식에서 인덱스 추출
            if chroma_id.startswith("chunk_"):
                parts = chroma_id.split("_")
                if len(parts) >= 2:
                    try:
                        chunk_idx = int(parts[1])
                        if 0 <= chunk_idx < len(self.chunks):
                            return chunk_idx
                    except (ValueError, IndexError):
                        pass
            
            logger.warning(f"Failed to find chunk index for ID: {chroma_id}")
            return None
            
        except Exception as e:
            logger.warning(f"Error finding chunk index: {e}", exc_info=True)
            return None