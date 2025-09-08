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

from core.document.pdf_processor import TextChunk

import logging
logger = logging.getLogger(__name__)

# VectorStore 클래스는 HybridVectorStore로 대체됨

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
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               use_hybrid_search: bool = True) -> List[Tuple[TextChunk, float]]:
        """
        쿼리 임베딩과 유사한 청크들을 검색 (개선된 버전)
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 최대 결과 수
            use_hybrid_search: 하이브리드 검색 사용 여부
            
        Returns:
            (TextChunk, 유사도 점수) 튜플 리스트
        """
        import time
        search_start = time.time()
        
        if len(self.chunks) == 0:
            return []
        
        # 1. 벡터 유사도 검색
        vector_start = time.time()
        if self.normalize_embeddings:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # FAISS 검색
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            min(top_k * 2, len(self.chunks))  # 더 많은 후보 검색
        )
        vector_time = time.time() - vector_start
        
        # 2. 하이브리드 검색 (키워드 매칭 + 벡터 유사도)
        hybrid_start = time.time()
        if use_hybrid_search:
            results = self._hybrid_search(query_embedding, scores[0], indices[0], top_k)
        else:
            results = [(self.chunks[i], float(s)) for s, i in zip(scores[0], indices[0])]
        hybrid_time = time.time() - hybrid_start
        
        # 3. 결과 필터링 및 정렬
        filter_start = time.time()
        filtered_results = self._filter_and_rank_results(results, top_k)
        filter_time = time.time() - filter_start
        
        total_time = time.time() - search_start
        print(f"    FAISS 검색 세부: 벡터({vector_time:.3f}s) | 하이브리드({hybrid_time:.3f}s) | 필터({filter_time:.3f}s) | 총({total_time:.3f}s)")
        
        return filtered_results[:top_k]
    
    def _hybrid_search(self, query_embedding: np.ndarray, 
                      vector_scores: np.ndarray, 
                      vector_indices: np.ndarray,
                      top_k: int) -> List[Tuple[TextChunk, float]]:
        """
        하이브리드 검색 (벡터 유사도 + 키워드 매칭)
        
        Args:
            query_embedding: 쿼리 임베딩
            vector_scores: 벡터 유사도 점수들
            vector_indices: 벡터 인덱스들
            top_k: 반환할 결과 수
            
        Returns:
            하이브리드 점수로 정렬된 결과들
        """
        hybrid_results = []
        
        for score, idx in zip(vector_scores, vector_indices):
            chunk = self.chunks[idx]
            
            # 벡터 유사도 점수 (0.6 가중치)
            vector_score = float(score) * 0.6
            
            # 키워드 매칭 점수 (0.4 가중치)
            keyword_score = self._calculate_keyword_score(chunk) * 0.4
            
            # 하이브리드 점수
            hybrid_score = vector_score + keyword_score
            
            hybrid_results.append((chunk, hybrid_score))
        
        # 점수로 정렬
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_results
    
    def _calculate_keyword_score(self, chunk: TextChunk) -> float:
        """
        청크의 키워드 매칭 점수 계산
        
        Args:
            chunk: 평가할 텍스트 청크
            
        Returns:
            키워드 매칭 점수 (0.0 ~ 1.0)
        """
        # 중요 키워드 패턴 (도메인별로 확장 가능)
        important_keywords = [
            '시스템', '프로그램', '소프트웨어', '애플리케이션',
            '데이터', '정보', '자료', '파일',
            '사용자', '관리자', '고객', '이용자',
            '보안', '인증', '권한', '접근',
            '성능', '속도', '효율', '품질',
            '오류', '문제', '장애', '해결',
            '설정', '구성', '환경', '옵션',
            '백업', '복구', '저장', '보관',
            '네트워크', '통신', '연결', '전송',
            '데이터베이스', 'DB', '테이블', '쿼리',
            # 교통 시스템 관련 키워드 추가
            '분석', '뷰', '활성화', '교차로', '교통',
            '네비게이션', '사이드바', '목록', '검색',
            '선택', '마커', '지도', '패널',
            '버튼', '클릭', '위치', '방법',
            '좌측', '우측', '상단', '하단'
        ]
        
        content_lower = chunk.content.lower()
        score = 0.0
        
        # 중요 키워드 매칭
        for keyword in important_keywords:
            if keyword in content_lower:
                score += 0.1
        
        # 키워드 밀도 계산
        total_words = len(content_lower.split())
        if total_words > 0:
            keyword_density = score / total_words
            score += keyword_density * 0.5
        
        return min(score, 1.0)
    
    def _filter_and_rank_results(self, results: List[Tuple[TextChunk, float]], 
                                top_k: int) -> List[Tuple[TextChunk, float]]:
        """
        검색 결과 필터링 및 재정렬
        
        Args:
            results: 원본 검색 결과
            top_k: 반환할 결과 수
            
        Returns:
            필터링 및 재정렬된 결과
        """
        filtered_results = []
        
        for chunk, score in results:
            # 최소 점수 임계값
            if score < 0.1:
                continue
            
            # 중복 내용 필터링
            is_duplicate = False
            for existing_chunk, _ in filtered_results:
                if self._is_similar_content(chunk.content, existing_chunk.content):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_results.append((chunk, score))
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def _is_similar_content(self, content1: str, content2: str, 
                           similarity_threshold: float = 0.8) -> bool:
        """
        두 텍스트 내용의 유사도 판단
        
        Args:
            content1: 첫 번째 텍스트
            content2: 두 번째 텍스트
            similarity_threshold: 유사도 임계값
            
        Returns:
            유사한 내용인지 여부
        """
        # 간단한 Jaccard 유사도 계산
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0.0
        
        return similarity >= similarity_threshold
    
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
        try:
            # 메타데이터 로드
            with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            self.embedding_dimension = metadata["embedding_dimension"]
            self.normalize_embeddings = metadata.get("normalize_embeddings", True)
            
            # FAISS 인덱스 로드
            self.index = faiss.read_index(os.path.join(path, "faiss_index.bin"))
            
            # 청크 데이터 로드 (import 경로 문제 해결)
            import sys
            from pathlib import Path
            import pickle
            
            # 현재 디렉토리를 Python 경로에 추가
            current_dir = Path(__file__).parent.parent.parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            
            # pickle 로딩을 위한 커스텀 unpickler
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # 모듈 경로 수정
                    if module == 'core.pdf_processor':
                        module = 'core.document.pdf_processor'
                    elif module == 'pdf_processor':
                        module = 'core.document.pdf_processor'
                    
                    return super().find_class(module, name)
            
            with open(os.path.join(path, "chunks.pkl"), "rb") as f:
                self.chunks = CustomUnpickler(f).load()
            
            logger.info(f"FAISS 벡터 저장소를 {path}에서 로드 완료 ({len(self.chunks)}개 청크)")
            
        except Exception as e:
            logger.error(f"FAISS 벡터 저장소 로드 실패: {e}")
            # 빈 상태로 초기화
            self.chunks = []
            self.index = faiss.IndexFlatIP(self.embedding_dimension)

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
    
    def _sanitize_metadata_value(self, value: Any) -> Any:
        """ChromaDB 메타데이터 값에서 None을 허용 타입으로 변환한다."""
        if value is None:
            return ""
        if isinstance(value, (bool, int, float, str)):
            return value
        # 그 외 타입은 문자열로 변환
        try:
            return str(value)
        except Exception:
            return ""
    
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
        
        # 중복 ID 필터링
        existing_ids = set()
        try:
            # 기존 컬렉션의 모든 ID 가져오기
            existing_data = self.collection.get()
            if existing_data and existing_data['ids']:
                existing_ids = set(existing_data['ids'])
        except Exception as e:
            logger.warning(f"기존 ID 확인 중 오류 발생: {e}")
        
        new_chunks = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"청크 {chunk.chunk_id}에 임베딩이 없습니다.")
            
            # 중복 ID인지 확인
            if chunk.chunk_id in existing_ids:
                logger.info(f"중복 ID 건너뛰기: {chunk.chunk_id}")
                continue
            
            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding.tolist())
            documents.append(chunk.content)
            
            # 메타데이터 None 방지 및 타입 정규화
            page_number = chunk.page_number if chunk.page_number is not None else 0
            if not isinstance(page_number, int):
                try:
                    page_number = int(page_number)
                except Exception:
                    page_number = 0
            pdf_id_val = (chunk.metadata.get("pdf_id") if chunk.metadata else "")
            chunk_index_val = (chunk.metadata.get("chunk_index") if chunk.metadata else 0)
            if chunk_index_val is None:
                chunk_index_val = 0
            elif not isinstance(chunk_index_val, int):
                try:
                    chunk_index_val = int(chunk_index_val)
                except Exception:
                    chunk_index_val = 0

            metadata = {
                "page_number": self._sanitize_metadata_value(page_number),
                "pdf_id": self._sanitize_metadata_value(pdf_id_val),
                "chunk_index": self._sanitize_metadata_value(chunk_index_val)
            }
            metadatas.append(metadata)
            new_chunks.append(chunk)
        
        if not new_chunks:
            logger.info("추가할 새로운 청크가 없습니다.")
            return
        
        # ChromaDB에 추가
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"{len(new_chunks)}개 청크를 ChromaDB에 추가")
        except Exception as e:
            logger.error(f"ChromaDB에 청크 추가 실패: {e}")
            raise
    
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
    하이브리드 벡터 저장소
    
    FAISS와 ChromaDB를 병행 사용하여 빠른 검색과 메타데이터 관리를 모두 제공합니다.
    다중 임베딩 모델과 표현별 인덱싱을 지원합니다.
    """
    
    def __init__(self, 
                 faiss_store: Optional[FAISSVectorStore] = None,
                 chroma_store: Optional[ChromaDBVectorStore] = None,
                 embedding_models: Optional[List[str]] = None,
                 primary_model: str = "jhgan/ko-sroberta-multitask"):
        """
        HybridVectorStore 초기화
        
        Args:
            faiss_store: FAISS 벡터 저장소
            chroma_store: ChromaDB 벡터 저장소
            embedding_models: 사용할 임베딩 모델 리스트
            primary_model: 주 임베딩 모델
        """
        self.faiss_store = faiss_store or FAISSVectorStore()
        self.chroma_store = chroma_store or ChromaDBVectorStore()
        
        # 다중 임베딩 모델 설정
        self.embedding_models = embedding_models or [
            "jhgan/ko-sroberta-multitask",  # 한국어 특화
            "all-MiniLM-L6-v2",            # 영어/다국어
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 다국어
        ]
        self.primary_model = primary_model
        
        # 표현별 인덱스 관리
        self.expression_indices: Dict[str, Dict] = {}
        self.model_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        
        # 기존 데이터 자동 로드
        try:
            self.load()
            logger.info("기존 벡터 저장소 데이터 로드 완료")
        except Exception as e:
            logger.warning(f"기존 데이터 로드 실패 (새로 시작): {e}")
        
        logger.info(f"Hybrid Vector Store 초기화 완료 (모델: {len(self.embedding_models)}개)")
    
    def add_chunks_with_expressions(self, chunks: List[TextChunk], 
                                  expression_enhancer=None) -> None:
        """
        표현을 고려한 청크 추가
        
        Args:
            chunks: 텍스트 청크들
            expression_enhancer: 표현 향상기 (KeywordEnhancer)
        """
        # 기본 청크 추가
        self.faiss_store.add_chunks(chunks)
        self.chroma_store.add_chunks(chunks)
        
        # 표현별 인덱스 생성
        if expression_enhancer:
            self._create_expression_indices(chunks, expression_enhancer)
    
    def _create_expression_indices(self, chunks: List[TextChunk], 
                                 expression_enhancer) -> None:
        """표현별 인덱스 생성"""
        for chunk in chunks:
            # 교통, 데이터베이스, 일반 컨텍스트별 표현 추출
            for context in ["traffic", "database", "general"]:
                expressions = expression_enhancer.get_multi_expressions(
                    chunk.content, context
                )
                
                if expressions:
                    if context not in self.expression_indices:
                        self.expression_indices[context] = {}
                    
                    for expr in expressions:
                        if expr not in self.expression_indices[context]:
                            self.expression_indices[context][expr] = []
                        self.expression_indices[context][expr].append(chunk.chunk_id)
    
    def search_with_expressions(self, query: str, 
                              expression_enhancer=None,
                              context: str = "general",
                              top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        """
        표현을 고려한 검색
        
        Args:
            query: 검색 쿼리
            expression_enhancer: 표현 향상기
            context: 컨텍스트
            top_k: 반환할 결과 수
            
        Returns:
            (청크, 유사도) 튜플 리스트
        """
        # 기본 검색
        basic_results = self.search(query, top_k=top_k)
        
        if not expression_enhancer:
            return basic_results
        
        # 표현 기반 검색
        expression_results = self._search_by_expressions(
            query, expression_enhancer, context, top_k
        )
        
        # 결과 통합 및 랭킹
        combined_results = self._combine_search_results(
            basic_results, expression_results, top_k
        )
        
        return combined_results
    
    def _search_by_expressions(self, query: str, 
                             expression_enhancer,
                             context: str,
                             top_k: int) -> List[Tuple[TextChunk, float]]:
        """표현 기반 검색"""
        expressions = expression_enhancer.get_multi_expressions(query, context)
        if not expressions:
            return []
        
        # 표현별 관련 청크 수집
        expression_chunks = set()
        for expr in expressions:
            if context in self.expression_indices and expr in self.expression_indices[context]:
                chunk_ids = self.expression_indices[context][expr]
                expression_chunks.update(chunk_ids)
        
        # 관련 청크들의 임베딩 검색
        results = []
        for chunk_id in expression_chunks:
            # ChromaDB에서 청크 정보 가져오기
            chunk_info = self.chroma_store.get_chunk_by_id(chunk_id)
            if chunk_info:
                # 임시 유사도 점수 (표현 매칭 기반)
                similarity = 0.8  # 기본 점수
                results.append((chunk_info, similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _combine_search_results(self, 
                              basic_results: List[Tuple[TextChunk, float]],
                              expression_results: List[Tuple[TextChunk, float]],
                              top_k: int) -> List[Tuple[TextChunk, float]]:
        """검색 결과 통합"""
        # 청크 ID별로 최고 점수 유지
        combined_scores = {}
        
        # 기본 결과 추가
        for chunk, score in basic_results:
            combined_scores[chunk.chunk_id] = score
        
        # 표현 결과 추가 (더 높은 점수로 업데이트)
        for chunk, score in expression_results:
            if chunk.chunk_id in combined_scores:
                combined_scores[chunk.chunk_id] = max(
                    combined_scores[chunk.chunk_id], score
                )
            else:
                combined_scores[chunk.chunk_id] = score
        
        # 점수순 정렬
        sorted_chunks = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 결과 구성
        final_results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            chunk = self.chroma_store.get_chunk_by_id(chunk_id)
            if chunk:
                final_results.append((chunk, score))
        
        return final_results
    
    def get_expression_statistics(self) -> Dict[str, Any]:
        """표현 인덱스 통계 반환"""
        stats = {
            "total_contexts": len(self.expression_indices),
            "context_details": {}
        }
        
        for context, expressions in self.expression_indices.items():
            stats["context_details"][context] = {
                "total_expressions": len(expressions),
                "total_chunks": sum(len(chunk_ids) for chunk_ids in expressions.values()),
                "top_expressions": sorted(
                    expressions.items(), 
                    key=lambda x: len(x[1]), 
                    reverse=True
                )[:5]
            }
        
        return stats
    
    def clear(self) -> None:
        """저장소 초기화"""
        try:
            # FAISS 저장소 초기화
            if hasattr(self.faiss_store, 'chunks'):
                self.faiss_store.chunks.clear()
            if hasattr(self.faiss_store, 'index'):
                self.faiss_store.index = faiss.IndexFlatIP(self.faiss_store.embedding_dimension)
            
            # ChromaDB 저장소 초기화
            if hasattr(self.chroma_store, 'collection'):
                try:
                    self.chroma_store.collection.delete(where={})
                    logger.info("ChromaDB 컬렉션 초기화 완료")
                except Exception as e:
                    logger.warning(f"ChromaDB 초기화 실패: {e}")
            
            # 표현 인덱스 초기화
            self.expression_indices.clear()
            self.model_embeddings.clear()
            
            logger.info("벡터 저장소 완전 초기화 완료")
            
        except Exception as e:
            logger.error(f"벡터 저장소 초기화 실패: {e}")
    
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """두 저장소에 모두 청크 추가 (중복 체크 포함)"""
        if not chunks:
            return
        
        # 중복 체크
        existing_chunk_ids = set()
        try:
            if hasattr(self.faiss_store, 'chunks'):
                existing_chunk_ids = {chunk.chunk_id for chunk in self.faiss_store.chunks}
        except Exception as e:
            logger.warning(f"기존 청크 ID 확인 실패: {e}")
        
        # 중복되지 않은 청크만 필터링
        new_chunks = []
        duplicate_count = 0
        
        for chunk in chunks:
            if chunk.chunk_id in existing_chunk_ids:
                duplicate_count += 1
                logger.debug(f"중복 청크 건너뛰기: {chunk.chunk_id}")
            else:
                new_chunks.append(chunk)
        
        if duplicate_count > 0:
            logger.info(f"중복 청크 {duplicate_count}개 건너뛰기, 새 청크 {len(new_chunks)}개 추가")
        
        if new_chunks:
            self.faiss_store.add_chunks(new_chunks)
            self.chroma_store.add_chunks(new_chunks)
            logger.info(f"{len(new_chunks)}개 새 청크 추가 완료")
        else:
            logger.info("추가할 새 청크가 없습니다.")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               use_metadata_filter: bool = False,
               filter_metadata: Optional[Dict] = None,
               similarity_threshold: float = 0.1) -> List[Tuple[TextChunk, float]]:
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
        import time
        search_start = time.time()
        
        if use_metadata_filter and filter_metadata:
            # 메타데이터 필터가 필요한 경우 ChromaDB 사용
            result = self.chroma_store.search(query_embedding, top_k, filter_metadata)
        else:
            # 빠른 검색이 필요한 경우 FAISS 사용
            result = self.faiss_store.search(query_embedding, top_k)
        
        # 유사도 임계값 필터링 추가
        filtered_result = [(chunk, score) for chunk, score in result if score >= similarity_threshold]
        
        search_time = time.time() - search_start
        print(f"  벡터 검색: {search_time:.3f}초 (FAISS 사용, 임계값: {similarity_threshold})")
        
        return filtered_result
    
    def save(self, path: Optional[str] = None) -> None:
        """두 저장소 모두 저장"""
        save_path = path or "./vector_store"
        
        faiss_path = os.path.join(save_path, "faiss")
        self.faiss_store.save(faiss_path)
        
        # ChromaDB는 자동 저장
        self.chroma_store.save("")
        
        logger.info(f"하이브리드 벡터 저장소 저장 완료: {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """저장된 데이터 로드"""
        load_path = path or "./vector_store"
        
        faiss_path = os.path.join(load_path, "faiss")
        if os.path.exists(faiss_path):
            self.faiss_store.load(faiss_path)
        
        # ChromaDB는 자동 로드
        logger.info(f"하이브리드 벡터 저장소 로드 완료: {load_path}")
    
    def get_total_chunks(self) -> int:
        """저장된 총 청크 수 반환"""
        try:
            return len(self.faiss_store.chunks) if hasattr(self.faiss_store, 'chunks') else 0
        except:
            return 0
    
    def get_all_pdfs(self) -> List[Dict]:
        """모든 PDF 정보 반환"""
        try:
            # FAISS에서 PDF 정보 추출
            pdf_info = {}
            for chunk in self.faiss_store.chunks:
                pdf_id = chunk.pdf_id or chunk.metadata.get("pdf_id") if chunk.metadata else "unknown"
                if pdf_id not in pdf_info:
                    pdf_info[pdf_id] = {
                        'id': pdf_id,
                        'filename': chunk.filename or chunk.metadata.get("filename", "unknown") if chunk.metadata else "unknown",
                        'total_chunks': 0,
                        'upload_time': chunk.upload_time or chunk.metadata.get("upload_time", "") if chunk.metadata else ""
                    }
                pdf_info[pdf_id]['total_chunks'] += 1
            
            return list(pdf_info.values())
        except Exception as e:
            logger.warning(f"PDF 정보 추출 실패: {e}")
            return []

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

# VectorStore 클래스 정의 (HybridVectorStore의 별칭)
class VectorStore:
    """벡터 저장소 (HybridVectorStore의 별칭)"""
    
    def __init__(self, embedding_dimension: int = 768):
        """
        벡터 저장소 초기화
        
        Args:
            embedding_dimension: 임베딩 벡터의 차원
        """
        self._store = HybridVectorStore(embedding_dimension)
    
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """텍스트 청크들을 저장소에 추가"""
        self._store.add_chunks(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               similarity_threshold: float = 0.5, use_metadata_filter: bool = False,
               filter_metadata: Optional[Dict] = None) -> List[Tuple[TextChunk, float]]:
        """쿼리 임베딩과 유사한 청크들을 검색"""
        return self._store.search(query_embedding, top_k, similarity_threshold, 
                                use_metadata_filter, filter_metadata)
    
    def save(self, path: Optional[str] = None) -> None:
        """저장소를 파일로 저장"""
        self._store.save(path)
    
    def load(self, path: Optional[str] = None) -> None:
        """파일에서 저장소 로드"""
        self._store.load(path)
    
    def clear(self) -> None:
        """저장소 초기화"""
        self._store.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """저장소 통계 정보 반환"""
        return self._store.get_stats()


if __name__ == "__main__":
    # 테스트 코드
    print("벡터 저장소 모듈이 정상적으로 로드되었습니다.")
    
    # 성능 비교 테스트 (실제 데이터가 있을 때)
    # embedding_dim = 768
    # faiss_store = FAISSVectorStore(embedding_dim)
    # chroma_store = ChromaDBVectorStore()
    # hybrid_store = HybridVectorStore(embedding_dim)
