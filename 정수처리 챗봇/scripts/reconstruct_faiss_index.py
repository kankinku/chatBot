#!/usr/bin/env python3
"""
FAISS 인덱스 재구성 스크립트

ChromaDB에 있는 데이터를 사용하여 FAISS 인덱스를 재구성합니다.
"""

import os
import sys
import numpy as np
import chromadb
import faiss
from sentence_transformers import SentenceTransformer
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document.vector_store import HybridVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reconstruct_faiss_index():
    """ChromaDB 데이터로 FAISS 인덱스 재구성"""
    
    print("=== FAISS 인덱스 재구성 시작 ===")
    
    try:
        # 1. ChromaDB에서 데이터 가져오기
        print("1. ChromaDB에서 데이터 가져오는 중...")
        client = chromadb.PersistentClient(path='./chroma_db')
        collection = client.get_collection('pdf_chunks')
        
        count = collection.count()
        print(f"   ChromaDB 컬렉션 크기: {count}")
        
        if count == 0:
            print("   ChromaDB에 데이터가 없습니다. 재구성을 중단합니다.")
            return False
        
        # 모든 데이터 가져오기
        all_data = collection.get(include=['embeddings', 'metadatas', 'documents'])
        
        print(f"   가져온 데이터: {len(all_data['ids'])}개")
        
        # 2. 임베딩 데이터 준비
        print("2. 임베딩 데이터 준비 중...")
        
        embeddings = []
        valid_ids = []
        valid_metadatas = []
        valid_documents = []
        
        for i, (doc_id, embedding, metadata, document) in enumerate(zip(
            all_data['ids'], 
            all_data['embeddings'], 
            all_data['metadatas'], 
            all_data['documents']
        )):
            if embedding is not None and len(embedding) > 0:
                try:
                    # 임베딩을 numpy 배열로 변환
                    embedding_array = np.array(embedding, dtype=np.float32)
                    embeddings.append(embedding_array)
                    valid_ids.append(doc_id)
                    valid_metadatas.append(metadata)
                    valid_documents.append(document)
                except Exception as e:
                    print(f"   임베딩 {i} 변환 실패: {e}")
                    continue
        
        if not embeddings:
            print("   유효한 임베딩이 없습니다. 재구성을 중단합니다.")
            return False
        
        embeddings_array = np.vstack(embeddings)
        print(f"   유효한 임베딩: {len(embeddings)}개")
        print(f"   임베딩 차원: {embeddings_array.shape}")
        
        # 3. FAISS 인덱스 생성
        print("3. FAISS 인덱스 생성 중...")
        
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product (코사인 유사도)
        
        # 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(embeddings_array)
        
        # 인덱스에 벡터 추가
        index.add(embeddings_array)
        
        print(f"   FAISS 인덱스 크기: {index.ntotal}")
        print(f"   FAISS 인덱스 차원: {index.d}")
        
        # 4. 벡터 스토어 디렉토리 생성
        print("4. 벡터 스토어 디렉토리 생성 중...")
        
        vector_store_path = './vector_store'
        os.makedirs(vector_store_path, exist_ok=True)
        
        # 5. FAISS 인덱스 저장
        print("5. FAISS 인덱스 저장 중...")
        
        faiss_index_path = os.path.join(vector_store_path, 'faiss_index.bin')
        faiss.write_index(index, faiss_index_path)
        
        print(f"   FAISS 인덱스 저장 완료: {faiss_index_path}")
        
        # 6. 메타데이터 저장
        print("6. 메타데이터 저장 중...")
        
        metadata_path = os.path.join(vector_store_path, 'metadata.npz')
        np.savez(metadata_path, 
                ids=valid_ids,
                metadatas=valid_metadatas,
                documents=valid_documents)
        
        print(f"   메타데이터 저장 완료: {metadata_path}")
        
        # 7. 테스트 검색
        print("7. 테스트 검색 수행 중...")
        
        test_query = "테스트 쿼리"
        embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        query_embedding = embedding_model.encode([test_query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding, k=5)
        
        print(f"   테스트 검색 결과: {len(indices[0])}개")
        print(f"   검색 점수: {scores[0]}")
        
        print("=== FAISS 인덱스 재구성 완료 ===")
        return True
        
    except Exception as e:
        print(f"FAISS 인덱스 재구성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_vector_store():
    """재구성된 벡터 스토어 테스트"""
    
    print("\n=== 하이브리드 벡터 스토어 테스트 ===")
    
    try:
        # 벡터 스토어 초기화
        vector_store = HybridVectorStore()
        
        # FAISS 상태 확인
        if hasattr(vector_store, 'faiss_store') and vector_store.faiss_store:
            print(f"FAISS 인덱스 크기: {vector_store.faiss_store.index.ntotal}")
            print(f"FAISS 인덱스 차원: {vector_store.faiss_store.index.d}")
        else:
            print("FAISS 스토어가 초기화되지 않았습니다.")
            return False
        
        # ChromaDB 상태 확인
        if hasattr(vector_store, 'chroma_store') and vector_store.chroma_store:
            try:
                collection = vector_store.chroma_store.client.get_collection('pdf_chunks')
                count = collection.count()
                print(f"ChromaDB 컬렉션 크기: {count}")
            except Exception as e:
                print(f"ChromaDB 확인 실패: {e}")
        else:
            print("ChromaDB 스토어가 초기화되지 않았습니다.")
            return False
        
        # 테스트 검색
        print("테스트 검색 수행 중...")
        test_query = "시스템 사용자 설명서"
        embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        query_embedding = embedding_model.encode([test_query])
        
        results = vector_store.search(query_embedding, top_k=5, similarity_threshold=0.1)
        
        print(f"검색 결과: {len(results)}개")
        for i, (chunk, score) in enumerate(results):
            print(f"  결과 {i+1}: 점수={score:.4f}, 내용={chunk.content[:100]}...")
        
        print("=== 하이브리드 벡터 스토어 테스트 완료 ===")
        return True
        
    except Exception as e:
        print(f"하이브리드 벡터 스토어 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # FAISS 인덱스 재구성
    success = reconstruct_faiss_index()
    
    if success:
        # 재구성된 벡터 스토어 테스트
        test_hybrid_vector_store()
    else:
        print("FAISS 인덱스 재구성에 실패했습니다.")
