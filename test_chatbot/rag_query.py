"""
RAG를 사용하여 벡터 DB에서 관련 문서를 검색하고 로컬 LLM으로 답변을 생성하는 스크립트
"""
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json


class RAGSystem:
    def __init__(self, db_path="./vectordb", model_name="llama3.1:8b-instruct-q4_K_M"):
        """RAG 시스템을 초기화합니다."""
        print("RAG 시스템 초기화 중...")
        
        # 임베딩 모델 로드
        print("임베딩 모델 로드 중...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # ChromaDB 클라이언트 연결
        print("벡터 DB 연결 중...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="pdf_documents")
        
        # Ollama 설정
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        
        print("RAG 시스템 준비 완료\n")
    
    def retrieve_documents(self, query, top_k=3):
        """쿼리와 관련된 문서를 검색합니다."""
        print(f"질의: {query}")
        print(f"관련 문서 검색 중 (top {top_k})...")
        
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_model.encode(query)
        
        # 유사 문서 검색
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        print(f"{len(documents)}개의 관련 문서 찾음\n")
        
        return documents, metadatas
    
    def generate_answer(self, query, documents):
        """검색된 문서를 바탕으로 LLM을 사용해 답변을 생성합니다."""
        # 컨텍스트 구성
        context = "\n\n".join([f"[문서 {i+1}]\n{doc}" for i, doc in enumerate(documents)])
        
        # 프롬프트 구성
        prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

참고 문서:
{context}

질문: {query}

답변:"""
        
        print("LLM 답변 생성 중...")
        
        # Ollama API 호출
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get('response', '')
            
            return answer
        
        except Exception as e:
            return f"오류 발생: {str(e)}\nOllama가 실행 중인지 확인하고 '{self.model_name}' 모델이 설치되어 있는지 확인하세요."
    
    def query(self, question, top_k=3):
        """질문에 대한 RAG 기반 답변을 생성합니다."""
        # 문서 검색
        documents, metadatas = self.retrieve_documents(question, top_k)
        
        # 검색된 문서 정보 출력
        print("검색된 문서:")
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            print(f"  {i+1}. 출처: {meta['source']}")
            print(f"     내용 미리보기: {doc[:100]}...")
        print()
        
        # 답변 생성
        answer = self.generate_answer(question, documents)
        
        return answer, documents, metadatas


if __name__ == "__main__":
    # RAG 시스템 초기화
    rag = RAGSystem()
    
    # 테스트 질의
    test_query = "이 문서의 주요 내용은 무엇인가요?"
    
    answer, docs, metas = rag.query(test_query)
    
    print("=" * 80)
    print("답변:")
    print(answer)
    print("=" * 80)

