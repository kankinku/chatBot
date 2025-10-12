"""
PDF 파일을 텍스트로 추출하고 벡터 DB에 저장하는 스크립트
"""
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


def clean_text(text):
    """텍스트에서 잘못된 유니코드 문자를 제거합니다."""
    # 서로게이트 문자와 기타 문제 문자 제거
    cleaned = ""
    for char in text:
        try:
            char.encode('utf-8')
            cleaned += char
        except UnicodeEncodeError:
            cleaned += " "
    return cleaned


def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 텍스트를 추출합니다."""
    print(f"PDF 파일 읽기: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text += page_text
        print(f"  페이지 {page_num + 1}/{len(reader.pages)} 추출 완료")
    
    # 텍스트 정리
    text = clean_text(text)
    return text


def split_text(text, chunk_size=500):
    """텍스트를 청크 단위로 분할합니다."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def build_vectordb(data_folder="./data", db_path="./vectordb"):
    """PDF 파일들을 처리하여 벡터 DB를 구축합니다."""
    
    # 임베딩 모델 로드 (고정된 기본 모델)
    print("임베딩 모델 로드 중...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # ChromaDB 클라이언트 생성
    print("벡터 DB 초기화 중...")
    client = chromadb.PersistentClient(path=db_path)
    
    # 기존 컬렉션이 있으면 삭제
    try:
        client.delete_collection(name="pdf_documents")
    except:
        pass
    
    # 새 컬렉션 생성
    collection = client.create_collection(name="pdf_documents")
    
    # PDF 파일 처리
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"경고: {data_folder} 폴더에 PDF 파일이 없습니다.")
        return
    
    print(f"\n총 {len(pdf_files)}개의 PDF 파일 처리 시작")
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_id = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        
        # 텍스트 추출
        text = extract_text_from_pdf(pdf_path)
        
        # 텍스트 분할
        chunks = split_text(text)
        print(f"  {len(chunks)}개의 청크로 분할됨")
        
        # 청크와 메타데이터 저장
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": pdf_file, "chunk_id": chunk_id})
            all_ids.append(f"doc_{chunk_id}")
            chunk_id += 1
    
    # 임베딩 생성
    print(f"\n총 {len(all_chunks)}개 청크에 대한 임베딩 생성 중...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    
    # 벡터 DB에 저장
    print("벡터 DB에 저장 중...")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    print(f"\n완료! 벡터 DB가 {db_path}에 저장되었습니다.")
    print(f"총 {len(all_chunks)}개의 문서 청크가 인덱싱되었습니다.")


if __name__ == "__main__":
    build_vectordb()

