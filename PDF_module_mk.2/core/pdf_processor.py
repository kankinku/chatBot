"""
PDF 텍스트 추출 및 임베딩 생성 모듈

이 모듈은 PDF 파일에서 텍스트를 추출하고, 의미적 검색을 위한 임베딩을 생성합니다.
여러 PDF 라이브러리를 사용하여 최대한 많은 텍스트를 추출하려고 시도합니다.
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib

import PyPDF2
import fitz  # pymupdf
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """텍스트 청크 데이터 클래스"""
    content: str
    page_number: int
    chunk_id: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

class PDFProcessor:
    """
    PDF 파일 처리 및 임베딩 생성 클래스
    
    주요 기능:
    1. 다중 PDF 라이브러리를 사용한 텍스트 추출
    2. 텍스트 청크화 (의미적 단위로 분할)
    3. 임베딩 생성 (sentence-transformers 사용)
    4. 메타데이터 추출 및 관리
    """
    
    def __init__(self, 
                 embedding_model: str = "jhgan/ko-sroberta-multitask",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        PDFProcessor 초기화
        
        Args:
            embedding_model: 한국어 특화 임베딩 모델
            chunk_size: 청크 크기 (토큰 단위)
            chunk_overlap: 청크 간 겹치는 부분
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 한국어 특화 임베딩 모델 로드
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"임베딩 모델 로드 완료: {embedding_model}")
        except Exception as e:
            logger.warning(f"한국어 모델 로드 실패, 기본 모델 사용: {e}")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # TF-IDF 벡터라이저 (키워드 기반 검색용)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # 한국어 불용어는 별도 처리
            ngram_range=(1, 2)
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        다중 라이브러리를 사용하여 PDF에서 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            (전체_텍스트, 메타데이터)
        """
        full_text = ""
        metadata = {"pages": 0, "extraction_method": []}
        
        # 1. PyPDF2로 시도
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["pages"] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        full_text += f"\n--- 페이지 {page_num + 1} ---\n{text}\n"
                
                if full_text.strip():
                    metadata["extraction_method"].append("PyPDF2")
                    logger.info("PyPDF2로 텍스트 추출 성공")
                    
        except Exception as e:
            logger.warning(f"PyPDF2 추출 실패: {e}")
        
        # 2. PyMuPDF (fitz)로 보완
        if not full_text.strip():
            try:
                doc = fitz.open(pdf_path)
                metadata["pages"] = doc.page_count
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        full_text += f"\n--- 페이지 {page_num + 1} ---\n{text}\n"
                
                if full_text.strip():
                    metadata["extraction_method"].append("PyMuPDF")
                    logger.info("PyMuPDF로 텍스트 추출 성공")
                    
                doc.close()
                
            except Exception as e:
                logger.warning(f"PyMuPDF 추출 실패: {e}")
        
        # 3. pdfplumber로 최종 시도
        if not full_text.strip():
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    metadata["pages"] = len(pdf.pages)
                    
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            full_text += f"\n--- 페이지 {page_num + 1} ---\n{text}\n"
                    
                    if full_text.strip():
                        metadata["extraction_method"].append("pdfplumber")
                        logger.info("pdfplumber로 텍스트 추출 성공")
                        
            except Exception as e:
                logger.error(f"모든 PDF 추출 방법 실패: {e}")
                raise Exception("PDF 텍스트 추출에 실패했습니다.")
        
        # 텍스트 전처리
        full_text = self._preprocess_text(full_text)
        metadata["total_characters"] = len(full_text)
        
        return full_text, metadata
    
    def _preprocess_text(self, text: str) -> str:
        """
        추출된 텍스트 전처리
        
        Args:
            text: 원본 텍스트
            
        Returns:
            전처리된 텍스트
        """
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 페이지 구분자 정리
        text = re.sub(r'--- 페이지 \d+ ---', '\n\n', text)
        
        # 연속된 줄바꿈 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def create_text_chunks(self, text: str, pdf_id: str) -> List[TextChunk]:
        """
        텍스트를 의미적 단위로 청크화
        
        Args:
            text: 전체 텍스트
            pdf_id: PDF 식별자
            
        Returns:
            TextChunk 리스트
        """
        chunks = []
        
        # 문단 단위로 우선 분할
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_page = 1
        chunk_counter = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 페이지 번호 추출 (만약 있다면)
            page_match = re.search(r'페이지 (\d+)', paragraph)
            if page_match:
                current_page = int(page_match.group(1))
                continue
            
            # 청크 크기 체크
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # 현재 청크 저장
                if current_chunk.strip():
                    chunk_id = f"{pdf_id}_chunk_{chunk_counter}"
                    chunks.append(TextChunk(
                        content=current_chunk.strip(),
                        page_number=current_page,
                        chunk_id=chunk_id,
                        metadata={"pdf_id": pdf_id, "chunk_index": chunk_counter}
                    ))
                    chunk_counter += 1
                
                # 새 청크 시작 (오버랩 고려)
                current_chunk = paragraph + "\n\n"
        
        # 마지막 청크 처리
        if current_chunk.strip():
            chunk_id = f"{pdf_id}_chunk_{chunk_counter}"
            chunks.append(TextChunk(
                content=current_chunk.strip(),
                page_number=current_page,
                chunk_id=chunk_id,
                metadata={"pdf_id": pdf_id, "chunk_index": chunk_counter}
            ))
        
        logger.info(f"총 {len(chunks)}개의 텍스트 청크 생성")
        return chunks
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        텍스트 청크들에 대한 임베딩 생성
        
        Args:
            chunks: TextChunk 리스트
            
        Returns:
            임베딩이 포함된 TextChunk 리스트
        """
        texts = [chunk.content for chunk in chunks]
        
        try:
            # 배치 처리로 임베딩 생성 (메모리 효율성)
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=32, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # 각 청크에 임베딩 할당
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            logger.info(f"{len(chunks)}개 청크의 임베딩 생성 완료")
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[TextChunk], Dict]:
        """
        PDF 파일 전체 처리 파이프라인
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            (임베딩이 포함된 청크 리스트, 메타데이터)
        """
        # PDF ID 생성 (파일 경로 해시)
        pdf_id = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
        
        logger.info(f"PDF 처리 시작: {pdf_path}")
        
        # 1. 텍스트 추출
        full_text, metadata = self.extract_text_from_pdf(pdf_path)
        
        # 2. 청크화
        chunks = self.create_text_chunks(full_text, pdf_id)
        
        # 3. 임베딩 생성
        chunks_with_embeddings = self.generate_embeddings(chunks)
        
        # 메타데이터 업데이트
        metadata.update({
            "pdf_id": pdf_id,
            "total_chunks": len(chunks_with_embeddings),
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension()
        })
        
        logger.info(f"PDF 처리 완료: {len(chunks_with_embeddings)}개 청크")
        
        return chunks_with_embeddings, metadata

# 파인튜닝 관련 함수들
def prepare_training_data(pdf_chunks: List[TextChunk], 
                         qa_pairs: List[Dict]) -> Dict:
    """
    임베딩 모델 파인튜닝을 위한 데이터 준비
    
    Args:
        pdf_chunks: PDF에서 추출한 텍스트 청크들
        qa_pairs: 질문-답변 쌍 [{"question": "질문", "answer": "답변", "relevant_chunks": [chunk_ids]}]
    
    Returns:
        훈련 데이터셋
        
    파인튜닝 이유:
    - 도메인 특화 문서에 대한 검색 성능 향상
    - 한국어 질문과 텍스트 간의 의미적 유사도 계산 개선
    - 전문 용어나 개념에 대한 이해도 향상
    """
    training_data = {
        "positive_pairs": [],  # (질문, 관련_텍스트) 쌍
        "negative_pairs": [],  # (질문, 무관한_텍스트) 쌍
        "triplets": []         # (앵커, 긍정, 부정) 삼중쌍
    }
    
    chunk_dict = {chunk.chunk_id: chunk for chunk in pdf_chunks}
    
    for qa_pair in qa_pairs:
        question = qa_pair["question"]
        relevant_chunk_ids = qa_pair.get("relevant_chunks", [])
        
        # 긍정 쌍 생성
        for chunk_id in relevant_chunk_ids:
            if chunk_id in chunk_dict:
                training_data["positive_pairs"].append({
                    "question": question,
                    "text": chunk_dict[chunk_id].content
                })
        
        # 부정 쌍 생성 (관련 없는 청크들과 매칭)
        all_chunk_ids = set(chunk_dict.keys())
        negative_chunk_ids = all_chunk_ids - set(relevant_chunk_ids)
        
        for chunk_id in list(negative_chunk_ids)[:3]:  # 부정 샘플 제한
            training_data["negative_pairs"].append({
                "question": question,
                "text": chunk_dict[chunk_id].content
            })
    
    return training_data

if __name__ == "__main__":
    # 테스트 코드
    processor = PDFProcessor()
    
    # 샘플 PDF 처리 (실제 파일이 있을 때)
    # chunks, metadata = processor.process_pdf("sample.pdf")
    # print(f"처리 완료: {metadata}")
    
    print("PDFProcessor 모듈이 정상적으로 로드되었습니다.")
