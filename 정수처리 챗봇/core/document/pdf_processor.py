"""
PDF 텍스트 추출 및 임베딩 생성 모듈

이 모듈은 PDF 파일에서 텍스트를 추출하고, 의미적 검색을 위한 임베딩을 생성합니다.
여러 PDF 라이브러리를 사용하여 최대한 많은 텍스트를 추출하려고 시도합니다.
키워드 추출 기능이 통합되어 있습니다.
메모리 최적화 기능이 추가되었습니다.
"""

import os
import re
import logging
import io
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

# OCR 라이브러리 추가
try:
    import pytesseract
    from PIL import Image
    import cv2
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    # logger는 나중에 정의되므로 print 사용
    print("OCR 라이브러리가 설치되지 않았습니다. pip install pytesseract pillow opencv-python")

# 키워드 추출기 import
from .pdf_keyword_extractor import PDFKeywordExtractor
from .units import normalize_unit, is_excluded_numeric_context
# 메모리 최적화 import
from core.utils.memory_optimizer import memory_profiler, model_memory_manager

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
    pdf_id: Optional[str] = None
    filename: Optional[str] = None
    upload_time: Optional[str] = None

class PDFProcessor:
    """
    PDF 파일 처리 및 임베딩 생성 클래스
    
    주요 기능:
    1. 다중 PDF 라이브러리를 사용한 텍스트 추출
    2. 텍스트 청크화 (의미적 단위로 분할)
    3. 임베딩 생성 (sentence-transformers 사용)
    4. 메타데이터 추출 및 관리
    5. 메모리 최적화 처리
    """
    
    def __init__(self, 
                 embedding_model: str = "jhgan/ko-sroberta-multitask",
                 chunk_size: int = 256,
                 chunk_overlap: int = 30,
                 enable_keyword_extraction: bool = True,
                 keyword_cache_threshold: int = 5,
                 max_memory_usage_gb: float = 2.0):
        """
        PDFProcessor 초기화
        
        Args:
            embedding_model: 한국어 특화 임베딩 모델
            chunk_size: 청크 크기 (토큰 단위)
            chunk_overlap: 청크 간 겹치는 부분
            enable_keyword_extraction: 키워드 추출 기능 활성화 여부
            keyword_cache_threshold: 키워드 추가 임계값
            max_memory_usage_gb: 최대 메모리 사용량 (GB)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_keyword_extraction = enable_keyword_extraction
        self.max_memory_usage_gb = max_memory_usage_gb
        
        # 키워드 추출기 초기화
        if self.enable_keyword_extraction:
            self.keyword_extractor = PDFKeywordExtractor(cache_threshold=keyword_cache_threshold)
            logger.debug(f"키워드 추출 기능 활성화 (임계값: {keyword_cache_threshold})")
        
        # 한국어 특화 임베딩 모델 로드 (지연 로딩)
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self._embedding_model_loaded = False
        
        # TF-IDF 벡터라이저 (키워드 기반 검색용)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # 한국어 불용어는 별도 처리
            ngram_range=(1, 2)
        )
        
        logger.info(f"PDFProcessor 초기화 완료 (최대 메모리: {max_memory_usage_gb}GB)")
    
    def _ensure_embedding_model_loaded(self):
        """임베딩 모델 지연 로딩"""
        if not self._embedding_model_loaded:
            with memory_profiler(f"임베딩 모델 로딩: {self.embedding_model_name}"):
                try:
                    self.embedding_model = SentenceTransformer(self.embedding_model_name)
                    logger.info(f"임베딩 모델 로드 완료: {self.embedding_model_name}")
                    
                    # 메모리 매니저에 등록
                    model_memory_manager.register_model(
                        self.embedding_model_name, 
                        self.embedding_model, 
                        estimated_memory_gb=0.5
                    )
                    
                    self._embedding_model_loaded = True
                except Exception as e:
                    logger.warning(f"한국어 모델 로드 실패, 기본 모델 사용: {e}")
                    self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                    self._embedding_model_loaded = True
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리 및 UTF-8 인코딩 문제 해결 (한국어 레이아웃·공백·하이픈 보정)"""
        if not text:
            return ""
        
        try:
            # UTF-8 인코딩 문제가 있는 문자들을 안전하게 처리
            # surrogate 문자들을 제거하거나 대체
            text = text.encode('utf-8', errors='replace').decode('utf-8')
            
            # 추가적인 텍스트 정리
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)  # 제어 문자 제거
            text = re.sub(r'\s+', ' ', text)  # 연속된 공백을 하나로
            
            # PDF 특화 정리
            text = re.sub(r'[^\w\s가-힣.,!?;:()\[\]{}"\'-]', ' ', text)  # 특수문자 정리
            text = re.sub(r'\s+', ' ', text)  # 다시 공백 정리
            
            # 한국어 텍스트 정규화 (과도 결합 방지: 문장부호/단위 앞뒤는 유지)
            # 하이픈으로 잘린 단어 재결합 (줄 끝 하이픈)
            text = re.sub(r"-\s*\n\s*", "", text)
            # 문단 내 줄바꿈을 공백으로 축소
            text = re.sub(r"\s*\n\s*", " ", text)
            # 한글 사이 과도 공백만 축소(완전 제거가 아니라 1칸 유지)
            text = re.sub(r"([가-힣])\s{2,}([가-힣])", r"\1 \2", text)
            # 숫자 사이 과도 공백 축소
            text = re.sub(r"([0-9])\s{2,}([0-9])", r"\1\2", text)
            
            # 불필요한 패턴 제거
            text = re.sub(r'^\s*[-=]+\s*$', '', text, flags=re.MULTILINE)  # 구분선 제거
            text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # 단독 숫자 제거
            
            text = text.strip()
            
            return text
        except Exception as e:
            logger.warning(f"텍스트 정리 중 에러: {e}")
            return text.encode('ascii', errors='ignore').decode('ascii')
    
    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        OCR을 사용하여 PDF에서 텍스트 추출 (이미지 기반 PDF용)
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        if not OCR_AVAILABLE:
            return ""
        
        try:
            # PyMuPDF로 PDF를 이미지로 변환
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # 페이지를 이미지로 변환 (고해상도)
                mat = fitz.Matrix(2.0, 2.0)  # 2배 확대
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # PIL Image로 변환
                image = Image.open(io.BytesIO(img_data))
                
                # 이미지 전처리 (OCR 정확도 향상)
                processed_image = self._preprocess_image_for_ocr(image)
                
                # OCR 수행 (한국어 + 영어)
                text = pytesseract.image_to_string(
                    processed_image, 
                    lang='kor+eng',
                    config='--psm 6'  # 단일 텍스트 블록으로 인식
                )
                
                if text.strip():
                    cleaned_text = self._clean_text(text)
                    if cleaned_text:
                        full_text += f"\n--- 페이지 {page_num + 1} ---\n{cleaned_text}\n"
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"OCR 텍스트 추출 실패: {e}")
            return ""
    
    def _preprocess_image_for_ocr(self, image) -> 'Image.Image':
        """
        OCR 정확도 향상을 위한 이미지 전처리
        
        Args:
            image: 원본 이미지
            
        Returns:
            전처리된 이미지
        """
        try:
            # PIL Image를 OpenCV 형식으로 변환
            img_array = np.array(image)
            
            # 그레이스케일 변환
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # 노이즈 제거
            denoised = cv2.medianBlur(gray, 3)
            
            # 대비 향상
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 이진화 (Otsu 방법)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 다시 PIL Image로 변환
            processed_image = Image.fromarray(binary)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"이미지 전처리 실패: {e}")
            return image

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        다중 라이브러리를 사용하여 PDF에서 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            (전체_텍스트, 메타데이터)
        """
        with memory_profiler(f"PDF 텍스트 추출: {os.path.basename(pdf_path)}"):
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
                            cleaned_text = self._clean_text(text)
                            if cleaned_text:
                                full_text += f"\n--- 페이지 {page_num + 1} ---\n{cleaned_text}\n"
                    
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
                            cleaned_text = self._clean_text(text)
                            if cleaned_text:
                                full_text += f"\n--- 페이지 {page_num + 1} ---\n{cleaned_text}\n"
                    
                    if full_text.strip():
                        metadata["extraction_method"].append("PyMuPDF")
                        logger.info("PyMuPDF로 텍스트 추출 성공")
                        
                    doc.close()
                    
                except Exception as e:
                    logger.warning(f"PyMuPDF 추출 실패: {e}")
            
            # 3. pdfplumber로 시도
            if not full_text.strip():
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        metadata["pages"] = len(pdf.pages)
                        
                        for page_num, page in enumerate(pdf.pages):
                            text = page.extract_text()
                            if text and text.strip():
                                cleaned_text = self._clean_text(text)
                                if cleaned_text:
                                    full_text += f"\n--- 페이지 {page_num + 1} ---\n{cleaned_text}\n"
                        
                        if full_text.strip():
                            metadata["extraction_method"].append("pdfplumber")
                            logger.info("pdfplumber로 텍스트 추출 성공")
                            
                except Exception as e:
                    logger.warning(f"pdfplumber 추출 실패: {e}")
            
            # 4. OCR 기반 텍스트 추출 (이미지 기반 PDF용)
            if not full_text.strip() and OCR_AVAILABLE:
                try:
                    ocr_text = self._extract_text_with_ocr(pdf_path)
                    if ocr_text.strip():
                        full_text = ocr_text
                        metadata["extraction_method"].append("OCR")
                        logger.info("OCR로 텍스트 추출 성공")
                        
                except Exception as e:
                    logger.warning(f"OCR 추출 실패: {e}")
            
            # 메모리 정리
            if not full_text.strip():
                logger.error(f"모든 PDF 라이브러리로 텍스트 추출 실패: {pdf_path}")
                return "", metadata
            
            return full_text.strip(), metadata
    
    def chunk_text(self, text: str, pdf_id: str = None) -> List[TextChunk]:
        """
        텍스트를 의미적 단위로 청크화 (메모리 최적화 적용)
        
        Args:
            text: 원본 텍스트
            pdf_id: PDF 식별자
            
        Returns:
            TextChunk 리스트
        """
        with memory_profiler(f"텍스트 청크화: {pdf_id or 'unknown'}"):
            if not text.strip():
                return []
            
            chunks = []
            lines = text.split('\n')
            current_chunk = []
            current_length = 0
            page_number = 1
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 페이지 구분자 확인
                if line.startswith('--- 페이지') and '---' in line:
                    try:
                        page_match = re.search(r'페이지 (\d+)', line)
                        if page_match:
                            page_number = int(page_match.group(1))
                    except:
                        pass
                    continue
                
                # 청크 크기 제한 체크
                if current_length + len(line) > self.chunk_size:
                    if current_chunk:
                        # 현재 청크 저장
                        chunk_content = '\n'.join(current_chunk)
                        cleaned_content = self._clean_text(chunk_content)
                        if cleaned_content:  # 정리된 텍스트가 있을 때만 청크 생성
                            chunk_id = self._generate_chunk_id(cleaned_content, pdf_id, len(chunks))
                            
                            chunks.append(TextChunk(
                                content=cleaned_content,
                                page_number=page_number,
                                chunk_id=chunk_id,
                                pdf_id=pdf_id,
                                metadata={"pdf_id": pdf_id, "chunk_index": len(chunks)}
                            ))
                        
                        # 오버랩을 위한 부분 유지
                        overlap_lines = current_chunk[-self.chunk_overlap//50:] if self.chunk_overlap > 0 else []
                        current_chunk = overlap_lines + [line]
                        current_length = sum(len(l) for l in current_chunk)
                    else:
                        # 단일 라인이 청크 크기를 초과하는 경우
                        current_chunk = [line]
                        current_length = len(line)
                else:
                    current_chunk.append(line)
                    current_length += len(line)
            
            # 마지막 청크 처리
            if current_chunk:
                chunk_content = '\n'.join(current_chunk)
                cleaned_content = self._clean_text(chunk_content)
                if cleaned_content:  # 정리된 텍스트가 있을 때만 청크 생성
                    chunk_id = self._generate_chunk_id(cleaned_content, pdf_id, len(chunks))
                    
                    chunks.append(TextChunk(
                        content=cleaned_content,
                        page_number=page_number,
                        chunk_id=chunk_id,
                        pdf_id=pdf_id,
                        metadata={"pdf_id": pdf_id, "chunk_index": len(chunks)}
                    ))
            
            logger.info(f"텍스트 청크화 완료: {len(chunks)}개 청크 생성")
            return chunks

    # ==== 숫자·단위·대상 구조화 추출 (경량 규칙 기반) ====
    _NUM_PATTERN = re.compile(
        r"(?P<val1>\d+(?:[\,\.]\d+)?)\s*(?:(?P<range>[~\-–]|~|to|~\s*|-)\s*(?P<val2>\d+(?:[\,\.]\d+)?))?\s*(?P<unit>%|mg\s*/\s*L|㎎\s*/\s*L|mg/L|mg·l-1|m³/h|m3/h|㎥/h|분|시간|초)?",
        re.IGNORECASE,
    )

    def _extract_measurements_from_text(self, text: str, page_number: int) -> List[Dict]:
        measurements: List[Dict] = []
        if not text or is_excluded_numeric_context(text):
            return measurements

        # 간단한 토큰 시퀀스 생성
        tokens = re.split(r"(\s+)", text)
        for match in self._NUM_PATTERN.finditer(text):
            unit_raw = match.group("unit") or ""
            unit = normalize_unit(unit_raw) if unit_raw else None

            val1_raw = match.group("val1")
            val2_raw = match.group("val2")
            rng = match.group("range")

            # 값 정규화(쉼표 제거)
            def _norm_num(v: Optional[str]) -> Optional[float]:
                if not v:
                    return None
                try:
                    return float(v.replace(",", ""))
                except Exception:
                    return None

            v1 = _norm_num(val1_raw)
            v2 = _norm_num(val2_raw)

            # 주변 문맥에서 대상 후보 추출(숫자 주변 8~12자 범위)
            start, end = match.span()
            left_ctx = text[max(0, start - 24):start]
            right_ctx = text[end:min(len(text), end + 24)]
            # 조사 제거 단순 휴리스틱
            def _clean_noun(s: str) -> str:
                s = re.sub(r"[\s\,\.:;\(\)\[\]]+", " ", s)
                s = re.sub(r"\b(을|를|은|는|이|가|과|와|의|에|로|에서|까지|부터)$", "", s.strip())
                return s[-12:].strip() if s else s

            target_left = _clean_noun(left_ctx)
            target_right = _clean_noun(right_ctx)
            # 우선 왼쪽 명사성 문구를 사용, 없으면 오른쪽 사용
            target = target_left or target_right or None

            mtype = "range" if (rng and v1 is not None and v2 is not None) else "exact"

            # 최소 유효성: unit 또는 대상 중 하나는 있어야 함
            if v1 is None:
                continue

            record: Dict = {
                "target": target,
                "type": mtype,
                "unit": unit,
                "value": v1 if mtype == "exact" else [v1, v2],
                "page": page_number,
            }
            measurements.append(record)

        return measurements
    
    def _generate_chunk_id(self, content: str, pdf_id: str, chunk_index: int) -> str:
        """청크 ID 생성"""
        try:
            # UTF-8 인코딩 에러 방지를 위해 errors='replace' 사용
            content_hash = hashlib.md5(content.encode('utf-8', errors='replace')).hexdigest()[:8]
        except Exception as e:
            logger.warning(f"청크 ID 생성 중 인코딩 에러: {e}, 기본 해시 사용")
            content_hash = hashlib.md5(f"chunk_{chunk_index}".encode('utf-8')).hexdigest()[:8]
        return f"{pdf_id}_{chunk_index}_{content_hash}" if pdf_id else f"chunk_{chunk_index}_{content_hash}"
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        청크들의 임베딩 생성 (메모리 최적화 적용)
        
        Args:
            chunks: TextChunk 리스트
            
        Returns:
            임베딩이 추가된 TextChunk 리스트
        """
        if not chunks:
            return chunks
        
        with memory_profiler(f"임베딩 생성: {len(chunks)}개 청크"):
            # 임베딩 모델 로드 확인
            self._ensure_embedding_model_loaded()
            
            # 배치 크기 계산 (메모리 제한 고려)
            batch_size = self._calculate_batch_size(len(chunks))
            
            processed_chunks = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # 배치 텍스트 추출
                texts = [chunk.content for chunk in batch_chunks]
                
                # 임베딩 생성
                try:
                    embeddings = self.embedding_model.encode(
                        texts,
                        batch_size=min(batch_size, 32),  # 내부 배치 크기 제한
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    # 임베딩 할당
                    for j, chunk in enumerate(batch_chunks):
                        chunk.embedding = embeddings[j]
                        processed_chunks.append(chunk)
                    
                    logger.debug(f"임베딩 배치 처리 완료: {i+1}-{min(i+batch_size, len(chunks))}/{len(chunks)}")
                    
                except Exception as e:
                    logger.error(f"임베딩 생성 실패: {e}")
                    # 임베딩 생성 실패 시에도 청크는 유지
                    processed_chunks.extend(batch_chunks)
            
            logger.info(f"임베딩 생성 완료: {len(processed_chunks)}개 청크")
            return processed_chunks
    
    def _calculate_batch_size(self, total_chunks: int) -> int:
        """메모리 제한을 고려한 배치 크기 계산"""
        # 기본 배치 크기
        base_batch_size = 16
        
        # 메모리 사용량에 따른 조정
        memory_factor = self.max_memory_usage_gb / 2.0  # 2GB 기준
        adjusted_batch_size = int(base_batch_size * memory_factor)
        
        # 최소/최대 제한
        min_batch_size = 4
        max_batch_size = 64
        
        batch_size = max(min_batch_size, min(adjusted_batch_size, max_batch_size))
        
        # 총 청크 수 고려
        batch_size = min(batch_size, total_chunks)
        
        return batch_size
    
    def process_pdf(self, pdf_path: str, pdf_id: str = None) -> Tuple[List[TextChunk], Dict]:
        """
        PDF 파일 전체 처리 (메모리 최적화 적용)
        
        Args:
            pdf_path: PDF 파일 경로
            pdf_id: PDF 식별자
            
        Returns:
            (처리된_청크_리스트, 메타데이터)
        """
        with memory_profiler(f"PDF 전체 처리: {os.path.basename(pdf_path)}"):
            try:
                # 1. 텍스트 추출
                text, metadata = self.extract_text_from_pdf(pdf_path)
                if not text:
                    return [], metadata
                
                # 2. 텍스트 청크화
                chunks = self.chunk_text(text, pdf_id)
                if not chunks:
                    return [], metadata
                
                # 3. 임베딩 생성
                chunks_with_embeddings = self.generate_embeddings(chunks)
                
                # 4. 파일명과 업로드 시간 설정
                filename = os.path.basename(pdf_path)
                upload_time = metadata.get("upload_time", "")
                for chunk in chunks_with_embeddings:
                    chunk.filename = filename
                    chunk.upload_time = upload_time
                    # 숫자·단위 측정치 추출 후 메타데이터에 저장
                    try:
                        ms = self._extract_measurements_from_text(chunk.content, chunk.page_number)
                        if ms:
                            if chunk.metadata is None:
                                chunk.metadata = {}
                            chunk.metadata["measurements"] = ms
                    except Exception as _e:
                        logger.debug(f"측정치 추출 스킵: {_e}")
                
                # 5. 키워드 추출 (선택적)
                if self.enable_keyword_extraction:
                    self._extract_keywords(chunks_with_embeddings, pdf_id)
                
                # 메타데이터 업데이트
                metadata.update({
                    "total_chunks": len(chunks_with_embeddings),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "embedding_model": self.embedding_model_name,
                    "processing_complete": True
                })
                
                logger.info(f"PDF 처리 완료: {os.path.basename(pdf_path)} - {len(chunks_with_embeddings)}개 청크")
                return chunks_with_embeddings, metadata
                
            except Exception as e:
                logger.error(f"PDF 처리 실패 {pdf_path}: {e}")
                return [], {"error": str(e), "processing_complete": False}
    
    def _extract_keywords(self, chunks: List[TextChunk], pdf_id: str):
        """키워드 추출 (메모리 최적화 적용)"""
        if not self.enable_keyword_extraction or not chunks:
            return
        
        with memory_profiler(f"키워드 추출: {pdf_id or 'unknown'}"):
            try:
                # 각 청크별 키워드 추출 및 저장
                for chunk in chunks:
                    per_chunk_keywords = self.keyword_extractor.extract_keywords_from_text(chunk.content)
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata["keywords"] = per_chunk_keywords
                
                # 전체 빈도 기반 파이프라인 키워드 업데이트 시도
                if hasattr(self.keyword_extractor, 'add_keywords_to_pipeline'):
                    self.keyword_extractor.add_keywords_to_pipeline()
                
                logger.debug("키워드 추출 완료(청크 단위)")
                
            except Exception as e:
                logger.warning(f"키워드 추출 실패: {e}")
    
    def cleanup(self):
        """메모리 정리"""
        try:
            # 임베딩 모델 언로드
            if self.embedding_model is not None:
                model_memory_manager.unload_model(self.embedding_model_name)
                self.embedding_model = None
                self._embedding_model_loaded = False
            
            # TF-IDF 벡터라이저 정리
            if hasattr(self, 'tfidf_vectorizer'):
                del self.tfidf_vectorizer
            
            logger.info("PDFProcessor 메모리 정리 완료")
            
        except Exception as e:
            logger.error(f"PDFProcessor 메모리 정리 실패: {e}")

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
