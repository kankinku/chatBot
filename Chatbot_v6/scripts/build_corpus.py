#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF로부터 Corpus 생성 스크립트

여러 PDF 파일을 처리하여 하나의 JSONL corpus 파일로 변환합니다.
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.preprocessing.pdf_extractor import PDFExtractor
from modules.preprocessing.text_cleaner import TextCleaner
from modules.preprocessing.normalizer import Normalizer
from modules.preprocessing.ocr_corrector import OCRCorrector
from modules.chunking.sliding_window_chunker import SlidingWindowChunker
from modules.chunking.numeric_chunker import NumericChunker
from modules.core.types import Chunk
from modules.core.logger import setup_logging, get_logger

setup_logging(log_dir="logs", log_level="INFO", log_format="json")
logger = get_logger(__name__)


def process_pdf(
    pdf_path: Path,
    use_ocr_correction: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """
    단일 PDF 파일 처리
    
    Args:
        pdf_path: PDF 파일 경로
        use_ocr_correction: OCR 오류 수정 여부
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        
    Returns:
        생성된 청크 리스트
    """
    logger.info(f"Processing PDF: {pdf_path.name}")
    
    # 1. PDF 텍스트 추출
    extractor = PDFExtractor()
    try:
        text = extractor.extract_text_from_file(str(pdf_path))
        logger.info(f"Extracted text length: {len(text)} chars")
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}", exc_info=True)
        return []
    
    if not text or len(text) < 50:
        logger.warning(f"No meaningful text extracted from {pdf_path.name}")
        return []
    
    # 2. OCR 오류 수정 (선택적)
    if use_ocr_correction:
        corrector = OCRCorrector()
        text = corrector.correct(text)
        logger.info(f"OCR correction completed")
    
    # 3. 텍스트 정제
    cleaner = TextCleaner()
    text = cleaner.clean(text)
    logger.info(f"Text cleaning completed")
    
    # 4. 정규화
    normalizer = Normalizer()
    text = normalizer.normalize(text)
    logger.info(f"Text normalization completed")
    
    # 5. 슬라이딩 윈도우 청킹
    sliding_chunker = SlidingWindowChunker(
        chunk_size=chunk_size,
        overlap_size=chunk_overlap,
    )
    
    base_chunks = sliding_chunker.chunk(
        text=text,
        doc_id=pdf_path.stem,
        filename=pdf_path.name,
    )
    
    logger.info(f"Sliding window chunking: {len(base_chunks)} chunks")
    
    # 6. 숫자 특화 청크 추가
    numeric_chunker = NumericChunker()
    numeric_chunks = numeric_chunker.chunk(
        text=text,
        doc_id=pdf_path.stem,
        filename=pdf_path.name,
    )
    
    logger.info(f"Numeric chunking: {len(numeric_chunks)} numeric chunks")
    
    # 7. 청크 병합
    all_chunks = base_chunks + numeric_chunks
    logger.info(f"Total chunks: {len(all_chunks)}")
    
    return all_chunks


def save_corpus(chunks: List[Chunk], output_path: Path):
    """
    청크를 JSONL 형식으로 저장
    
    Args:
        chunks: 청크 리스트
        output_path: 출력 파일 경로
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # Chunk를 dict로 변환
            chunk_dict = {
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "start_offset": chunk.start_offset,
                "length": chunk.length,
                "text": chunk.text,
            }
            
            f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
    
    logger.info(f"Corpus saved to: {output_path}")


def load_corpus(corpus_path: Path) -> List[Chunk]:
    """
    JSONL 형식의 corpus 파일을 로드
    
    Args:
        corpus_path: corpus 파일 경로
        
    Returns:
        청크 리스트
    """
    chunks = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunk = Chunk(
                doc_id=data["doc_id"],
                filename=data["filename"],
                page=data.get("page", 0),
                start_offset=data.get("start_offset", 0),
                length=data.get("length", len(data["text"])),
                text=data["text"],
            )
            chunks.append(chunk)
    
    return chunks


def main():
    parser = argparse.ArgumentParser(description="PDF로부터 Corpus 생성")
    parser.add_argument("--pdf-dir", default="../data", help="PDF 디렉토리")
    parser.add_argument("--output", default="../data/corpus.jsonl", help="출력 파일명")
    parser.add_argument("--chunk-size", type=int, default=800, help="청크 크기")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="청크 오버랩")
    parser.add_argument("--no-ocr-correction", action="store_true", help="OCR 수정 비활성화")
    parser.add_argument("--pattern", default="*.pdf", help="PDF 파일 패턴")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PDF Corpus 생성 시작")
    logger.info("=" * 80)
    logger.info(f"PDF 디렉토리: {args.pdf_dir}")
    logger.info(f"출력 파일: {args.output}")
    logger.info(f"청크 크기: {args.chunk_size}")
    logger.info(f"청크 오버랩: {args.chunk_overlap}")
    logger.info(f"OCR 수정: {'비활성화' if args.no_ocr_correction else '활성화'}")
    
    # PDF 파일 찾기
    pdf_dir = Path(args.pdf_dir)
    pdf_files = list(pdf_dir.glob(f"**/{args.pattern}"))
    pdf_files = [f for f in pdf_files if f.suffix.lower() == '.pdf']
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # 각 PDF 처리
    all_chunks = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            chunks = process_pdf(
                pdf_path,
                use_ocr_correction=not args.no_ocr_correction,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}", exc_info=True)
            continue
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Total chunks generated: {len(all_chunks)}")
    
    # Corpus 저장
    output_path = Path(args.output)
    save_corpus(all_chunks, output_path)
    
    logger.info("=" * 80)
    logger.info("Corpus 생성 완료!")
    logger.info("=" * 80)
    logger.info(f"출력 파일: {output_path.absolute()}")
    logger.info(f"총 청크: {len(all_chunks)}")
    
    # 통계 출력
    from collections import Counter
    doc_counter = Counter(chunk.doc_id for chunk in all_chunks)
    logger.info("\n문서별 청크 수:")
    for doc_id, count in doc_counter.most_common():
        logger.info(f"  {doc_id}: {count}")


if __name__ == "__main__":
    main()

