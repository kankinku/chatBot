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
project_root = Path(__file__).parent.parent  # scripts의 부모 디렉토리
sys.path.insert(0, str(project_root))

from modules.preprocessing.pdf_extractor import PDFExtractor
from modules.preprocessing.text_cleaner import TextCleaner
from modules.preprocessing.ocr_corrector import OCRCorrector
from modules.chunking.sliding_window_chunker import SlidingWindowChunker
from modules.chunking.numeric_chunker import NumericChunker
from modules.core.types import Chunk
from modules.core.logger import setup_logging, get_logger
from config.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_USE_PAGE_BASED_CHUNKING,
)

setup_logging(log_dir="logs", log_level="INFO", log_format="json")
logger = get_logger(__name__)


def process_pdf(
    pdf_path: Path,
    use_ocr_correction: bool = False,  # 기본값 False (속도 우선)
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    use_page_based_chunking: bool = DEFAULT_USE_PAGE_BASED_CHUNKING,
) -> List[Chunk]:
    """
    단일 PDF 파일 처리 (페이지별 청킹 지원)
    
    Args:
        pdf_path: PDF 파일 경로
        use_ocr_correction: OCR 오류 수정 여부
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        use_page_based_chunking: 페이지별 청킹 사용 여부
        
    Returns:
        생성된 청크 리스트
    """
    logger.info(f"Processing PDF: {pdf_path.name}")
    
    # 1. PDF 텍스트 추출
    extractor = PDFExtractor()
    
    try:
        if use_page_based_chunking:
            # 페이지별 추출
            pages = extractor.extract_pages_from_file(str(pdf_path))
            logger.info(f"Extracted {len(pages)} pages")
        else:
            # 전체 텍스트 추출
            text = extractor.extract_text_from_file(str(pdf_path))
            pages = [(None, text)]
            logger.info(f"Extracted text length: {len(text)} chars")
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}", exc_info=True)
        return []
    
    if not pages or (len(pages) == 1 and len(pages[0][1]) < 50):
        logger.warning(f"No meaningful text extracted from {pdf_path.name}")
        return []
    
    # 청킹 설정
    from modules.chunking.base_chunker import ChunkingConfig
    
    chunking_config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_numeric_chunking=True,
        numeric_context_window=3,
        preserve_table_context=True,
    )
    
    sliding_chunker = SlidingWindowChunker(config=chunking_config)
    numeric_chunker = NumericChunker(config=chunking_config)
    
    all_base_chunks = []
    all_numeric_chunks = []
    
    # 페이지별 처리
    for page_num, page_text in pages:
        if not page_text or len(page_text) < 10:
            continue
        
        # 2. OCR 오류 수정 (선택적)
        if use_ocr_correction:
            corrector = OCRCorrector()
            page_text = corrector.correct_single(page_text)
        
        # 3. 텍스트 정제
        cleaner = TextCleaner()
        page_text = cleaner.clean(page_text)
        
        # 4. 슬라이딩 윈도우 청킹
        base_chunks = sliding_chunker.chunk_text(
            text=page_text,
            doc_id=pdf_path.stem,
            filename=pdf_path.name,
            page=page_num,
        )
        
        all_base_chunks.extend(base_chunks)
        
        # 5. 숫자 특화 청크 추가
        numeric_chunks = numeric_chunker._create_numeric_expanded_chunks(
            doc_id=pdf_path.stem,
            filename=pdf_path.name,
            full_text=page_text,
            base_chunks=base_chunks,
            page=page_num,
        )
        
        all_numeric_chunks.extend(numeric_chunks)
    
    logger.info(f"Base chunks: {len(all_base_chunks)}, Numeric chunks: {len(all_numeric_chunks)}")
    
    # 6. 청크 병합
    all_chunks = all_base_chunks + all_numeric_chunks
    
    # 7. 중복 제거
    all_chunks = deduplicate_chunks(all_chunks)
    
    logger.info(f"Total chunks after deduplication: {len(all_chunks)}")
    
    return all_chunks


def deduplicate_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    corpus 생성 시 중복 청크 제거
    
    Args:
        chunks: 청크 리스트
        
    Returns:
        중복이 제거된 청크 리스트
    """
    from modules.filtering.deduplicator import Deduplicator
    from config.pipeline_config import DeduplicationConfig
    from modules.core.types import RetrievedSpan
    
    if not chunks:
        return chunks
    
    logger.info(f"Deduplicating {len(chunks)} chunks")
    
    # 중복 제거 설정
    dedup_config = DeduplicationConfig(
        jaccard_threshold=0.9,
        semantic_threshold=0.0,
        enable_semantic_dedup=False,
        min_chunk_length=50,
    )
    
    deduplicator = Deduplicator(dedup_config)
    
    # RetrievedSpan으로 변환
    spans = [
        RetrievedSpan(
            chunk=chunk,
            source="chunking",
            score=1.0,
            rank=i + 1,
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # 중복 제거
    deduplicated_spans = deduplicator.deduplicate(spans)
    
    # 다시 Chunk로 변환
    deduplicated_chunks = [span.chunk for span in deduplicated_spans]
    
    logger.info(f"Removed {len(chunks) - len(deduplicated_chunks)} duplicate chunks")
    
    return deduplicated_chunks


def save_corpus(chunks: List[Chunk], output_path: Path):
    """
    청크를 JSONL 형식으로 저장 (확장된 메타데이터 포함)
    
    Args:
        chunks: 청크 리스트
        output_path: 출력 파일 경로
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # Chunk를 dict로 변환 (모든 메타데이터 포함)
            chunk_dict = {
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "start_offset": chunk.start_offset,
                "length": chunk.length,
                "text": chunk.text,
                "neighbor_hint": chunk.neighbor_hint,  # 이웃 정보
                "extra": chunk.extra,  # 측정값 등 추가 메타데이터
            }
            
            f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
    
    logger.info(f"Corpus saved to: {output_path}")


def load_corpus(corpus_path: Path) -> List[Chunk]:
    """
    JSONL 형식의 corpus 파일을 로드 (확장된 메타데이터 포함)
    
    Args:
        corpus_path: corpus 파일 경로
        
    Returns:
        청크 리스트
    """
    chunks = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            # neighbor_hint 처리 (tuple로 변환)
            neighbor_hint = data.get("neighbor_hint")
            if neighbor_hint and isinstance(neighbor_hint, list):
                neighbor_hint = tuple(neighbor_hint)
            
            chunk = Chunk(
                doc_id=data["doc_id"],
                filename=data["filename"],
                page=data.get("page"),
                start_offset=data.get("start_offset", 0),
                length=data.get("length", len(data["text"])),
                text=data["text"],
                neighbor_hint=neighbor_hint,  # 이웃 정보 복원
                extra=data.get("extra", {}),  # 추가 메타데이터 복원
            )
            chunks.append(chunk)
    
    return chunks


def main():
    parser = argparse.ArgumentParser(description="PDF로부터 Corpus 생성 (개선된 청킹)")
    parser.add_argument("--pdf-dir", default="../data", help="PDF 디렉토리")
    parser.add_argument("--output", default="../data/corpus.jsonl", help="출력 파일명")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"청크 크기 (디폴트: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help=f"청크 오버랩 (디폴트: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--use-ocr-correction", action="store_true", help="OCR 수정 활성화 (느림, LLM 필요)")
    parser.add_argument("--no-page-based", action="store_true", help=f"페이지별 청킹 비활성화 (디폴트: {'활성화' if DEFAULT_USE_PAGE_BASED_CHUNKING else '비활성화'})")
    parser.add_argument("--pattern", default="*.pdf", help="PDF 파일 패턴")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PDF Corpus 생성 시작")
    logger.info("=" * 80)
    logger.info(f"PDF 디렉토리: {args.pdf_dir}")
    logger.info(f"출력 파일: {args.output}")
    logger.info(f"청크 크기: {args.chunk_size}")
    logger.info(f"청크 오버랩: {args.chunk_overlap}")
    logger.info(f"OCR 수정: {'활성화' if args.use_ocr_correction else '비활성화 (속도 우선)'}")
    logger.info(f"페이지별 청킹: {'비활성화' if args.no_page_based else '활성화'}")
    
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
                use_ocr_correction=args.use_ocr_correction,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                use_page_based_chunking=not args.no_page_based,
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
    
    # 측정값 통계
    measurements_count = sum(1 for chunk in all_chunks if chunk.extra.get('measurements'))
    logger.info(f"\n측정값 포함 청크: {measurements_count}/{len(all_chunks)}")
    
    # 이웃 정보 통계
    neighbor_count = sum(1 for chunk in all_chunks if chunk.neighbor_hint)
    logger.info(f"이웃 정보 포함 청크: {neighbor_count}/{len(all_chunks)}")


if __name__ == "__main__":
    main()

