"""
Document Loader

PDF 문서 로딩, 청킹, 캐싱 관리.
디렉토리 스캔 및 batch 처리 지원.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from modules.core.types import Chunk, DocumentMetadata
from modules.core.logger import get_logger
from modules.core.exceptions import PreprocessingError, PDFLoadError

logger = get_logger(__name__)


class DocumentLoader:
    """문서 로딩, 청킹, 캐싱 관리"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.chunks: List[Chunk] = []
        self.metadata: List[DocumentMetadata] = []
        
        logger.info("DocumentLoader initialized", data_dir=str(self.data_dir))
    
    def load_from_directory(
        self, 
        pattern: str = "*.pdf",
        use_cache: bool = True,
    ) -> List[Chunk]:
        """
        디렉토리에서 문서 로드
        
        Args:
            pattern: 파일 패턴 (glob)
            use_cache: 캐시 사용 여부
            
        Returns:
            청크 리스트
            
        Raises:
            FileNotFoundError: 문서가 없을 때
        """
        # 캐시 확인
        cache_path = self.data_dir / "chunks_cache.pkl"
        if use_cache and cache_path.exists():
            try:
                logger.info("Loading chunks from cache")
                return self.load_from_cache(str(cache_path))
            except Exception as e:
                logger.warning(f"Cache loading failed, will reload: {e}")
        
        # 파일 검색
        pdf_files = list(self.data_dir.glob(pattern))
        
        if not pdf_files:
            raise FileNotFoundError(
                f"No documents found in {self.data_dir} with pattern '{pattern}'"
            )
        
        logger.info(f"Found {len(pdf_files)} documents", pattern=pattern)
        
        # 각 파일 로드
        all_chunks = []
        for pdf_path in pdf_files:
            try:
                chunks = self._load_pdf(pdf_path)
                all_chunks.extend(chunks)
                logger.info(f"Loaded {len(chunks)} chunks from {pdf_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {pdf_path.name}: {e}", exc_info=True)
                # 개별 파일 실패는 계속 진행
                continue
        
        if not all_chunks:
            raise PreprocessingError(
                "No chunks were extracted from any document",
                error_code="E402"
            )
        
        self.chunks = all_chunks
        
        # 캐시 저장
        if use_cache:
            try:
                self.save_to_cache(str(cache_path))
            except Exception as e:
                logger.warning(f"Cache saving failed (non-critical): {e}")
        
        logger.info(f"Total {len(all_chunks)} chunks loaded from {len(pdf_files)} documents")
        
        return all_chunks
    
    def _load_pdf(self, pdf_path: Path) -> List[Chunk]:
        """
        단일 PDF 로드
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            청크 리스트
            
        Raises:
            PDFLoadError: PDF 로드 실패
        """
        try:
            from modules.preprocessing.pdf_extractor import PDFExtractor
            from modules.chunking.sliding_window_chunker import SlidingWindowChunker
            
            # PDF 텍스트 추출
            extractor = PDFExtractor()
            pages_text, page_count = extractor.extract(str(pdf_path))
            
            # 메타데이터 생성
            doc_id = pdf_path.stem  # 파일명 (확장자 제외)
            metadata = DocumentMetadata(
                doc_id=doc_id,
                filename=pdf_path.name,
                file_path=str(pdf_path),
                total_pages=page_count,
                total_chars=sum(len(text) for text in pages_text),
                created_at=datetime.now().isoformat(),
                processed_at=datetime.now().isoformat(),
            )
            self.metadata.append(metadata)
            
            # 청킹
            chunker = SlidingWindowChunker()
            chunks = chunker.chunk_pages(
                pages_text=pages_text,
                doc_id=doc_id,
                filename=pdf_path.name,
            )
            
            return chunks
        
        except Exception as e:
            raise PDFLoadError(
                file_path=str(pdf_path),
                cause=e
            ) from e
    
    def save_to_cache(self, cache_path: str) -> None:
        """
        청크를 캐시에 저장
        
        Args:
            cache_path: 캐시 파일 경로
        """
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "chunks": self.chunks,
            "metadata": self.metadata,
            "created_at": datetime.now().isoformat(),
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved {len(self.chunks)} chunks to cache", path=cache_path)
    
    @classmethod
    def load_from_cache(cls, cache_path: str) -> List[Chunk]:
        """
        캐시에서 청크 로드
        
        Args:
            cache_path: 캐시 파일 경로
            
        Returns:
            청크 리스트
        """
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        chunks = cache_data.get("chunks", [])
        created_at = cache_data.get("created_at", "unknown")
        
        logger.info(
            f"Loaded {len(chunks)} chunks from cache",
            path=cache_path,
            cache_created=created_at
        )
        
        return chunks
    
    def get_document_by_id(self, doc_id: str) -> Optional[DocumentMetadata]:
        """
        문서 ID로 메타데이터 조회
        
        Args:
            doc_id: 문서 ID
            
        Returns:
            문서 메타데이터 또는 None
        """
        for meta in self.metadata:
            if meta.doc_id == doc_id:
                return meta
        return None
    
    def get_chunks_by_doc_id(self, doc_id: str) -> List[Chunk]:
        """
        문서 ID로 청크 조회
        
        Args:
            doc_id: 문서 ID
            
        Returns:
            해당 문서의 청크 리스트
        """
        return [chunk for chunk in self.chunks if chunk.doc_id == doc_id]
    
    def clear_cache(self, cache_path: Optional[str] = None) -> None:
        """
        캐시 삭제
        
        Args:
            cache_path: 캐시 파일 경로 (None이면 기본 경로)
        """
        if cache_path is None:
            cache_path = str(self.data_dir / "chunks_cache.pkl")
        
        cache_file = Path(cache_path)
        if cache_file.exists():
            cache_file.unlink()
            logger.info("Cache cleared", path=cache_path)
        else:
            logger.warning("Cache file not found", path=cache_path)

