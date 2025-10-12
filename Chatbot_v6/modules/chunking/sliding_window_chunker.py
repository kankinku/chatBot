"""
Sliding Window Chunker - 슬라이딩 윈도우 청커

슬라이딩 윈도우 방식으로 텍스트를 청크로 나눕니다 (단일 책임).
"""

from __future__ import annotations

from typing import List

from .base_chunker import BaseChunker, ChunkingConfig
from modules.core.types import Chunk
from modules.core.exceptions import ChunkingError
from modules.core.logger import get_logger

logger = get_logger(__name__)


class SlidingWindowChunker(BaseChunker):
    """
    슬라이딩 윈도우 청커
    
    단일 책임: 슬라이딩 윈도우 방식으로만 청킹
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.chunk_size, self.chunk_overlap = config.get_effective_size_and_overlap()
        logger.debug("Effective chunking parameters", 
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap)
    
    def chunk_text(
        self,
        doc_id: str,
        filename: str,
        text: str,
        page: int | None = None,
    ) -> List[Chunk]:
        """
        슬라이딩 윈도우 방식으로 텍스트 청킹
        
        Args:
            doc_id: 문서 ID
            filename: 파일명
            text: 텍스트
            page: 페이지 번호
            
        Returns:
            청크 리스트
        """
        if not text:
            logger.warning("Empty text provided for chunking")
            return []
        
        # 너무 작은 문서는 한 덩어리로
        if len(text) <= self.chunk_size:
            logger.debug(f"Text too small ({len(text)} chars), creating single chunk")
            return [self._make_chunk(
                doc_id=doc_id,
                filename=filename,
                start=0,
                text=text,
                page=page,
                extra={"chunk_id": 0}
            )]
        
        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        
        chunk_id = 0
        i = 0
        
        while i < len(text):
            start = i
            end = min(len(text), i + self.chunk_size)
            
            # 문장 경계에 스냅 (±5% 윈도우)
            snap_margin = max(5, int(self.chunk_size * 0.05))
            
            # 뒤로 스냅: 공백/개행을 찾음
            if end < len(text):  # 마지막 청크가 아닐 때만
                j = end
                while j > start and (end - j) < snap_margin:
                    if text[j - 1] in {' ', '\n', '\t', '.', '。'}:
                        end = j
                        break
                    j -= 1
            
            # 청크 생성
            chunk_text = text[start:end]
            
            chunk = self._make_chunk(
                doc_id=doc_id,
                filename=filename,
                start=start,
                text=chunk_text,
                page=page,
                extra={
                    "chunk_id": chunk_id,
                    "method": "sliding_window",
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # 마지막 청크이면 종료
            if end >= len(text):
                break
            
            i = start + step
        
        logger.info(f"Created {len(chunks)} chunks",
                   doc_id=doc_id,
                   original_length=len(text))
        
        return chunks

