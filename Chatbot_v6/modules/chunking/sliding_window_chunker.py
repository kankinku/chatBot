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
            
            # 개선된 문장 경계 스냅
            if end < len(text) and self.config.enable_boundary_snap:  # 마지막 청크가 아닐 때만
                snap_margin = max(10, int(self.chunk_size * self.config.boundary_snap_margin_ratio))
                end = self._find_boundary_snap(text, start, end, snap_margin)
            
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
        
        # 이웃 정보 추가
        chunks = self._add_neighbor_hints(chunks)
        
        return chunks
    
    def _find_boundary_snap(
        self,
        text: str,
        start: int,
        end: int,
        margin: int,
    ) -> int:
        """
        개선된 경계 스냅 메커니즘
        
        문장 부호를 우선적으로 찾아 자연스러운 경계에서 분할합니다.
        
        Args:
            text: 전체 텍스트
            start: 시작 위치
            end: 현재 끝 위치
            margin: 탐색 범위 (±margin)
            
        Returns:
            조정된 끝 위치
        """
        search_start = max(start, end - margin)
        
        # 1순위: 문장 종결 부호
        for char in ['.', '。', '!', '?', ';']:
            pos = text.rfind(char, search_start, end)
            if pos != -1 and pos > start:
                # 종결 부호 다음 위치로 이동
                return pos + 1
        
        # 2순위: 절 구분자
        for char in [',', ':', '\n']:
            pos = text.rfind(char, search_start, end)
            if pos != -1 and pos > start:
                return pos + 1
        
        # 3순위: 공백 (마지막 수단)
        pos = text.rfind(' ', search_start, end)
        if pos != -1 and pos > start:
            return pos + 1
        
        # 적절한 경계를 찾지 못한 경우 원래 위치 유지
        return end
    
    def _add_neighbor_hints(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        청크에 이웃 정보 추가
        
        각 청크에 이전 청크의 위치 정보를 저장하여
        나중에 문맥 확장 시 활용할 수 있도록 합니다.
        
        Args:
            chunks: 청크 리스트
            
        Returns:
            이웃 정보가 추가된 청크 리스트
        """
        for i in range(len(chunks)):
            if i > 0:  # 이전 청크가 있으면
                prev = chunks[i - 1]
                chunks[i].neighbor_hint = (
                    prev.doc_id,
                    prev.page,
                    prev.start_offset
                )
        
        return chunks

