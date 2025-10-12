"""
Base Chunker - 청킹 기본 클래스

모든 청커의 베이스 클래스입니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from config.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from modules.core.types import Chunk
from modules.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkingConfig:
    """청킹 설정"""
    
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    
    # 정수장 특화 설정
    enable_wastewater_mode: bool = False
    wastewater_chunk_size: int = 900
    wastewater_overlap_ratio: float = 0.25
    
    # 숫자 중심 청킹
    enable_numeric_chunking: bool = True
    numeric_context_window: int = 3
    preserve_table_context: bool = True
    
    def get_effective_size_and_overlap(self) -> tuple[int, int]:
        """실제 사용할 chunk_size와 overlap 반환"""
        if self.enable_wastewater_mode:
            size = self.wastewater_chunk_size
            overlap = max(1, int(size * self.wastewater_overlap_ratio))
        else:
            size = self.chunk_size
            overlap = self.chunk_overlap
        
        # overlap이 size보다 크면 조정
        overlap = min(max(0, overlap), max(0, size - 1))
        
        return size, overlap


class BaseChunker(ABC):
    """
    청커 베이스 클래스
    
    모든 청커는 이 클래스를 상속받아 chunk_text 메서드를 구현합니다.
    """
    
    def __init__(self, config: ChunkingConfig):
        """
        Args:
            config: 청킹 설정
        """
        self.config = config
        logger.info(f"{self.__class__.__name__} initialized", config=config.__dict__)
    
    @abstractmethod
    def chunk_text(
        self,
        doc_id: str,
        filename: str,
        text: str,
        page: int | None = None,
    ) -> List[Chunk]:
        """
        텍스트를 청크로 나누기
        
        Args:
            doc_id: 문서 ID
            filename: 파일명
            text: 텍스트
            page: 페이지 번호 (옵션)
            
        Returns:
            청크 리스트
        """
        pass
    
    def _make_chunk(
        self,
        doc_id: str,
        filename: str,
        start: int,
        text: str,
        page: int | None = None,
        extra: dict | None = None,
    ) -> Chunk:
        """
        청크 객체 생성
        
        Args:
            doc_id: 문서 ID
            filename: 파일명
            start: 시작 오프셋
            text: 텍스트
            page: 페이지 번호
            extra: 추가 메타데이터
            
        Returns:
            Chunk 객체
        """
        return Chunk(
            doc_id=doc_id,
            filename=filename,
            page=page,
            start_offset=start,
            length=len(text),
            text=text.strip(),
            extra=extra or {},
        )

