"""
Numeric Chunker - 숫자 중심 청커

숫자가 포함된 청크에 대해 문맥을 확장한 추가 청크를 생성합니다 (단일 책임).
"""

from __future__ import annotations

from typing import List

from .base_chunker import BaseChunker, ChunkingConfig
from .sliding_window_chunker import SlidingWindowChunker
from modules.core.types import Chunk
from modules.preprocessing.normalizer import MeasurementNormalizer
from modules.core.logger import get_logger

logger = get_logger(__name__)


class NumericChunker(BaseChunker):
    """
    숫자 중심 청커
    
    단일 책임: 숫자가 포함된 청크의 문맥 확장만 수행
    """
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.base_chunker = SlidingWindowChunker(config)
        self.measurement_normalizer = MeasurementNormalizer()
    
    def chunk_text(
        self,
        doc_id: str,
        filename: str,
        text: str,
        page: int | None = None,
    ) -> List[Chunk]:
        """
        슬라이딩 윈도우 + 숫자 중심 확장 청킹
        
        Args:
            doc_id: 문서 ID
            filename: 파일명
            text: 텍스트
            page: 페이지 번호
            
        Returns:
            청크 리스트 (기본 청크 + 숫자 확장 청크)
        """
        # 기본 슬라이딩 윈도우 청킹
        base_chunks = self.base_chunker.chunk_text(doc_id, filename, text, page)
        
        if not self.config.enable_numeric_chunking:
            return base_chunks
        
        # 숫자 중심 확장 청크 생성
        numeric_chunks = self._create_numeric_expanded_chunks(
            doc_id,
            filename,
            text,
            base_chunks,
            page,
        )
        
        logger.info(f"Created {len(numeric_chunks)} numeric expanded chunks",
                   doc_id=doc_id,
                   base_chunks=len(base_chunks),
                   numeric_chunks=len(numeric_chunks))
        
        return base_chunks + numeric_chunks
    
    def _create_numeric_expanded_chunks(
        self,
        doc_id: str,
        filename: str,
        full_text: str,
        base_chunks: List[Chunk],
        page: int | None,
    ) -> List[Chunk]:
        """
        숫자가 포함된 청크에 대해 문맥 확장 청크 생성
        
        Args:
            doc_id: 문서 ID
            filename: 파일명
            full_text: 전체 텍스트
            base_chunks: 기본 청크 리스트
            page: 페이지 번호
            
        Returns:
            숫자 확장 청크 리스트
        """
        numeric_chunks = []
        
        for chunk in base_chunks:
            # 측정값 추출
            measurements = self.measurement_normalizer.extract_measurements(chunk.text)
            
            if not measurements:
                continue  # 숫자가 없으면 스킵
            
            # 이웃 청크 찾기
            neighbors = self._find_neighbor_chunks(chunk, base_chunks)
            
            if not neighbors:
                continue  # 이웃이 없으면 스킵
            
            # 확장된 텍스트 생성
            expanded_text = self._build_expanded_text(
                chunk.text,
                neighbors,
                measurements
            )
            
            # 원본과 다르면 추가
            if expanded_text != chunk.text:
                expanded_chunk = self._make_chunk(
                    doc_id=doc_id,
                    filename=filename,
                    start=chunk.start_offset,
                    text=expanded_text,
                    page=page,
                    extra={
                        "chunk_id": chunk.extra.get("chunk_id", 0),
                        "method": "numeric_expanded",
                        "numeric_expanded": True,
                        "original_length": len(chunk.text),
                        "context_chunks": len(neighbors),
                        "measurements_count": len(measurements),
                    }
                )
                numeric_chunks.append(expanded_chunk)
        
        return numeric_chunks
    
    def _find_neighbor_chunks(
        self,
        target: Chunk,
        all_chunks: List[Chunk],
    ) -> List[Chunk]:
        """
        대상 청크 주변의 이웃 청크 찾기
        
        Args:
            target: 대상 청크
            all_chunks: 전체 청크 리스트
            
        Returns:
            이웃 청크 리스트
        """
        target_start = target.start_offset
        window = self.config.numeric_context_window
        
        neighbors = []
        
        for chunk in all_chunks:
            if chunk.start_offset == target_start:
                continue  # 자기 자신 제외
            
            # 거리 계산 (대략적)
            distance = abs(chunk.start_offset - target_start)
            
            # 윈도우 범위 내에 있으면 추가
            if distance <= window * 500:  # 500자 기준
                neighbors.append((distance, chunk))
        
        # 거리순 정렬 후 상위 N개 선택
        neighbors.sort(key=lambda x: x[0])
        return [chunk for _, chunk in neighbors[:window]]
    
    def _build_expanded_text(
        self,
        original_text: str,
        neighbors: List[Chunk],
        measurements: list,
    ) -> str:
        """
        이웃 청크를 활용하여 확장된 텍스트 생성
        
        Args:
            original_text: 원본 텍스트
            neighbors: 이웃 청크 리스트
            measurements: 측정값 리스트
            
        Returns:
            확장된 텍스트
        """
        parts = []
        
        # 앞쪽 이웃 (원본보다 앞에 있는 것)
        window_half = self.config.numeric_context_window // 2
        
        for neighbor in neighbors[:window_half]:
            if neighbor.start_offset < len(original_text.split('\n')[0]):
                # 뒤쪽 200자만 추가
                parts.append(neighbor.text[-200:])
        
        # 원본 텍스트
        parts.append(original_text)
        
        # 뒤쪽 이웃 (원본보다 뒤에 있는 것)
        for neighbor in neighbors[window_half:]:
            if neighbor.start_offset > len(original_text):
                # 앞쪽 200자만 추가
                parts.append(neighbor.text[:200])
        
        # 결합
        expanded = '\n'.join(parts)
        
        # 숫자 강조 (검색 시 가중치 증가)
        if measurements and self.config.preserve_table_context:
            numeric_values = [num for num, unit in measurements]
            # 측정값을 끝에 반복 추가 (2번)
            emphasis = ' ' + ' '.join(numeric_values) * 2
            expanded += emphasis
        
        return expanded

