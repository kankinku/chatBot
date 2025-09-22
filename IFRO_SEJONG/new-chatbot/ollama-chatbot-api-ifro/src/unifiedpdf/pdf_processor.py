from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .types import Chunk
from .measurements import extract_measurements


@dataclass
class PDFChunkConfig:
    # 기본 슬라이딩 윈도우(문자 단위) 설정
    chunk_size: int = 802
    chunk_overlap: int = 200
    # 도메인 특화(예: 정수처리) 옵션
    enable_wastewater_chunking: bool = False
    wastewater_chunk_size: int = 900
    wastewater_overlap_ratio: float = 0.25  # chunk_size 대비 비율
    # 숫자 중심 청킹 강화 옵션
    enable_numeric_chunking: bool = True
    numeric_context_window: int = 3  # 숫자 주변 문맥 확장
    preserve_table_context: bool = True  # 표 주변 텍스트 보존


class PDFProcessor:
    def __init__(self, cfg: Optional[PDFChunkConfig] = None) -> None:
        self.cfg = cfg or PDFChunkConfig()

    def _effective_params(self) -> tuple[int, int]:
        size = self.cfg.chunk_size
        ov = self.cfg.chunk_overlap
        if self.cfg.enable_wastewater_chunking:
            size = self.cfg.wastewater_chunk_size
            ov = max(1, int(size * self.cfg.wastewater_overlap_ratio))
        # step은 size - overlap
        ov = min(max(0, ov), max(0, size - 1))
        return size, ov

    def chunk_text(self, doc_id: str, filename: str, text: str) -> List[Chunk]:
        size, overlap = self._effective_params()
        if not text:
            return []
        # 경계 보정: 너무 작은 문서는 한 덩어리로 처리
        if len(text) <= size:
            return [self._make_chunk(doc_id, filename, 0, text, 0)]

        chunks: List[Chunk] = []
        
        # 기본 슬라이딩 윈도우 청킹
        step = max(1, size - overlap)
        i = 0
        cid = 0
        while i < len(text):
            start = i
            end = min(len(text), i + size)
            # 경계 스냅: 문장/공백 경계에 약간 스냅 (±5% 윈도우)
            snap_margin = max(5, int(size * 0.05))
            # 뒤로 스냅
            j = end
            while j > start and j - end < snap_margin and text[j - 1 : j] not in {" ", "\n", "\t"}:
                j -= 1
            if j > start and end - j <= snap_margin:
                end = j
            # 청크 구성
            chunk_text = text[start:end]
            chunks.append(self._make_chunk(doc_id, filename, start, chunk_text, cid))
            cid += 1
            if end >= len(text):
                break
            i = start + step
        
        # 숫자 중심 확장 청킹 추가
        if self.cfg.enable_numeric_chunking:
            numeric_chunks = self._create_numeric_chunks(doc_id, filename, text, chunks)
            chunks.extend(numeric_chunks)
        
        return chunks

    def _create_numeric_chunks(self, doc_id: str, filename: str, text: str, existing_chunks: List[Chunk]) -> List[Chunk]:
        """숫자가 포함된 청크에 대해 문맥을 확장한 추가 청크 생성"""
        import re
        numeric_chunks = []
        
        # 숫자 패턴 찾기 (수치, 날짜, 백분율, 상관계수 등)
        numeric_patterns = [
            r'\d+(?:\.\d+)?\s*(?:mg/L|ppm|ppb|NTU|pH|℃|°C|kWh|RPM|bar|MPa|%)',  # 단위가 있는 수치
            r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',  # 날짜
            r'(?:상관계수|R²|R2)\s*[=:]\s*\d+(?:\.\d+)?',  # 상관계수
            r'\d+(?:\.\d+)?\s*%',  # 백분율
        ]
        
        for chunk in existing_chunks:
            chunk_text = chunk.text
            measurements = extract_measurements(chunk_text)
            
            # 숫자가 포함된 청크만 처리
            if not measurements:
                continue
            
            # 문맥 확장을 위한 이웃 청크 찾기
            context_chunks = self._find_context_chunks(chunk, existing_chunks)
            
            # 확장된 텍스트 생성
            expanded_text = self._build_expanded_text(chunk_text, context_chunks, measurements)
            
            if expanded_text != chunk_text:  # 확장된 경우만 추가
                numeric_chunk = Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    page=chunk.page,
                    start_offset=chunk.start_offset,
                    length=len(expanded_text),
                    text=expanded_text,
                    extra={
                        **chunk.extra,
                        "numeric_expanded": True,
                        "original_length": len(chunk_text),
                        "context_chunks": len(context_chunks)
                    }
                )
                numeric_chunks.append(numeric_chunk)
        
        return numeric_chunks

    def _find_context_chunks(self, target_chunk: Chunk, all_chunks: List[Chunk]) -> List[Chunk]:
        """대상 청크 주변의 문맥 청크들을 찾기"""
        context_chunks = []
        target_start = target_chunk.start_offset
        
        for chunk in all_chunks:
            if chunk.start_offset == target_start:  # 자기 자신은 제외
                continue
            
            # 설정된 윈도우 범위 내의 청크들 찾기
            distance = abs(chunk.start_offset - target_start)
            if distance <= self.cfg.numeric_context_window * 500:  # 대략적인 거리 계산
                context_chunks.append(chunk)
        
        # 거리순으로 정렬하여 가까운 것부터 선택
        context_chunks.sort(key=lambda c: abs(c.start_offset - target_start))
        return context_chunks[:self.cfg.numeric_context_window]

    def _build_expanded_text(self, original_text: str, context_chunks: List[Chunk], measurements: List[Tuple[str, str]]) -> str:
        """문맥 청크들을 활용하여 확장된 텍스트 생성"""
        expanded_parts = []
        
        # 앞쪽 문맥 추가
        for chunk in context_chunks[:self.cfg.numeric_context_window//2]:
            if chunk.start_offset < len(original_text.split('\n')[0]):  # 첫 줄보다 앞에 있는 경우
                expanded_parts.append(chunk.text[-200:])  # 뒤쪽 200자만
        
        # 원본 텍스트
        expanded_parts.append(original_text)
        
        # 뒤쪽 문맥 추가
        for chunk in context_chunks[self.cfg.numeric_context_window//2:]:
            if chunk.start_offset > len(original_text):  # 원본보다 뒤에 있는 경우
                expanded_parts.append(chunk.text[:200])  # 앞쪽 200자만
        
        # 숫자 강조를 위한 반복 추가
        expanded_text = '\n'.join(expanded_parts)
        if measurements:
            # 측정값들을 텍스트 끝에 반복 추가하여 검색 시 강조
            numeric_values = [num for num, unit in measurements]
            emphasis_text = ' ' + ' '.join(numeric_values) * 2  # 2번 반복
            expanded_text += emphasis_text
        
        return expanded_text

    def _make_chunk(self, doc_id: str, filename: str, start: int, text: str, paragraph_id: int) -> Chunk:
        measures = extract_measurements(text)
        extra = {
            "section": None,
            "paragraph_id": paragraph_id,
            "measurements": measures,
            "wastewater_mode": self.cfg.enable_wastewater_chunking,
        }
        return Chunk(
            doc_id=doc_id,
            filename=filename,
            page=None,
            start_offset=start,
            length=len(text),
            text=text.strip(),
            extra=extra,
        )

