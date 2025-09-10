from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .types import Chunk
from .measurements import extract_measurements


@dataclass
class PDFChunkConfig:
    # 기본 슬라이딩 윈도우(문자 단위) 설정
    chunk_size: int = 800
    chunk_overlap: int = 200
    # 도메인 특화(예: 정수처리) 옵션
    enable_wastewater_chunking: bool = False
    wastewater_chunk_size: int = 900
    wastewater_overlap_ratio: float = 0.25  # chunk_size 대비 비율


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
        return chunks

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

