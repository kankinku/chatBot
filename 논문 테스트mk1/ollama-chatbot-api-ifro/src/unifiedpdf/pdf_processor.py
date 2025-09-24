from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from .types import Chunk
from .measurements import extract_measurements, extract_measure_spans
from .config import PipelineConfig


@dataclass
class PDFChunkConfig:
    # 기본 슬라이딩 윈도우(문자 단위)
    chunk_size: int = 800
    chunk_overlap: int = 200
    # 도메인 특화(유지하되 기본은 off)
    enable_wastewater_chunking: bool = False
    wastewater_chunk_size: int = 900
    wastewater_overlap_ratio: float = 0.25
    # 숫자 앵커 기반 추가 청크
    enable_numeric_chunking: bool = True


class PDFProcessor:
    def __init__(self, cfg: Optional[PDFChunkConfig] = None, pipe_cfg: Optional[PipelineConfig] = None) -> None:
        self.cfg = cfg or PDFChunkConfig()
        self.pipe_cfg = pipe_cfg or PipelineConfig()

    def _effective_params(self) -> tuple[int, int]:
        size = self.cfg.chunk_size
        ov = self.cfg.chunk_overlap
        if self.cfg.enable_wastewater_chunking:
            size = self.cfg.wastewater_chunk_size
            ov = max(1, int(size * self.cfg.wastewater_overlap_ratio))
        ov = min(max(0, ov), max(0, size - 1))
        return size, ov

    def chunk_text(self, doc_id: str, filename: str, text: str) -> List[Chunk]:
        size, overlap = self._effective_params()
        if not text:
            return []
        if len(text) <= size:
            base = [self._make_chunk(doc_id, filename, 0, text, 0)]
        else:
            base: List[Chunk] = []
            step = max(1, size - overlap)
            i = 0
            cid = 0
            while i < len(text):
                start = i
                end = min(len(text), i + size)
                # 경계 스냅: 문장/공백 경계로 근접 조정 (±5%)
                snap_margin = max(5, int(size * 0.05))
                # 뒤로 스냅
                j = end
                while j > start and j - end < snap_margin and text[j - 1 : j] not in {" ", "\n", "\t"}:
                    j -= 1
                if j > start and end - j <= snap_margin:
                    end = j
                chunk_text = text[start:end]
                base.append(self._make_chunk(doc_id, filename, start, chunk_text, cid))
                cid += 1
                if end >= len(text):
                    break
                i = start + step

        chunks = list(base)

        # 숫자 앵커 기반 추가 청크
        if self.cfg.enable_numeric_chunking or self.pipe_cfg.numeric_enable_dynamic_window:
            chunks.extend(self._create_numeric_anchor_chunks(doc_id, filename, text, base))

        return chunks

    def _create_numeric_anchor_chunks(self, doc_id: str, filename: str, text: str, base_chunks: List[Chunk]) -> List[Chunk]:
        spans = extract_measure_spans(text)
        if not spans:
            return []

        # 단위별 동적 하프 윈도우 (min,max) → 평균 사용
        dyn_window = {
            "%": (120, 160),
            "ph": (120, 160),
            "°c": (150, 220),
            "ntu": (150, 220),
            "mg/l": (180, 240),
            "ppm": (180, 240),
            "µs/cm": (180, 240),
            "bar": (180, 260),
            "mpa": (180, 260),
            "l/s": (180, 260),
            "m³/h": (180, 260),
            "m3/h": (180, 260),
            "m³/d": (180, 260),
            "m3/d": (180, 260),
            "kwh": (180, 260),
        }

        def halfwin(u: str) -> int:
            u = (u or "").lower()
            if u in dyn_window:
                lo, hi = dyn_window[u]
                return (lo + hi) // 2
            lo, hi = (150, 200)
            return (lo + hi) // 2

        intervals: List[Tuple[int, int]] = []
        for sp in spans:
            hw = halfwin(sp.unit)
            s = max(0, sp.start - hw)
            e = min(len(text), sp.end + hw)
            # 문장/공백 스냅 (±5%)
            margin = max(5, int((e - s) * 0.05))
            # 왼쪽 스냅
            i = s
            while i > 0 and s - i < margin and text[i] not in {" ", "\n", "\t", ".", "?", "!"}:
                i -= 1
            if s - i < margin:
                s = i
            # 오른쪽 스냅
            j = e
            while j < len(text) and j - e < margin and text[j - 1] not in {" ", "\n", "\t", ".", "?", "!"}:
                j += 1
            if j - e < margin:
                e = j
            intervals.append((s, e))

        # IoU≥thr 병합
        def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
            union = (a[1] - a[0]) + (b[1] - b[0]) - inter
            return inter / union if union > 0 else 0.0

        intervals.sort()
        merged: List[Tuple[int, int]] = []
        for iv in intervals:
            if not merged:
                merged.append(iv)
                continue
            last = merged[-1]
            thr = float(getattr(self.pipe_cfg, "numeric_iou_threshold", 0.3))
            if iou(last, iv) >= thr or iv[0] <= last[1]:
                merged[-1] = (min(last[0], iv[0]), max(last[1], iv[1]))
            else:
                merged.append(iv)

        # 길이 제한 및 메타 생성
        MAXLEN = int(getattr(self.pipe_cfg, "numeric_max_anchor_len", 1000))
        anchored: List[Chunk] = []
        for (s, e) in merged:
            if e - s > MAXLEN:
                e = s + MAXLEN
            txt = text[s:e].strip()
            if not txt:
                continue
            inside = [sp for sp in spans if sp.start >= s and sp.end <= e]
            measures = []
            for sp in inside:
                measures.append((sp.value if sp.value is not None else sp.range_, sp.unit, sp.start - s, sp.end - s, sp.label_hint))
            lines = txt.splitlines()
            short_lines = sum(1 for ln in lines if len(ln.strip()) <= 40)
            sep_ratio = sum(1 for ln in lines if any(ch in ln for ch in [",", ";", ":", "|"])) / max(1, len(lines))
            table_like = (short_lines >= max(3, len(lines) // 3)) or (sep_ratio >= 0.3)
            anchor_density = len(inside) / max(1.0, len(txt))
            anchored.append(Chunk(
                doc_id=doc_id,
                filename=filename,
                page=None,
                start_offset=s,
                length=len(txt),
                text=txt,
                extra={
                    "numeric_anchor": True,
                    "measures": measures,
                    "table_like": table_like,
                    "anchor_density": anchor_density,
                }
            ))

        # Aux numeric chunks (정책: M=1/문단, K/문서, 길이 80~200, 권장 120~180, dedup)
        AUX_PER_DOC = int(getattr(self.pipe_cfg, "numeric_aux_per_doc", 50))
        AUX_PER_PAR = int(getattr(self.pipe_cfg, "numeric_aux_per_paragraph", 1))

        # paragraph id lookup from base_chunks
        par_ranges: List[Tuple[int, int, int]] = []  # (start, end, paragraph_id)
        for bc in base_chunks:
            par_ranges.append((bc.start_offset, bc.start_offset + bc.length, int(bc.extra.get("paragraph_id", 0))))
        par_ranges.sort()

        def paragraph_id_for_offset(pos: int) -> int:
            for s, e, pid in par_ranges:
                if s <= pos < e:
                    return pid
            # fallback: nearest preceding
            prev = [pid for s, e, pid in par_ranges if s <= pos]
            return prev[-1] if prev else 0

        def char_ngrams(s: str, n: int = 5) -> set:
            t = s.replace("\n", " ")
            return {t[i:i+n] for i in range(0, max(0, len(t) - n + 1))}

        def jaccard(a: set, b: set) -> float:
            if not a and not b:
                return 1.0
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union else 0.0

        aux: List[Chunk] = []
        par_count: Dict[int, int] = {}
        seen_keys: set = set()
        ngram_sets: List[set] = []

        for ch in anchored:
            if len(aux) >= AUX_PER_DOC:
                break
            pid = paragraph_id_for_offset(ch.start_offset)
            if par_count.get(pid, 0) >= AUX_PER_PAR:
                continue
            # choose snippet around first measure
            inside = ch.extra.get("measures", [])
            if inside:
                rel_start = inside[0][2]  # start offset relative to chunk
            else:
                rel_start = 0
            base_txt = ch.text
            begin = max(0, rel_start - 90)
            end = min(len(base_txt), begin + 180)
            snip = base_txt[begin:end].strip()
            if len(snip) < 80 or len(snip) > 200:
                continue
            # dedup by (value, unit, label)
            v = inside[0][0] if inside else None
            u = inside[0][1] if inside else None
            lab = inside[0][4] if inside else None
            key = (str(v), str(u), str(lab))
            if key in seen_keys:
                continue
            # n-gram jaccard dedup
            grams = char_ngrams(snip, 5)
            if any(jaccard(grams, g) >= 0.8 for g in ngram_sets):
                continue

            aux.append(Chunk(
                doc_id=doc_id,
                filename=filename,
                page=None,
                start_offset=ch.start_offset + begin,
                length=len(snip),
                text=snip,
                extra={**ch.extra, "aux_numeric": True}
            ))
            par_count[pid] = par_count.get(pid, 0) + 1
            seen_keys.add(key)
            ngram_sets.append(grams)

        return anchored + aux

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
