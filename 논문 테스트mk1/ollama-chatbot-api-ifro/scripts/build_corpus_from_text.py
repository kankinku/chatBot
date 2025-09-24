import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from unifiedpdf.pdf_processor import PDFProcessor, PDFChunkConfig
from unifiedpdf.config import PipelineConfig


def load_extracted(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Extracted text file not found: {path}")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build JSONL corpus from cached extracted text (no PDF IO)")
    ap.add_argument("--input", default="data/extracted_text.jsonl", help="Path to extracted text JSONL")
    ap.add_argument("--out", default="data/corpus_from_text.jsonl", help="Path to output JSONL corpus")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument(
        "--strategy",
        choices=["processor", "sentence", "paragraph", "sliding", "fixed"],
        default="processor",
        help="Chunking strategy: PDFProcessor (default), sentence-level, paragraph-level, sliding window, or fixed (non-sliding)",
    )
    # Numeric-aware controls (PDFProcessor strategy only)
    ap.add_argument("--enable-numeric-window", action="store_true")
    ap.add_argument("--disable-numeric-window", action="store_true")
    ap.add_argument("--aux-per-doc", type=int, default=50)
    ap.add_argument("--aux-per-paragraph", type=int, default=1)
    ap.add_argument("--ocr-correct", choices=["off", "basic"], default="off")
    args = ap.parse_args()

    rows = load_extracted(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Optional OCR corrections at text level (A4 experiments)
    if args.ocr_correct != "off":
        texts = [r.get("text", "") for r in rows]
        if args.ocr_correct == "basic":
            from unifiedpdf.ocr_corrector import apply_basic_corrections
            texts = apply_basic_corrections(texts)
        for i, t in enumerate(texts):
            rows[i]["text"] = t

    lines: List[dict] = []

    if args.strategy == "processor":
        # Configure processor (map flags to config best-effort)
        cfg = PDFChunkConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if args.disable_numeric_window:
            cfg.enable_numeric_chunking = False
        elif args.enable_numeric_window:
            cfg.enable_numeric_chunking = True

        pipe_cfg = PipelineConfig()
        # 명시적 설정으로 기본값 오버라이드
        if args.disable_numeric_window:
            pipe_cfg.numeric_enable_dynamic_window = False
            pipe_cfg.numeric_aux_per_doc = 0
            pipe_cfg.numeric_aux_per_paragraph = 0
        elif args.enable_numeric_window:
            pipe_cfg.numeric_enable_dynamic_window = True
            pipe_cfg.numeric_aux_per_doc = args.aux_per_doc
            pipe_cfg.numeric_aux_per_paragraph = args.aux_per_paragraph
        proc = PDFProcessor(cfg, pipe_cfg)

        for row in rows:
            doc_id = row.get("doc_id") or row.get("filename") or "doc"
            filename = row.get("filename") or f"{doc_id}.txt"
            text = row.get("text", "")
            if not text:
                continue
            chunks = proc.chunk_text(doc_id=doc_id, filename=filename, text=text)
            for ch in chunks:
                extra = ch.extra.copy() if ch.extra else {}
                if args.disable_numeric_window or (args.enable_numeric_window and args.aux_per_doc == 0):
                    extra.pop("measurements", None)
                    extra.pop("wastewater_mode", None)
                lines.append({
                    "doc_id": ch.doc_id,
                    "filename": ch.filename,
                    "page": ch.page,
                    "start": ch.start_offset,
                    "length": ch.length,
                    "text": ch.text,
                    "extra": extra,
                })
    else:
        import re

        def _emit(doc_id: str, filename: str, start: int, text: str):
            lines.append({
                "doc_id": doc_id,
                "filename": filename,
                "page": None,
                "start": int(start),
                "length": int(len(text)),
                "text": text,
                "extra": {"strategy": args.strategy},
            })

        for row in rows:
            doc_id = row.get("doc_id") or row.get("filename") or "doc"
            filename = row.get("filename") or f"{doc_id}.txt"
            text = row.get("text", "") or ""
            if not text:
                continue
            if args.strategy == "sentence":
                parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
                cursor = 0
                for p in parts:
                    s = (p or "").strip()
                    if not s:
                        continue
                    pos = text.find(s, cursor)
                    if pos == -1:
                        pos = cursor
                    _emit(doc_id, filename, pos, s)
                    cursor = pos + len(s)
            elif args.strategy == "paragraph":
                paras = re.split(r"\n{2,}", text)
                cursor = 0
                for p in paras:
                    s = (p or "").strip()
                    if not s:
                        continue
                    pos = text.find(s, cursor)
                    if pos == -1:
                        pos = cursor
                    _emit(doc_id, filename, pos, s)
                    cursor = pos + len(s)
            elif args.strategy == "sliding":
                sz = max(1, int(args.chunk_size))
                ov = max(0, int(args.chunk_overlap))
                step = max(1, sz - ov)
                for start in range(0, len(text), step):
                    chunk = text[start:start + sz]
                    if not chunk:
                        break
                    _emit(doc_id, filename, start, chunk)
            elif args.strategy == "fixed":
                sz = max(1, int(args.chunk_size))
                # non-sliding: no overlap, fixed windows
                for start in range(0, len(text), sz):
                    chunk = text[start:start + sz]
                    if not chunk:
                        break
                    _emit(doc_id, filename, start, chunk)

    with out.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved corpus with {len(lines)} chunks to {out}")


if __name__ == "__main__":
    main()
