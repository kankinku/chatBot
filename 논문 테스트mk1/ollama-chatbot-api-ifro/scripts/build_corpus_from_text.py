import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from unifiedpdf.types import Chunk
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
    # Numeric-aware controls (kept minimal; rely on PDFProcessor implementation)
    ap.add_argument("--enable-numeric-window", action="store_true")
    ap.add_argument("--disable-numeric-window", action="store_true")
    ap.add_argument("--aux-per-doc", type=int, default=50)
    ap.add_argument("--aux-per-paragraph", type=int, default=1)
    ap.add_argument("--ocr-correct", choices=["off", "basic"], default="off")
    args = ap.parse_args()

    rows = load_extracted(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Configure processor (map flags to config best-effort)
    cfg = PDFChunkConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    # For now, PDFChunkConfig has enable_numeric_chunking boolean only.
    # Map enable/disable flags to that field; aux behavior handled internally by PDFProcessor if present.
    if args.disable_numeric_window:
        cfg.enable_numeric_chunking = False
    elif args.enable_numeric_window:
        cfg.enable_numeric_chunking = True

    pipe_cfg = PipelineConfig()
    # 숫자 인식 동적 윈도우 설정
    if args.disable_numeric_window:
        pipe_cfg.numeric_enable_dynamic_window = False
    elif args.enable_numeric_window:
        pipe_cfg.numeric_enable_dynamic_window = True
    # 기본값은 PipelineConfig의 기본값 유지
    
    pipe_cfg.numeric_aux_per_doc = args.aux_per_doc
    pipe_cfg.numeric_aux_per_paragraph = args.aux_per_paragraph
    proc = PDFProcessor(cfg, pipe_cfg)

    lines = []
    # Optional OCR corrections at text level (A4 experiments)
    if args.ocr_correct != "off":
        texts = [r.get("text", "") for r in rows]
        if args.ocr_correct == "basic":
            from unifiedpdf.ocr_corrector import apply_basic_corrections
            texts = apply_basic_corrections(texts)
        for i, t in enumerate(texts):
            rows[i]["text"] = t

    for row in rows:
        doc_id = row.get("doc_id") or row.get("filename") or "doc"
        filename = row.get("filename") or f"{doc_id}.txt"
        text = row.get("text", "")
        if not text:
            continue
        chunks = proc.chunk_text(doc_id=doc_id, filename=filename, text=text)
        for ch in chunks:
            extra = ch.extra.copy() if ch.extra else {}
            # 베이스라인(A1)과 보조 청크 없는 A2에서는 measurements 필드를 제거
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

    with out.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved corpus with {len(lines)} chunks to {out}")


if __name__ == "__main__":
    main()
