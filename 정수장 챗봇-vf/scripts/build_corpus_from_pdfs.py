import argparse
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from unifiedpdf.types import Chunk
from unifiedpdf.utils import char_ngrams, jaccard


def deduplicate_chunks(chunks: list[Chunk], threshold: float = 0.9, min_length: int = 50) -> list[Chunk]:
    """코퍼스 구축 시 청크 중복 제거"""
    if not chunks:
        return chunks
    
    # 길이가 너무 짧은 청크는 중복 제거 대상에서 제외
    filtered_chunks = [c for c in chunks if len(c.text) >= min_length]
    
    if len(filtered_chunks) <= 1:
        return chunks
    
    # 중복 제거 로직
    seen_hashes = set()
    unique_chunks = []
    duplicate_count = 0
    
    for chunk in chunks:
        if len(chunk.text) < min_length:
            unique_chunks.append(chunk)
            continue
            
        # 텍스트 해시로 정확한 중복 확인
        text_hash = hash(chunk.text)
        if text_hash in seen_hashes:
            duplicate_count += 1
            continue
            
        # Jaccard 유사도로 근사 중복 확인
        chunk_ngrams = char_ngrams(chunk.text)
        is_duplicate = False
        
        for existing in unique_chunks:
            if len(existing.text) < min_length:
                continue
            existing_ngrams = char_ngrams(existing.text)
            similarity = jaccard(chunk_ngrams, existing_ngrams)
            
            if similarity >= threshold:
                is_duplicate = True
                duplicate_count += 1
                break
        
        if not is_duplicate:
            seen_hashes.add(text_hash)
            unique_chunks.append(chunk)
    
    print(f"중복 제거 완료: {duplicate_count}개 중복 청크 제거, {len(unique_chunks)}개 유니크 청크 유지")
    return unique_chunks


def extract_text_pdfplumber(pdf_path: Path) -> str:
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return ""
    text = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def extract_text_fitz(pdf_path: Path) -> str:
    """Extract text using PyMuPDF (fitz). Better for 디지털 PDF."""
    try:
        import fitz  # type: ignore
    except Exception:
        return ""
    text = []
    try:
        doc = fitz.open(str(pdf_path))
        for page in doc:
            text.append(page.get_text("text"))
        doc.close()
    except Exception:
        return ""
    return "\n".join(text)


def extract_text_pymupdf4llm(pdf_path: Path) -> str:
    """Extract text using pymupdf4llm. Better for complex layouts and LLM-optimized output."""
    try:
        import pymupdf4llm  # type: ignore
    except Exception:
        return ""
    try:
        # Convert PDF to markdown format optimized for LLM processing
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        # Convert markdown to plain text while preserving structure
        import re
        # Remove markdown formatting but keep line breaks
        text = re.sub(r'#{1,6}\s*', '', md_text)  # Remove headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # Remove code
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove links
        return text
    except Exception:
        return ""


def extract_text_auto(pdf_path: Path) -> str:
    # Try pymupdf4llm first (best for complex layouts), then pdfplumber, then fitz
    t = extract_text_pymupdf4llm(pdf_path)
    if t and t.strip():
        return t
    t = extract_text_pdfplumber(pdf_path)
    if t and t.strip():
        return t
    t = extract_text_fitz(pdf_path)
    return t


def main():
    ap = argparse.ArgumentParser(description="Build JSONL corpus from PDFs (best-effort)")
    ap.add_argument("--pdf_dir", default="./pdfs")
    ap.add_argument("--out", default="data/corpus_v1.jsonl")
    ap.add_argument("--pdf-extractor", default="auto", choices=["auto", "plumber", "fitz", "pymupdf4llm"], help="PDF text extractor")
    ap.add_argument("--ocr", default="auto", choices=["auto", "always", "off"], help="OCR fallback policy")
    ap.add_argument("--ocr-lang", default="kor+eng", help="OCR languages (tesseract/ocrmypdf)")
    ap.add_argument("--ocr-dpi", type=int, default=200, help="OCR rasterization DPI (via fitz)")
    ap.add_argument("--chunking", default="paragraph", choices=["paragraph", "page", "window"], help="Chunking strategy for extracted text")
    ap.add_argument("--chunk-size", type=int, default=800, help="Sliding window chunk size (window mode)")
    ap.add_argument("--chunk-overlap", type=int, default=200, help="Sliding window overlap (window mode)")
    ap.add_argument("--wt-enable", action="store_true", help="Enable domain-specific wastewater chunking presets")
    ap.add_argument("--wt-size", type=int, default=900, help="Wastewater chunk size")
    ap.add_argument("--wt-overlap-ratio", type=float, default=0.25, help="Wastewater overlap ratio (0-1)")
    # LLM 기반 후교정(모델 비종속) — 새 이름
    ap.add_argument("--llm-correct", action="store_true", help="Use local LLM to post-correct low-quality spans (model-agnostic)")
    ap.add_argument("--llm-correct-threshold", type=float, default=0.5, help="Noise threshold [0,1] to select spans for correction")
    ap.add_argument("--llm-correct-max-chars", type=int, default=10000, help="Max total characters to correct per document")
    ap.add_argument("--llm-correct-batch", type=int, default=4, help="LLM correction batch size")
    ap.add_argument("--llm-dict-file", default=None, help="Optional domain dictionary file (UTF-8 text)")
    # 중복 제거 옵션
    ap.add_argument("--dedup", action="store_true", help="Enable chunk deduplication during corpus building")
    ap.add_argument("--dedup-threshold", type=float, default=0.9, help="Jaccard similarity threshold for deduplication [0,1]")
    ap.add_argument("--dedup-min-length", type=int, default=50, help="Minimum chunk length to consider for deduplication")
    # 구버전 플래그(호환): --ocr-lite*
    ap.add_argument("--ocr-lite", action="store_true", help="[deprecated] alias of --llm-correct")
    ap.add_argument("--ocr-lite-threshold", type=float, default=0.6, help="[deprecated] alias of --llm-correct-threshold")
    ap.add_argument("--ocr-lite-max-chars", type=int, default=10000, help="[deprecated] alias of --llm-correct-max-chars")
    ap.add_argument("--ocr-lite-batch", type=int, default=4, help="[deprecated] alias of --llm-correct-batch")
    # 숫자 중심 확장 청크 옵션
    ap.add_argument("--numeric-expand", action="store_true", help="Create numeric-focused expanded chunks with neighbor context")
    ap.add_argument("--numeric-window", type=int, default=1, help="Neighbor lines to include on each side for numeric expansion")
    ap.add_argument("--numeric-emphasis", type=int, default=3, help="Repeat factor for numeric tokens to emphasize in retrieval")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def has_meaningful_text(t: str) -> bool:
        if not t:
            return False
        s = t.strip()
        if len(s) >= 200:
            return True
        # At least 50 non-space chars
        if sum(1 for c in s if not c.isspace()) >= 50:
            return True
        # Presence of Hangul or letters + numbers
        import re
        if re.search(r"[가-힣A-Za-z0-9]", s):
            return len(s) >= 80
        return False

    def _ensure_tesseract_cmd() -> None:
        try:
            import pytesseract  # type: ignore
        except Exception:
            return
        # If already configured, skip
        if getattr(pytesseract.pytesseract, "tesseract_cmd", None):
            return
        # ENV override
        tcmd = os.environ.get("TESSERACT_CMD")
        if tcmd and Path(tcmd).exists():
            pytesseract.pytesseract.tesseract_cmd = tcmd
            return
        # Try common Windows install paths
        candidates = [
            Path(os.environ.get("PROGRAMFILES", r"C:\\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", r"C:\\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
            Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        ]
        for c in candidates:
            try:
                if c.exists():
                    pytesseract.pytesseract.tesseract_cmd = str(c)
                    break
            except Exception:
                continue

    def ocr_with_tesseract_via_fitz(pdf: Path, dpi: int, lang: str) -> str:
        try:
            import fitz  # type: ignore
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
        except Exception:
            return ""
        _ensure_tesseract_cmd()
        text_chunks = []
        try:
            doc = fitz.open(str(pdf))
            for page in doc:
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pm.tobytes("png")))
                try:
                    txt = pytesseract.image_to_string(img, lang=lang)
                except Exception:
                    # fallback to English only if language not available
                    try:
                        txt = pytesseract.image_to_string(img, lang="eng")
                    except Exception:
                        txt = pytesseract.image_to_string(img)
                if txt:
                    text_chunks.append(txt)
            doc.close()
        except Exception:
            return ""
        return "\n\n".join(text_chunks)

    def ocr_with_ocrmypdf(pdf: Path, lang: str) -> str:
        # Requires ocrmypdf CLI
        exe = shutil.which("ocrmypdf")
        if not exe:
            return ""
        with tempfile.TemporaryDirectory() as td:
            out_pdf = Path(td) / "ocr.pdf"
            cmd = [exe, "--quiet", "--skip-text", "-l", lang, str(pdf), str(out_pdf)]
            try:
                import subprocess
                r = subprocess.run(cmd, check=False)
                if r.returncode != 0 or not out_pdf.exists():
                    return ""
                # Re-extract with fitz
                return extract_text_fitz(out_pdf)
            except Exception:
                return ""

    lines = []
    for pdf in pdf_dir.glob("**/*.pdf"):
        # 1) text extractor
        if args.pdf_extractor == "plumber":
            text = extract_text_pdfplumber(pdf)
        elif args.pdf_extractor == "fitz":
            text = extract_text_fitz(pdf)
        elif args.pdf_extractor == "pymupdf4llm":
            text = extract_text_pymupdf4llm(pdf)
        else:
            text = extract_text_auto(pdf)

        # 2) OCR policy
        need_ocr = (args.ocr == "always") or (args.ocr == "auto" and not has_meaningful_text(text))
        if need_ocr:
            txt = ocr_with_tesseract_via_fitz(pdf, dpi=args.ocr_dpi, lang=args.ocr_lang)
            if not has_meaningful_text(txt):
                txt = ocr_with_ocrmypdf(pdf, lang=args.ocr_lang)
            if has_meaningful_text(txt):
                text = txt

        if not has_meaningful_text(text):
            continue

        # 3) Chunking
        if args.chunking == "page":
            # Page-based chunking using fitz for page text map (best-effort)
            try:
                import fitz  # type: ignore
                doc = fitz.open(str(pdf))
                for idx, page in enumerate(doc, start=1):
                    txt = page.get_text("text") if text else ""
                    if not txt:
                        continue
                    from unifiedpdf.measurements import extract_measurements
                    obj = {
                        "doc_id": pdf.stem,
                        "filename": pdf.name,
                        "page": idx,
                        "start": 0,
                        "length": len(txt),
                        "text": txt.strip(),
                        "extra": {"section": None, "paragraph_id": idx, "measurements": extract_measurements(txt)},
                    }
                    lines.append(obj)
                doc.close()
            except Exception:
                # Fallback: one chunk whole doc
                obj = {
                    "doc_id": pdf.stem,
                    "filename": pdf.name,
                    "page": None,
                    "start": 0,
                    "length": len(text),
                    "text": text.strip(),
                    "extra": {"section": None, "paragraph_id": 0},
                }
                lines.append(obj)
        elif args.chunking == "window":
            try:
                from unifiedpdf.pdf_processor import PDFProcessor, PDFChunkConfig
                cfg = PDFChunkConfig(
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    enable_wastewater_chunking=args.wt_enable,
                    wastewater_chunk_size=args.wt_size,
                    wastewater_overlap_ratio=args.wt_overlap_ratio,
                )
                proc = PDFProcessor(cfg)
                chunks = proc.chunk_text(doc_id=pdf.stem, filename=pdf.name, text=text)
                for ch in chunks:
                    lines.append(
                        {
                            "doc_id": ch.doc_id,
                            "filename": ch.filename,
                            "page": ch.page,
                            "start": ch.start_offset,
                            "length": ch.length,
                            "text": ch.text,
                            "extra": ch.extra,
                        }
                    )
            except Exception:
                # Fallback to paragraph chunking on failure
                paras = [p.strip() for p in text.split("\n\n") if p.strip()]
                offset = 0
                from unifiedpdf.measurements import extract_measurements
                for i, p in enumerate(paras):
                    obj = {
                        "doc_id": pdf.stem,
                        "filename": pdf.name,
                        "page": None,
                        "start": offset,
                        "length": len(p),
                        "text": p,
                        "extra": {"section": None, "paragraph_id": i, "measurements": extract_measurements(p)},
                    }
                    lines.append(obj)
                    offset += len(p) + 2
        else:
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            offset = 0
            from unifiedpdf.measurements import extract_measurements
            for i, p in enumerate(paras):
                obj = {
                    "doc_id": pdf.stem,
                    "filename": pdf.name,
                    "page": None,
                    "start": offset,
                    "length": len(p),
                    "text": p,
                    "extra": {"section": None, "paragraph_id": i, "measurements": extract_measurements(p)},
                }
                lines.append(obj)
                offset += len(p) + 2

        # 4) Optional LLM-based post-correction (model-agnostic, local LLM) on low-quality spans
        use_llm_correction = bool(args.llm_correct or args.ocr_lite)
        lc_threshold = args.llm_correct_threshold if args.llm_correct else args.ocr_lite_threshold
        lc_max_chars = args.llm_correct_max_chars if args.llm_correct else args.ocr_lite_max_chars
        lc_batch = args.llm_correct_batch if args.llm_correct else args.ocr_lite_batch
        if use_llm_correction and lines:
            try:
                from unifiedpdf.ocr_corrector import apply_llm_post_correction
                from unifiedpdf.config import PipelineConfig
                model_name = PipelineConfig().model_name
                # Load domain dictionary string
                dictionary = None
                if args.llm_dict_file:
                    try:
                        dictionary = Path(args.llm_dict_file).read_text(encoding="utf-8")
                    except Exception:
                        dictionary = None
                if not dictionary:
                    dictionary = "\n".join([
                        "[응집제] 폴리염화알루미늄, 황산알루미늄, 염화철",
                        "[시설물] 1차침전지, 2차침전지, 모래여과지, 활성탄여과지",
                        "[공정] 응집, 침전, 여과, 소독, 고도처리",
                        "[수질지표] 탁도, 잔류염소, pH, 용존산소, 생물학적산소요구량",
                        "단위: mg/L, ppm, ug/L, ppb, L/s, m3/d, ℃, °C, µS/cm, μS/cm",
                    ])
                # collect current doc lines (just added) by filename match
                doc_indices = [i for i, o in enumerate(lines) if o.get("filename") == pdf.name]
                texts = [lines[i]["text"] for i in doc_indices]
                corrected = apply_llm_post_correction(
                    texts,
                    model_name=model_name,
                    threshold=lc_threshold,
                    max_chars=lc_max_chars,
                    batch=lc_batch,
                    dictionary=dictionary,
                )
                for k, idx in enumerate(doc_indices):
                    lines[idx]["text"] = corrected[k]
            except Exception:
                # best-effort; ignore failures silently to keep pipeline light
                pass

        # 5) Numeric-focused expanded chunks (doc-level)
        if args.numeric_expand and lines:
            try:
                from unifiedpdf.measurements import extract_measurements
                # gather current doc lines for this pdf
                doc_idxs = [i for i, o in enumerate(lines) if o.get("filename") == pdf.name]
                # sort by start offset for neighbor lookup
                doc_idxs.sort(key=lambda i: (lines[i].get("start", 0)))
                added_records = []
                for pos, idx in enumerate(doc_idxs):
                    rec = lines[idx]
                    text = rec.get("text", "")
                    meas = extract_measurements(text)
                    if not meas:
                        continue
                    # Neighbor window
                    prev_text = lines[doc_idxs[pos - 1]].get("text", "") if pos - 1 >= 0 and args.numeric_window >= 1 else ""
                    next_text = lines[doc_idxs[pos + 1]].get("text", "") if pos + 1 < len(doc_idxs) and args.numeric_window >= 1 else ""
                    # Build expanded text (limit to reasonable size)
                    expanded = (prev_text[-300:] + "\n" if prev_text else "") + text + ("\n" + next_text[:300] if next_text else "")
                    # Emphasize numbers by repetition at tail
                    if args.numeric_emphasis > 1:
                        nums = [n for n, _ in meas]
                        if nums:
                            tail = " " + " ".join(nums) * (args.numeric_emphasis - 1)
                            expanded = expanded + tail
                    new_obj = {
                        "doc_id": rec.get("doc_id"),
                        "filename": rec.get("filename"),
                        "page": rec.get("page"),
                        "start": rec.get("start", 0),
                        "length": len(expanded),
                        "text": expanded,
                        "extra": {**(rec.get("extra", {})), "numeric_expanded": True},
                    }
                    added_records.append(new_obj)
                lines.extend(added_records)
            except Exception:
                pass

    # Write only if we have content; otherwise keep existing corpus (if any)
    if len(lines) > 0:
        # 중복 제거 적용
        if args.dedup:
            print(f"중복 제거 전: {len(lines)}개 청크")
            # JSON 객체를 Chunk 객체로 변환
            chunks = []
            for obj in lines:
                chunks.append(Chunk(
                    doc_id=obj.get("doc_id", obj.get("filename", "doc")),
                    filename=obj.get("filename", "doc"),
                    page=obj.get("page"),
                    start_offset=int(obj.get("start", 0)),
                    length=int(obj.get("length", len(obj.get("text", "")))),
                    text=obj.get("text", ""),
                    extra=obj.get("extra", {}),
                ))
            
            # 중복 제거 실행
            unique_chunks = deduplicate_chunks(
                chunks, 
                threshold=args.dedup_threshold, 
                min_length=args.dedup_min_length
            )
            
            # Chunk 객체를 다시 JSON 객체로 변환
            lines = []
            for chunk in unique_chunks:
                lines.append({
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "start": chunk.start_offset,
                    "length": chunk.length,
                    "text": chunk.text,
                    "extra": chunk.extra,
                })
            print(f"중복 제거 후: {len(lines)}개 청크")
        
        # Backup existing file once
        if out.exists():
            try:
                bak = out.with_suffix(out.suffix + ".bak")
                if not bak.exists():
                    out.replace(bak)
                else:
                    # overwrite bak
                    bak.unlink(missing_ok=True)
                    out.replace(bak)
            except Exception:
                pass
        with out.open("w", encoding="utf-8") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(lines)} chunks to {out}")


if __name__ == "__main__":
    main()
