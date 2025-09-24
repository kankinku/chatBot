import argparse
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


# Reuse unifiedpdf via src path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


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
    try:
        import pymupdf4llm  # type: ignore
    except Exception:
        return ""
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        import re
        text = re.sub(r'#{1,6}\s*', '', md_text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        return text
    except Exception:
        return ""


def extract_text_auto(pdf_path: Path) -> str:
    t = extract_text_pymupdf4llm(pdf_path)
    if t and t.strip():
        return t
    t = extract_text_pdfplumber(pdf_path)
    if t and t.strip():
        return t
    t = extract_text_fitz(pdf_path)
    return t


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
    if getattr(pytesseract.pytesseract, "tesseract_cmd", None):
        return
    tcmd = os.environ.get("TESSERACT_CMD")
    if tcmd and Path(tcmd).exists():
        pytesseract.pytesseract.tesseract_cmd = tcmd
        return
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
            return extract_text_fitz(out_pdf)
        except Exception:
            return ""


def main():
    ap = argparse.ArgumentParser(description="Extract and cache plain text from PDFs (one-time)")
    ap.add_argument("--pdf_dir", default="ollama-chatbot-api-ifro/data/pdfs")
    ap.add_argument("--out", default="data/extracted_text.jsonl")
    ap.add_argument("--pdf-extractor", default="auto", choices=["auto", "plumber", "fitz", "pymupdf4llm"], help="PDF text extractor")
    # Default to OCR always for image-heavy PDFs
    ap.add_argument("--ocr", default="always", choices=["auto", "always", "off"], help="OCR policy (default: always)")
    # Choose OCR engine (default: ocrmypdf only)
    ap.add_argument("--ocr-engine", default="ocrmypdf", choices=["ocrmypdf", "tesseract"], help="OCR engine to use when OCR is enabled")
    ap.add_argument("--ocr-lang", default="kor+eng")
    ap.add_argument("--ocr-dpi", type=int, default=300)
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for pdf in pdf_dir.glob("**/*.pdf"):
        # Extract text
        if args.pdf_extractor == "plumber":
            text = extract_text_pdfplumber(pdf)
        elif args.pdf_extractor == "fitz":
            text = extract_text_fitz(pdf)
        elif args.pdf_extractor == "pymupdf4llm":
            text = extract_text_pymupdf4llm(pdf)
        else:
            text = extract_text_auto(pdf)

        # OCR policy
        need_ocr = (args.ocr == "always") or (args.ocr == "auto" and not has_meaningful_text(text))
        if need_ocr:
            txt = ""
            if args.ocr_engine == "ocrmypdf":
                txt = ocr_with_ocrmypdf(pdf, lang=args.ocr_lang)
            else:
                txt = ocr_with_tesseract_via_fitz(pdf, dpi=args.ocr_dpi, lang=args.ocr_lang)
            if has_meaningful_text(txt):
                text = txt
            else:
                if args.ocr_engine == "ocrmypdf":
                    print(f"[WARN] OCR (ocrmypdf) failed for {pdf.name}. Ensure ocrmypdf, Tesseract, Ghostscript, and QPDF are installed.")
                else:
                    print(f"[WARN] OCR (tesseract) failed for {pdf.name}. Ensure Tesseract is installed and TESSERACT_CMD is set.")

        if not has_meaningful_text(text):
            continue

        lines.append({
            "doc_id": pdf.stem,
            "filename": pdf.name,
            "text": text.strip(),
            "text_len": len(text.strip()),
        })

    with out.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved extracted text for {len(lines)} docs to {out}")


if __name__ == "__main__":
    main()
