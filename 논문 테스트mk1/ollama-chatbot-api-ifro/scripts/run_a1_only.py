import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent  # ollama-chatbot-api-ifro/
SCRIPTS = API_DIR / "scripts"


def run(cmd: list[str], step: str = "") -> None:
    if step:
        print(f"\n[STEP] {step}")
    print("$", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("OLLAMA_HOST", "127.0.0.1")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"Command failed ({r.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description="Build A1 (fixed sliding) corpus and run QA benchmark")
    ap.add_argument("--pdf_dir", default=str(API_DIR / "data" / "pdfs"))
    ap.add_argument("--qa", default=str(API_DIR / "data" / "tests" / "qa.json"))
    ap.add_argument("--outdir", default=str((API_DIR.parent) / "out"))
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=128)
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    qa_path = Path(args.qa)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = API_DIR / "data" / "extracted_text.jsonl"
    need_extract = args.force_extract or (not extracted.exists()) or (extracted.exists() and extracted.stat().st_size == 0)
    if need_extract:
        extracted.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(SCRIPTS / "extract_text_from_pdfs.py"),
            "--pdf_dir", str(pdf_dir),
            "--out", str(extracted),
            "--ocr", "off",
        ]
        run(cmd, step="Extract PDF text (no OCR)")
    else:
        print(f"[INFO] Using cached extracted text: {extracted}")

    # Build A1: fixed-size sliding windows, numeric-aware disabled
    corpus_a1 = API_DIR / "data" / "corpus_A1.jsonl"
    cmd = [
        sys.executable, str(SCRIPTS / "build_corpus_from_text.py"),
        "--input", str(extracted),
        "--out", str(corpus_a1),
        "--chunk-size", str(args.chunk_size),
        "--chunk-overlap", "0",
        "--strategy", "sliding",
        "--disable-numeric-window",
    ]
    run(cmd, step="Build A1 corpus (fixed sliding)")

    # Run QA benchmark on A1
    rep = out_dir / "report_A1.json"
    csv = out_dir / "report_A1.csv"
    cmd = [
        sys.executable, str(SCRIPTS / "run_qa_benchmark.py"),
        "--input", str(qa_path),
        "--corpus", str(corpus_a1),
        "--report", str(rep),
        "--csv", str(csv),
    ]
    run(cmd, step="QA benchmark on A1")

    print(f"\n[DONE] A1 corpus + benchmark complete. Reports under: {out_dir}")


if __name__ == "__main__":
    main()

