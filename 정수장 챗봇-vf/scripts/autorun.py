import argparse
import os
import subprocess
import sys
from pathlib import Path
import json
import time
import urllib.request


def sh(cmd: list[str], check: bool = False) -> int:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check).returncode


def has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def ensure_pip_packages() -> None:
    # Minimal fast path check; otherwise install from requirements
    needed = [
        ("pymupdf", "pymupdf"),
        ("pytesseract", "pytesseract"),
        ("sentence_transformers", "sentence-transformers"),
        ("hnswlib", "hnswlib"),
    ]
    missing = [pypkg for mod, pypkg in needed if not has_module(mod)]
    if missing or not Path("requirements.txt").exists():
        # Install pinned stack; include extra index for PyTorch wheels if available
        sh(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "torch==2.1.2",
                "torchvision==0.16.2",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        args = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        # Prefer extra index URL for torch ecosystem as well
        args += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
        sh(args)


def _ollama_request(path: str, method: str = "GET", data: dict | None = None, timeout: int = 5):
    url = f"http://127.0.0.1:11434{path}"
    body = None
    headers = {"Content-Type": "application/json"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        try:
            return json.loads(raw)
        except Exception:
            return raw


def ensure_ollama_up() -> bool:
    try:
        _ollama_request("/api/tags", "GET", None, timeout=2)
        return True
    except Exception:
        print("Ollama server unreachable at http://127.0.0.1:11434 — please run 'ollama serve'.")
        return False


def ensure_ollama_model(model: str) -> bool:
    try:
        tags = _ollama_request("/api/tags")
        models = tags.get("models", []) if isinstance(tags, dict) else []
        if any(m.get("name") == model for m in models):
            return True
        print(f"Model '{model}' not found. Pulling via Ollama API...")
        # Trigger pull
        _ollama_request("/api/pull", method="POST", data={"name": model}, timeout=60)
        # Poll until available or timeout
        for _ in range(60):
            time.sleep(2)
            try:
                tags = _ollama_request("/api/tags")
                models = tags.get("models", []) if isinstance(tags, dict) else []
                if any(m.get("name") == model for m in models):
                    print(f"Model '{model}' is now available.")
                    return True
            except Exception:
                pass
        print(f"Model '{model}' not available after pull attempt.")
        return False
    except Exception:
        return False


def find_tesseract() -> str | None:
    # ENV override
    tcmd = os.environ.get("TESSERACT_CMD")
    if tcmd and Path(tcmd).exists():
        return tcmd
    # pytesseract configured?
    try:
        import pytesseract  # type: ignore

        if getattr(pytesseract.pytesseract, "tesseract_cmd", None):
            return pytesseract.pytesseract.tesseract_cmd
    except Exception:
        pass
    # Common Windows locations
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # PATH lookup
    which = shutil.which("tesseract") if "shutil" in globals() else None
    if not which:
        import shutil as _sh

        which = _sh.which("tesseract")
    return which


def have_tesseract() -> bool:
    return bool(find_tesseract())


def main():
    ap = argparse.ArgumentParser(description="Auto-setup and run benchmark or question")
    # --pdf 및 --qa 인자 제거: 고정 기본 경로 사용
    ap.add_argument("--question", default=None, help="Single question instead of benchmark")
    ap.add_argument("--server", action="store_true", help="Start API server after setup")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--backend", default="auto", choices=["auto", "faiss", "hnsw"])
    ap.add_argument("--auto", action="store_true", help="Run auto benchmark using QA file and save reports")
    args = ap.parse_args()

    # 고정 기본 경로 설정
    DEFAULT_PDF_DIR = Path("data/pdfs")
    DEFAULT_QA_FILE = "data/tests/qa.json"

    # Resolve PDF directory from fixed default
    candidates = [DEFAULT_PDF_DIR]
    pdf_dir = None
    pdfs = []
    for c in candidates:
        if c.exists():
            found = list(c.glob("**/*.pdf"))
            if found:
                pdf_dir = c
                pdfs = found
                break
    use_pdf = True
    if pdf_dir is None:
        # Fallback to existing corpus
        corpus = Path("data/corpus.jsonl")
        if corpus.exists() and corpus.stat().st_size > 0:
            print(f"No PDFs found in {', '.join(str(p) for p in candidates)}. Using existing corpus: {corpus}")
            use_pdf = False
        else:
            print(f"No PDFs found in {', '.join(str(p) for p in candidates)}, and no existing corpus at {corpus}.")
            print("Place PDFs in one of those directories or provide a corpus.jsonl and retry.")
            sys.exit(1)

    ensure_pip_packages()
    # Ensure Ollama and model availability
    if not ensure_ollama_up():
        sys.exit(1)
    # Use default model from config
    try:
        from unifiedpdf.config import PipelineConfig
        model_name = PipelineConfig().model_name
    except Exception:
        model_name = "llama3:8b-instruct-q4_K_M"
    if not ensure_ollama_model(model_name):
        sys.exit(1)

    # Decide OCR mode: avoid auto-install; use auto only if tesseract exists, else off
    ocr_mode = "auto" if have_tesseract() else "off"

    if args.server:
        cmd = [
            sys.executable, "scripts/quickstart.py",
            "--backend", args.backend,
        ]
        if use_pdf:
            # default to pymupdf4llm if available
            extractor = "pymupdf4llm"
            cmd += ["--pdf", str(pdf_dir), "--pdf-extractor", extractor, "--ocr", ocr_mode]
        else:
            cmd += ["--corpus", "data/corpus.jsonl"]
        cmd += ["--server", "--host", args.host, "--port", args.port]
        sh(cmd)
        return

    if args.auto:
        # Ensure corpus exists (build if PDFs available)
        if use_pdf:
            # Build only if PDFs changed since last build
            manifest_path = Path("data/corpus.manifest.json")
            def make_manifest(dirp: Path):
                items = []
                for p in dirp.glob("**/*.pdf"):
                    try:
                        st = p.stat()
                        items.append({"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)})
                    except Exception:
                        pass
                return {"files": items}
            new_man = make_manifest(pdf_dir)
            old_man = {}
            if manifest_path.exists():
                try:
                    old_man = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    old_man = {}
            changed = json.dumps(new_man, sort_keys=True) != json.dumps(old_man, sort_keys=True)
            if changed or not Path("data/corpus.jsonl").exists():
                sh([
                    sys.executable, "scripts/build_corpus_from_pdfs.py",
                    "--pdf_dir", str(pdf_dir),
                    "--out", "data/corpus.jsonl",
                    "--pdf-extractor", "pymupdf4llm",
                    "--ocr", ocr_mode,
                    "--chunking", "window",
                    "--chunk-size", "800",
                    "--chunk-overlap", "200",
                    "--llm-correct",
                ])
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(new_man, ensure_ascii=False, indent=2), encoding="utf-8")
        # Run benchmark with default QA and timestamped outputs
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = Path("out") / f"report_autorun_{ts}.json"
        out_csv = Path("out") / f"report_autorun_{ts}.csv"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "scripts/run_qa_benchmark.py",
            "--input", DEFAULT_QA_FILE,
            "--corpus", "data/corpus.jsonl",
            "--mode", "accuracy",
            "--store-backend", args.backend,
            "--report", str(out_json),
            "--csv", str(out_csv),
        ]
        sh(cmd)
        print(f"\nDone. Reports written to:\n- {out_json}\n- {out_csv}")
        return

    if args.question:
        cmd = [
            sys.executable, "scripts/quickstart.py",
            "--backend", args.backend,
        ]
        if use_pdf:
            cmd += ["--pdf", str(pdf_dir), "--pdf-extractor", "pymupdf4llm", "--ocr", ocr_mode]
        else:
            cmd += ["--corpus", "data/corpus.jsonl"]
        cmd += ["--question", args.question]
        sh(cmd)
        return

    # Default: 바로 CLI(대화형) 실행
    cmd = [
        sys.executable, "scripts/manual_cli.py",
        "--mode", "accuracy",
        "--store-backend", args.backend,
    ]
    if use_pdf:
        # build corpus first (no OCR if tesseract absent)
        sh([
            sys.executable, "scripts/build_corpus_from_pdfs.py",
            "--pdf_dir", str(pdf_dir),
            "--out", "data/corpus.jsonl",
            "--pdf-extractor", "pymupdf4llm",
            "--ocr", ocr_mode,
            "--chunking", "window",
            "--chunk-size", "800",
            "--chunk-overlap", "200",
            "--llm-correct",
        ])
        cmd += ["--corpus", "data/corpus.jsonl"]
    else:
        cmd += ["--corpus", "data/corpus.jsonl"]
    sh(cmd)


if __name__ == "__main__":
    main()
