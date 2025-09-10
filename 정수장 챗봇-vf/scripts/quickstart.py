import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    ap = argparse.ArgumentParser(description="Easy runner: build corpus/index then run QA benchmark")
    ap.add_argument("--qa", default="data/tests/qa.json", help="QA JSON file path (for benchmark)")
    ap.add_argument("--pdf", default=None, help="PDF file or directory (if set, builds corpus)")
    ap.add_argument("--corpus", default="data/corpus.jsonl", help="Use existing corpus JSONL if provided")
    ap.add_argument("--backend", default="auto", choices=["auto", "faiss", "hnsw"], help="Vector backend")
    ap.add_argument("--use-cross-reranker", action="store_true", help="Enable cross-encoder reranker in accuracy mode")
    ap.add_argument("--rerank-top-n", type=int, default=50)
    ap.add_argument("--thr-base", type=float, default=None)
    ap.add_argument("--thr-numeric", type=float, default=None)
    ap.add_argument("--thr-long", type=float, default=None)
    ap.add_argument("--pdf-extractor", default="auto", choices=["auto", "plumber", "fitz"], help="PDF text extractor")
    ap.add_argument("--ocr", default="auto", choices=["auto", "always", "off"], help="OCR fallback policy")
    ap.add_argument("--ocr-lang", default="kor+eng", help="OCR languages")
    ap.add_argument("--ocr-dpi", type=int, default=200, help="OCR rasterization DPI")
    ap.add_argument("--chunking", default="window", choices=["window", "paragraph", "page"], help="Chunking strategy")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--wt-enable", action="store_true")
    ap.add_argument("--wt-size", type=int, default=900)
    ap.add_argument("--wt-overlap-ratio", type=float, default=0.25)
    # Simple modes
    ap.add_argument("--question", default=None, help="Ask a single question (skips benchmark)")
    ap.add_argument("--interactive", action="store_true", help="Start interactive CLI after corpus/index")
    ap.add_argument("--server", action="store_true", help="Start API server after corpus/index")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default="8000")
    args = ap.parse_args()

    # 1) Build corpus from PDFs if requested
    corpus_path = Path(args.corpus)
    if args.pdf is not None:
        pdf_path = Path(args.pdf)
        pdf_dir = pdf_path if pdf_path.is_dir() else pdf_path.parent
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable,
            "scripts/build_corpus_from_pdfs.py",
            "--pdf_dir", str(pdf_dir),
            "--out", str(corpus_path),
            "--pdf-extractor", args.pdf_extractor,
            "--ocr", args.ocr,
            "--ocr-lang", args.ocr_lang,
            "--ocr-dpi", str(args.ocr_dpi),
            "--chunking", args.chunking,
            "--chunk-size", str(args.chunk_size),
            "--chunk-overlap", str(args.chunk_overlap),
            *( ["--wt-enable"] if args.wt_enable else [] ),
            "--wt-size", str(args.wt_size),
            "--wt-overlap-ratio", str(args.wt_overlap_ratio),
        ])

    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}. Provide --pdf or a valid --corpus path.")
        sys.exit(1)

    # if corpus exists but empty, stop early with guidance
    try:
        with corpus_path.open("r", encoding="utf-8") as f:
            first = f.readline()
            if not first.strip():
                # Try backup
                bak = corpus_path.with_suffix(corpus_path.suffix + ".bak")
                if bak.exists():
                    print("Corpus empty. Falling back to previous backup corpus.")
                    corpus_path = bak
                else:
                    print("Corpus is empty (0 chunks). PDF가 스캔본일 수 있습니다.")
                    print("- 해결: --pdf-extractor fitz --ocr off 로 재시도하거나, Tesseract 설치 후 --ocr auto 사용")
                    sys.exit(1)
    except Exception:
        pass

    # 2) Build vector index if backend is faiss or hnsw
    if args.backend in {"faiss", "hnsw"}:
        run([sys.executable, "scripts/build_vector_index.py", "--corpus", str(corpus_path), "--backend", args.backend, "--outdir", "vector_store"])

    # 3) Simple modes
    if args.question:
        cmd = [
            sys.executable, "scripts/manual_cli.py",
            "--corpus", str(corpus_path),
            "--mode", "accuracy",
            "--store-backend", args.backend,
            "--question", args.question,
        ]
        if args.use_cross_reranker:
            cmd += ["--use-cross-reranker", "--rerank-top-n", str(args.rerank_top_n)]
        run(cmd)
        return

    if args.interactive:
        cmd = [
            sys.executable, "scripts/manual_cli.py",
            "--corpus", str(corpus_path),
            "--mode", "accuracy",
            "--store-backend", args.backend,
        ]
        if args.use_cross_reranker:
            cmd += ["--use-cross-reranker", "--rerank-top-n", str(args.rerank_top_n)]
        run(cmd)
        return

    if args.server:
        cmd = [
            sys.executable, "-m", "uvicorn", "server.app:app",
            "--host", args.host, "--port", args.port,
        ]
        run(cmd)
        return

    # 4) Default: run benchmark
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path("out") / f"report_{args.backend}_{'xrerank' if args.use_cross_reranker else 'base'}_{ts}.json"
    out_csv = Path("out") / f"report_{args.backend}_{'xrerank' if args.use_cross_reranker else 'base'}_{ts}.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_qa_benchmark.py",
        "--input", args.qa,
        "--corpus", str(corpus_path),
        "--mode", "accuracy",
        "--store-backend", args.backend,
        "--report", str(out_json),
        "--csv", str(out_csv),
    ]
    if args.use_cross_reranker:
        cmd += ["--use-cross-reranker", "--rerank-top-n", str(args.rerank_top_n)]
    if args.thr_base is not None:
        cmd += ["--thr-base", str(args.thr_base)]
    if args.thr_numeric is not None:
        cmd += ["--thr-numeric", str(args.thr_numeric)]
    if args.thr_long is not None:
        cmd += ["--thr-long", str(args.thr_long)]

    run(cmd)
    print(f"\nDone. Reports written to:\n- {out_json}\n- {out_csv}")


if __name__ == "__main__":
    main()
