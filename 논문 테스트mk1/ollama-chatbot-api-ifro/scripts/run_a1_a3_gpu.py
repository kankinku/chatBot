import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
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


def write_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build A1/A2/A3 corpora and benchmark using GPU vector search")
    ap.add_argument("--pdf_dir", default=str(API_DIR / "data" / "pdfs"))
    ap.add_argument("--qa", default=str(API_DIR / "data" / "tests" / "qa.json"))
    ap.add_argument("--outdir", default=str((API_DIR.parent) / "out"))
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--aux-per-doc", type=int, default=50)
    ap.add_argument("--backend", default="faiss", choices=["faiss", "hnsw"])  # GPU-friendly backends
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--no-clean", action="store_true", help="Do not delete cached extracted/corpus/vector_store before running")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    qa_path = Path(args.qa)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Clean previous artifacts unless --no-clean
    if not args.no_clean:
        extracted = API_DIR / "data" / "extracted_text.jsonl"
        corpus_a1 = API_DIR / "data" / "corpus_A1.jsonl"
        corpus_a2 = API_DIR / "data" / "corpus_A2.jsonl"
        corpus_a3 = API_DIR / "data" / "corpus_A3.jsonl"
        vs_a1 = API_DIR / f"vector_store_A1_{args.backend}"
        vs_a2 = API_DIR / f"vector_store_A2_{args.backend}"
        vs_a3 = API_DIR / f"vector_store_A3_{args.backend}"

        def _rm(p: Path):
            if p.is_file():
                try:
                    p.unlink()
                    print(f"[CLEAN] Removed file: {p}")
                except Exception:
                    pass
            elif p.is_dir():
                import shutil
                try:
                    shutil.rmtree(p)
                    print(f"[CLEAN] Removed directory: {p}")
                except Exception:
                    pass

        for p in [extracted, corpus_a1, corpus_a2, corpus_a3, vs_a1, vs_a2, vs_a3]:
            _rm(p)

    # 1) Extract once (no OCR)
    extracted = API_DIR / "data" / "extracted_text.jsonl"
    need_extract = args.force_extract or (not extracted.exists()) or (extracted.exists() and extracted.stat().st_size == 0)
    if need_extract:
        extracted.parent.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, str(SCRIPTS / "extract_text_from_pdfs.py"),
            "--pdf_dir", str(pdf_dir),
            "--out", str(extracted),
            "--ocr", "off",
        ], step="Extract PDF text (no OCR)")
    else:
        print(f"[INFO] Using cached extracted text: {extracted}")

    # 2) Build A1 (fixed non-sliding), A2 (sliding), and A3 (numeric-aware + aux)
    corpus_a1 = API_DIR / "data" / "corpus_A1.jsonl"
    corpus_a2 = API_DIR / "data" / "corpus_A2.jsonl"
    corpus_a3 = API_DIR / "data" / "corpus_A3.jsonl"

    run([
        sys.executable, str(SCRIPTS / "build_corpus_from_text.py"),
        "--input", str(extracted), "--out", str(corpus_a1),
        "--chunk-size", str(args.chunk_size), "--chunk-overlap", "0",
        "--strategy", "fixed", "--disable-numeric-window",
    ], step="Build A1 corpus (fixed non-sliding)")

    run([
        sys.executable, str(SCRIPTS / "build_corpus_from_text.py"),
        "--input", str(extracted), "--out", str(corpus_a2),
        "--chunk-size", str(args.chunk_size), "--chunk-overlap", str(args.chunk_overlap),
        "--strategy", "sliding", "--disable-numeric-window",
    ], step="Build A2 corpus (baseline sliding)")

    run([
        sys.executable, str(SCRIPTS / "build_corpus_from_text.py"),
        "--input", str(extracted), "--out", str(corpus_a3),
        "--chunk-size", str(args.chunk_size), "--chunk-overlap", str(args.chunk_overlap),
        "--enable-numeric-window", "--aux-per-doc", str(args.aux_per_doc),
    ], step="Build A3 corpus (numeric-aware + aux)")

    # 3) Build GPU-backed vector indices (FAISS/HNSW)
    vs_a1 = API_DIR / f"vector_store_A1_{args.backend}"
    vs_a2 = API_DIR / f"vector_store_A2_{args.backend}"
    vs_a3 = API_DIR / f"vector_store_A3_{args.backend}"
    run([
        sys.executable, str(SCRIPTS / "build_vector_index.py"),
        "--corpus", str(corpus_a1), "--backend", args.backend, "--outdir", str(vs_a1), "--use-gpu",
    ], step=f"Build {args.backend.upper()} index for A1")
    run([
        sys.executable, str(SCRIPTS / "build_vector_index.py"),
        "--corpus", str(corpus_a2), "--backend", args.backend, "--outdir", str(vs_a2), "--use-gpu",
    ], step=f"Build {args.backend.upper()} index for A2")
    run([
        sys.executable, str(SCRIPTS / "build_vector_index.py"),
        "--corpus", str(corpus_a3), "--backend", args.backend, "--outdir", str(vs_a3), "--use-gpu",
    ], step=f"Build {args.backend.upper()} index for A3")

    # 4) Prepare per-corpus GPU configs
    cfg_a1 = out_dir / "config_A1_gpu.json"
    cfg_a2 = out_dir / "config_A2_gpu.json"
    cfg_a3 = out_dir / "config_A3_gpu.json"
    write_json(cfg_a1, {
        "flags": {"use_gpu": True, "store_backend": args.backend, "enable_parallel_search": True},
        "vector_store_dir": str(vs_a1),
    })
    write_json(cfg_a2, {
        "flags": {"use_gpu": True, "store_backend": args.backend, "enable_parallel_search": True},
        "vector_store_dir": str(vs_a2),
    })
    write_json(cfg_a3, {
        "flags": {"use_gpu": True, "store_backend": args.backend, "enable_parallel_search": True},
        "vector_store_dir": str(vs_a3),
    })

    # 5) Run QA benchmark for A1, A2 and A3
    rep_a1 = out_dir / "report_A1.json"
    rep_a2 = out_dir / "report_A2.json"
    rep_a3 = out_dir / "report_A3.json"
    csv_a1 = out_dir / "report_A1.csv"
    csv_a2 = out_dir / "report_A2.csv"
    csv_a3 = out_dir / "report_A3.csv"

    run([
        sys.executable, str(SCRIPTS / "run_qa_benchmark_v2.py"),
        "--input", str(qa_path), "--corpus", str(corpus_a1),
        "--report", str(rep_a1), "--csv", str(csv_a1), "--config", str(cfg_a1),
    ], step="Benchmark A1 (GPU vector search)")

    run([
        sys.executable, str(SCRIPTS / "run_qa_benchmark_v2.py"),
        "--input", str(qa_path), "--corpus", str(corpus_a2),
        "--report", str(rep_a2), "--csv", str(csv_a2), "--config", str(cfg_a2),
    ], step="Benchmark A2 (GPU vector search)")

    run([
        sys.executable, str(SCRIPTS / "run_qa_benchmark_v2.py"),
        "--input", str(qa_path), "--corpus", str(corpus_a3),
        "--report", str(rep_a3), "--csv", str(csv_a3), "--config", str(cfg_a3),
    ], step="Benchmark A3 (GPU vector search)")

    # 6) Combined summary (A1 vs A2 vs A3) with deltas
    def _load_metrics(p: Path) -> dict:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            g = (obj.get("paper", {}) or {}).get("global", {})
            return {
                "NUM": float(g.get("numeric_match", 0.0)),
                "TOKF1": float(g.get("token_f1", 0.0)),
                "RELAX": float(g.get("relaxed_em", 0.0)),
                "SEM": float(g.get("semantic_sim", 0.0)),
                "COMP": float(g.get("composite", 0.0)),
                "NP": float(g.get("numeric_preservation", 0.0)) if "numeric_preservation" in g else None,
                "Latency_ms": g.get("latency_mean_ms", {}),
            }
        except Exception:
            return {}

    m_a1 = _load_metrics(rep_a1)
    m_a2 = _load_metrics(rep_a2)
    m_a3 = _load_metrics(rep_a3)
    summary = {
        "A1": m_a1,
        "A2": m_a2,
        "A3": m_a3,
        "deltas": {
            "A2_minus_A1": {
                k: (m_a2.get(k, 0.0) - m_a1.get(k, 0.0))
                for k in ["EM@5%", "UEM@5%", "NCI", "NP", "RefF1", "BLEU1"]
                if m_a1.get(k) is not None and m_a2.get(k) is not None
            },
            "A3_minus_A1": {
                k: (m_a3.get(k, 0.0) - m_a1.get(k, 0.0))
                for k in ["EM@5%", "UEM@5%", "NCI", "NP", "RefF1", "BLEU1"]
                if m_a1.get(k) is not None and m_a3.get(k) is not None
            }
        }
    }
    out_sum = out_dir / "summary_a1_a2_a3.json"
    write_json(out_sum, summary)

    # Pretty print concise summary
    def _fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
    print("\n=== A1 vs A2 vs A3 (GPU) Summary (Composite) ===")
    for name, m in [("A1", m_a1), ("A2", m_a2), ("A3", m_a3)]:
        if not m:
            print(f"{name}: (no data)")
            continue
        print(f"{name}: COMP {_fmt(m.get('COMP', 0))}  NUM {_fmt(m.get('NUM', 0))}  TOKF1 {_fmt(m.get('TOKF1', 0))}  SEM {_fmt(m.get('SEM', 0))}  NP {_fmt(m.get('NP', 0))}")
    d2 = summary["deltas"].get("A2_minus_A1", {})
    if d2:
        print("Delta (A2 - A1): ", ", ".join([f"{k} {_fmt(v)}" for k, v in d2.items()]))
    d3 = summary["deltas"].get("A3_minus_A1", {})
    if d3:
        print("Delta (A3 - A1): ", ", ".join([f"{k} {_fmt(v)}" for k, v in d3.items()]))

    print(f"\n[DONE] A1/A2/A3 GPU benchmarks complete. Reports under: {out_dir}")


if __name__ == "__main__":
    main()
