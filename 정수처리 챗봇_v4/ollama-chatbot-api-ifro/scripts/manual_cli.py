import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List

# PyTorch 경고 메시지 숨기기
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")

# Ensure 'src' is on sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from unifiedpdf.config import PipelineConfig
from unifiedpdf.facade import UnifiedPDFPipeline
from unifiedpdf.types import Chunk


def load_corpus(path: str) -> List[Chunk]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    chunks: List[Chunk] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            chunks.append(
                Chunk(
                    doc_id=obj.get("doc_id", obj.get("filename", "doc")),
                    filename=obj.get("filename", "doc"),
                    page=obj.get("page"),
                    start_offset=int(obj.get("start", 0)),
                    length=int(obj.get("length", len(obj.get("text", "")))),
                    text=obj.get("text", ""),
                    extra=obj.get("extra", {}),
                )
            )
    return chunks


def main():
    ap = argparse.ArgumentParser(description="Manual CLI for UnifiedPDFPipeline")
    ap.add_argument("--corpus", default="data/corpus_v1.jsonl", help="Path to JSONL corpus")
    ap.add_argument("--mode", default="accuracy", choices=["accuracy", "speed"])
    ap.add_argument("--k", default="auto")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-warmup", action="store_true", help="Skip initial model warm-up")
    ap.add_argument("--store-backend", default="auto", choices=["auto", "faiss", "hnsw"]) 
    ap.add_argument("--vector-store-dir", default="vector_store", help="Directory of prebuilt vector index")
    ap.add_argument("--use-cross-reranker", action="store_true")
    ap.add_argument("--rerank-top-n", type=int, default=50)
    ap.add_argument("--question", default=None, help="Optional single-shot question")
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.flags.mode = args.mode
    cfg.flags.use_gpu = args.gpu
    cfg.flags.store_backend = args.store_backend
    cfg.vector_store_dir = args.vector_store_dir
    cfg.flags.use_cross_reranker = args.use_cross_reranker
    cfg.flags.rerank_top_n = args.rerank_top_n

    chunks = load_corpus(args.corpus)
    pipe = UnifiedPDFPipeline(chunks, cfg)

    # Warm-up to load models/modules once (LLM, reranker, vector backends)
    if not args.no_warmup:
        try:
            _ = pipe.ask("웜업", mode=args.mode)
        except Exception:
            pass

    def run_once(q: str):
        ans = pipe.ask(q, mode=args.mode)
        print("\n=== Answer ===")
        print(ans.text)
        print("\n=== Confidence ===")
        print(f"{ans.confidence:.3f}")
        print("\n=== Sources ===")
        for i, s in enumerate(ans.sources, start=1):
            conf = s.calibrated_conf if s.calibrated_conf is not None else 0.0
            print(f"[{i}] {s.chunk.filename} p.{s.chunk.page} off {s.chunk.start_offset} (conf={conf:.3f})")
            print(s.chunk.text[:200].replace("\n", " ") + ("..." if len(s.chunk.text) > 200 else ""))
        print("\n=== Metrics ===")
        print(json.dumps(ans.metrics, ensure_ascii=False, indent=2))
        print(f"fallback_used: {ans.fallback_used}")

    if args.question:
        run_once(args.question)
        return

    print("UnifiedPDFPipeline manual CLI. Type a question (or 'exit').")
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        run_once(q)


if __name__ == "__main__":
    main()
