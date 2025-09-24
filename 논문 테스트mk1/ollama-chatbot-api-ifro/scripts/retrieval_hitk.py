import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from unifiedpdf.config import PipelineConfig
from unifiedpdf.facade import UnifiedPDFPipeline
from unifiedpdf.measurements import extract_measure_spans, normalize_unit


def load_corpus(path: str):
    chunks = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            from unifiedpdf.types import Chunk
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


def parse_gold_value_unit(s: str):
    import re
    # match number + unit (similar to run_qa_benchmark)
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*([A-Za-z%°µ/‰]+)", s or "")
    if not m:
        return None, None
    try:
        v = float(m.group(1).replace(",", ""))
    except Exception:
        return None, None
    return v, normalize_unit(m.group(2))


def compute_nci(contexts: List[str], gold: str) -> int:
    gv, gu = parse_gold_value_unit(gold)
    if gv is None or gu is None:
        return 0
    tok = f"{gv}".replace(",", "")
    for ctx in contexts:
        if (tok in ctx) and any(sp.label_hint for sp in extract_measure_spans(ctx)):
            return 1
    return 0


def hit_at_k(pipe: UnifiedPDFPipeline, question: str, gold: str, k: int = 5) -> int:
    # retrieve only; use balanced rrf weights
    spans, _ = pipe.retriever.retrieve(question, topk_each=50, rrf_k=pipe.cfg.rrf.base_rrf_k, rrf_weights=(0.5, 0.5))
    contexts = spans[:k]
    texts = [s.chunk.text for s in contexts]
    return compute_nci(texts, gold)


def main():
    ap = argparse.ArgumentParser(description="Compute retrieval Hit@k (numeric+label co-occurrence) for one or more corpora")
    ap.add_argument("--input", default="data/tests/qa.json")
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--store-backend", default="auto", choices=["auto", "faiss", "hnsw", "tfidf"])
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Normalize input schema (align with run_qa_benchmark)
    if isinstance(raw, dict):
        if "qa_pairs" in raw and isinstance(raw["qa_pairs"], list):
            items = []
            for obj in raw["qa_pairs"]:
                if not isinstance(obj, dict):
                    continue
                q = obj.get("question") or obj.get("question_kr") or obj.get("q") or ""
                a = obj.get("answer") or obj.get("answer_kr") or obj.get("a") or ""
                items.append({"question": q, "answer": a})
        else:
            items = raw.get("items") or raw.get("data") or []
    else:
        items = raw

    names = [Path(p).stem for p in args.corpus]
    if args.names:
        ns = [n.strip() for n in args.names.split(",") if n.strip()]
        if len(ns) == len(args.corpus):
            names = ns

    out = {}
    for name, cpath in zip(names, args.corpus):
        chunks = load_corpus(cpath)
        cfg = PipelineConfig()
        cfg.flags.store_backend = args.store_backend
        pipe = UnifiedPDFPipeline(chunks, cfg)
        hits = 0
        for it in items:
            if isinstance(it, dict):
                q = it.get("question") or it.get("question_kr") or it.get("q") or ""
                gold = it.get("answer") or it.get("answer_kr") or it.get("a") or ""
            else:
                # if item is a plain string, treat as question with empty gold
                q = str(it)
                gold = ""
            hits += hit_at_k(pipe, q, gold, k=args.k)
        N = max(1, len(items))
        out[name] = {"k": args.k, "hit@k": hits / N}

    print(json.dumps({"hitk": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

