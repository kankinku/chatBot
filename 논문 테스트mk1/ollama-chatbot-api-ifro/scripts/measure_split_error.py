import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from unifiedpdf.measurements import extract_measure_spans


def load_extracted(path: str) -> Dict[str, str]:
    """Return mapping doc_id -> full text."""
    texts: Dict[str, str] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id") or obj.get("filename") or "doc"
            texts[str(doc_id)] = obj.get("text", "") or ""
    return texts


def load_corpus(path: str) -> Dict[str, List[dict]]:
    per_doc: Dict[str, List[dict]] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            did = str(obj.get("doc_id") or obj.get("filename") or "doc")
            per_doc.setdefault(did, []).append(obj)
    return per_doc


def anchor_contained(sp, chunks: List[dict]) -> bool:
    """Check if a measure span (number+unit and optional label) is contained within any single chunk."""
    raw = sp.raw or ""
    label = (sp.label_hint or "").strip()
    for ch in chunks:
        t = ch.get("text", "") or ""
        if not t:
            continue
        if raw and raw in t:
            if label:
                if label in t:
                    return True
                else:
                    continue
            return True
        # fallback: both number and unit separately
        if sp.value is not None and sp.unit:
            num_token = str(sp.value)
            if num_token in t and sp.unit in t:
                if not label or label in t:
                    return True
    return False


def eval_split_error(extracted_path: str, corpus_path: str) -> Tuple[int, int, float]:
    texts = load_extracted(extracted_path)
    corp = load_corpus(corpus_path)
    total = 0
    ok = 0
    for did, full in texts.items():
        spans = extract_measure_spans(full or "")
        if not spans:
            continue
        chunks = corp.get(did, [])
        for sp in spans:
            # consider only anchors that are meaningful (has unit)
            if not sp.unit:
                continue
            total += 1
            if anchor_contained(sp, chunks):
                ok += 1
    err = 0.0 if total == 0 else (1.0 - ok / total)
    return total, ok, err


def main():
    ap = argparse.ArgumentParser(description="Measure Split Error Rate of numeric anchors in a corpus")
    ap.add_argument("--input", default="data/extracted_text.jsonl", help="Path to extracted text JSONL")
    ap.add_argument("--corpus", action="append", required=True, help="Corpus JSONL path (repeatable)")
    ap.add_argument("--names", default=None, help="Optional comma-separated names for corpora")
    ap.add_argument("--out", default=None, help="Optional JSON output file for summary")
    args = ap.parse_args()

    names = [Path(p).stem for p in args.corpus]
    if args.names:
        ns = [n.strip() for n in args.names.split(",") if n.strip()]
        if len(ns) == len(args.corpus):
            names = ns

    results = {}
    for name, path in zip(names, args.corpus):
        tot, ok, err = eval_split_error(args.input, path)
        results[name] = {"anchors": tot, "contained": ok, "split_error_rate": err}

    # Compute deltas vs first
    order = names
    if order:
        base = results[order[0]]
        for n in order[1:]:
            results[n]["delta_vs_" + order[0]] = base["split_error_rate"] - results[n]["split_error_rate"]

    print(json.dumps({"split_error": results}, ensure_ascii=False, indent=2))
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({"split_error": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

