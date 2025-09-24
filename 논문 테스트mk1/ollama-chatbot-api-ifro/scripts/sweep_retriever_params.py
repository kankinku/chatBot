import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from unifiedpdf.config import PipelineConfig
from unifiedpdf.facade import UnifiedPDFPipeline
from unifiedpdf.types import Chunk
from unifiedpdf.measurements import normalize_unit, convert_value, extract_measure_spans


def load_corpus(path: str) -> List[Chunk]:
    p = Path(path)
    chunks: List[Chunk] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            chunks.append(Chunk(
                doc_id=obj.get("doc_id", obj.get("filename", "doc")),
                filename=obj.get("filename", "doc"),
                page=obj.get("page"),
                start_offset=int(obj.get("start", 0)),
                length=int(obj.get("length", len(obj.get("text", "")))),
                text=obj.get("text", ""),
                extra=obj.get("extra", {}),
            ))
    return chunks


def parse_first_value_and_unit(s: str) -> Tuple[Optional[float], Optional[str]]:
    import re
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*([A-Za-z%°µ/‰]+)", s or "")
    if not m:
        return None, None
    try:
        val = float(m.group(1).replace(",", ""))
    except Exception:
        return None, None
    unit = normalize_unit(m.group(2))
    return val, unit


def score_em_uem(pred: str, gold: str, rel_tol: float = 0.05) -> Tuple[int, int]:
    gv, gu = parse_first_value_and_unit(gold)
    pv, pu = parse_first_value_and_unit(pred)
    if gv is None or gu is None or pv is None or pu is None:
        return 0, 0
    cv = convert_value(pv, pu, gu)
    if cv is None:
        return 0, 0
    em = int(abs(cv - gv) / max(1e-9, abs(gv)) <= rel_tol)
    uem = int(em == 1 and normalize_unit(pu) == normalize_unit(gu))
    return em, uem


def compute_nci(contexts: List[str], gold: str) -> int:
    gv, gu = parse_first_value_and_unit(gold)
    if gv is None or gu is None:
        return 0
    tok = f"{gv}".replace(",", "")
    for ctx in contexts:
        if (tok in ctx) and any(sp.label_hint for sp in extract_measure_spans(ctx)):
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description="Sweep retriever numeric bonuses and report EM/UEM/NCI")
    ap.add_argument("--input", default="data/tests/qa.json")
    ap.add_argument("--corpus", default="data/corpus_A2.jsonl")
    ap.add_argument("--aux", default="0.06,0.08,0.10")
    ap.add_argument("--dens", default="0.02,0.04,0.06")
    args = ap.parse_args()

    chunks = load_corpus(args.corpus)
    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    aux_vals = [float(x) for x in args.aux.split(",") if x]
    dens_vals = [float(x) for x in args.dens.split(",") if x]

    rows = []
    best = None
    for a in aux_vals:
        for d in dens_vals:
            cfg = PipelineConfig()
            cfg.retriever_aux_bonus = a
            cfg.retriever_density_bonus = d
            pipe = UnifiedPDFPipeline(chunks, cfg)
            em = uem = nci = 0
            for it in items:
                res = pipe.ask(it["question"], mode="accuracy")
                e, u = score_em_uem(res.text, it.get("answer", ""))
                em += e; uem += u
                nci += compute_nci([s.chunk.text for s in res.sources], it.get("answer", ""))
            N = max(1, len(items))
            row = {"aux_bonus": a, "density_bonus": d, "em@5%": em/N, "uem@5%": uem/N, "nci": nci/N}
            rows.append(row)
            if best is None or (row["em@5%"], row["uem@5%"], row["nci"]) > (best["em@5%"], best["uem@5%"], best["nci"]):
                best = row

    print(json.dumps({"results": rows, "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

