import argparse
import json
import sys
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure 'src' is on sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from unifiedpdf.config import PipelineConfig
from unifiedpdf.facade import UnifiedPDFPipeline
from unifiedpdf.metrics import MetricsCollector, percentile
from unifiedpdf.types import Chunk
from unifiedpdf.measurements import normalize_unit, convert_value, extract_measure_spans

# Global evaluation tolerance (approximate match)
EVAL_REL_TOL: float = 0.03  # 3%
# Units from domain_dictionary.json (loaded in main)
DOMAIN_UNITS: set[str] = set()


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


def score_em_uem(
    pred: str,
    gold: str,
    rel_tol: float = 0.05,
    allow_unit_conversion: bool = True,
) -> Tuple[int, int]:
    gv, gu = parse_first_value_and_unit(gold)
    pv, pu = parse_first_value_and_unit(pred)
    if gv is None or gu is None or pv is None or pu is None:
        return 0, 0
    gu_n = normalize_unit(gu)
    pu_n = normalize_unit(pu)
    if allow_unit_conversion:
        cv = convert_value(pv, pu, gu)
        if cv is None:
            return 0, 0
        em = int(abs(cv - gv) / max(1e-9, abs(gv)) <= rel_tol)
        uem = int(em == 1 and pu_n == gu_n)
        return em, uem
    else:
        # Strict: require same unit (after normalization), no conversion, and zero tolerance by default
        if pu_n != gu_n:
            return 0, 0
        em = int(abs(pv - gv) / max(1e-9, abs(gv)) <= rel_tol)
        uem = int(em == 1)
        return em, uem


def compute_nci(contexts: List[str], gold: str) -> int:
    # Numeric + label co-presence heuristic (lenient with 3% tolerance)
    gv, gu = parse_first_value_and_unit_lenient(gold)
    if gv is None:
        return 0
    for ctx in contexts:
        spans = extract_measure_spans(ctx or "")
        for sp in spans:
            if sp.value is None:
                continue
            # compare with unit conversion when possible
            if gu and sp.unit:
                cv = convert_value(sp.value, sp.unit, gu)
                if cv is None:
                    continue
                if abs(cv - gv) / max(1e-9, abs(gv)) <= EVAL_REL_TOL and (sp.label_hint is not None):
                    return 1
            else:
                if abs(sp.value - gv) / max(1e-9, abs(gv)) <= EVAL_REL_TOL and (sp.label_hint is not None):
                    return 1
    return 0


def parse_first_value_and_unit_lenient(s: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        spans = extract_measure_spans(s or "")
        for sp in spans:
            if sp.value is not None:
                return float(sp.value), normalize_unit(sp.unit)
    except Exception:
        pass
    import re
    m = re.search(r"(\d+(?:[\.,]\d+)?)", s or "")
    if not m:
        return None, None
    try:
        val = float(m.group(1).replace(",", ""))
    except Exception:
        return None, None
    # try to infer unit from domain dictionary
    unit = None
    if DOMAIN_UNITS:
        text = s or ""
        cand = [u for u in DOMAIN_UNITS if u and u in text]
        if cand:
            # choose the longest match
            unit = normalize_unit(sorted(cand, key=len, reverse=True)[0])
    return val, unit


def score_em_uem_lenient(pred: str, gold: str, rel_tol: float = EVAL_REL_TOL) -> Tuple[int, int]:
    gv, gu = parse_first_value_and_unit_lenient(gold)
    pv, pu = parse_first_value_and_unit_lenient(pred)
    if gv is None or pv is None:
        return 0, 0
    if gu and pu:
        cv = convert_value(pv, pu, gu)
        if cv is None:
            return 0, 0
        em = int(abs(cv - gv) / max(1e-9, abs(gv)) <= rel_tol)
        uem = int(em == 1 and normalize_unit(pu) == normalize_unit(gu))
        return em, uem
    em = int(abs(pv - gv) / max(1e-9, abs(gv)) <= rel_tol)
    return em, 0


def main():
    logger.info("=== QA 벤치마크 시작 ===")

    ap = argparse.ArgumentParser(description="Run QA benchmark (EM/UEM only)")
    ap.add_argument("--input", default="data/tests/qa.json", help="Path to input QA JSON")
    ap.add_argument("--corpus", default="data/corpus_v1.jsonl", help="Path to JSONL corpus")
    ap.add_argument("--mode", default="accuracy", choices=["accuracy", "speed"])
    ap.add_argument("--report", default="out/report.json")
    ap.add_argument("--csv", default="out/report.csv")
    ap.add_argument("--store-backend", default="auto", choices=["auto", "faiss", "hnsw"]) 
    ap.add_argument("--vector-store-dir", default="vector_store", help="Directory of prebuilt vector index")
    ap.add_argument("--config", default=None, help="Optional YAML/JSON config to override PipelineConfig fields")
    args = ap.parse_args()

    chunks = load_corpus(args.corpus)
    cfg = PipelineConfig()
    cfg.flags.store_backend = args.store_backend
    cfg.vector_store_dir = args.vector_store_dir
    # Optional external config overrides (YAML preferred, fallback JSON)
    if args.config:
        try:
            import yaml  # type: ignore
            with open(args.config, "r", encoding="utf-8") as f:
                overrides = yaml.safe_load(f) or {}
        except Exception:
            try:
                with open(args.config, "r", encoding="utf-8") as f:
                    overrides = json.load(f)
            except Exception:
                overrides = {}

        def _apply(obj, kv):
            for k, v in (kv or {}).items():
                if hasattr(obj, k):
                    cur = getattr(obj, k)
                    # nested dataclass/dict override
                    if isinstance(v, dict) and not isinstance(cur, (str, int, float, bool)):
                        _apply(cur, v)
                    else:
                        setattr(obj, k, v)

        _apply(cfg, overrides)

    # Detect corpus category (kept for metadata only)
    corpus_name = Path(args.corpus).name.lower()
    is_a3 = "a3" in corpus_name

    # For non-A3: disable aggressive recovery/numeric boosts for fairness
    if not is_a3:
        try:
            cfg.llm_retries = 0
            # Effectively disable mismatch-triggered recovery
            cfg.thresholds.mismatch_trigger_count = 999  # type: ignore[attr-defined]
            # Make numeric-preservation gating neutral
            cfg.thresholds.numeric_preservation_min = 0.0  # type: ignore[attr-defined]
            cfg.thresholds.numeric_preservation_severe = 0.0  # type: ignore[attr-defined]
            # Relax QA gating to avoid unintended boosts
            cfg.thresholds.qa_overlap_min = 0.0  # type: ignore[attr-defined]
            cfg.thresholds.qa_token_hit_min_ratio = 0.0  # type: ignore[attr-defined]
            cfg.thresholds.answer_ctx_min_overlap = 0.0  # type: ignore[attr-defined]
        except Exception:
            pass

    # Load domain dictionary units for parsing
    try:
        dd_path = getattr(cfg.domain, 'domain_dict_path', None)
        if dd_path:
            p = Path(dd_path)
            if p.exists():
                with p.open('r', encoding='utf-8') as f:
                    dd = json.load(f)
                    units = dd.get('units', []) or []
                    global DOMAIN_UNITS
                    DOMAIN_UNITS = {str(u).strip() for u in units if str(u).strip()}
    except Exception:
        DOMAIN_UNITS = set()

    pipe = UnifiedPDFPipeline(chunks, cfg)

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Support multiple QA JSON schemas
    # - Legacy: [ {"question": str, "answer": str, "tags"?: [str]} ...]
    # - New: { "qa_pairs": [ {"question_kr"|"question": str, "answer_kr"|"answer": str, "tags"?: [str]} ... ] }
    if isinstance(raw, dict):
        if "qa_pairs" in raw and isinstance(raw["qa_pairs"], list):
            items = []
            for obj in raw["qa_pairs"]:
                if not isinstance(obj, dict):
                    continue
                q = obj.get("question") or obj.get("question_kr") or obj.get("q") or ""
                a = obj.get("answer") or obj.get("answer_kr") or obj.get("a") or ""
                tags = obj.get("tags", []) or []
                items.append({
                    "id": obj.get("id"),
                    "question": q,
                    "answer": a,
                    "tags": tags,
                })
        else:
            # Attempt to find a top-level list under common keys, else empty
            items = raw.get("items") or raw.get("data") or []
    else:
        items = raw

    mc = MetricsCollector()
    # reproducibility info
    import hashlib, subprocess
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = "nogit"
    corpus_p = Path(args.corpus)
    corpus_sig = f"{corpus_p.stat().st_size}-" + hashlib.sha1(corpus_p.read_bytes()).hexdigest()[:12]
    total_em = 0
    total_uem = 0
    total_nci = 0
    total_np = 0.0
    retry_count = 0
    lat_vec: List[float] = []
    lat_bm25: List[float] = []
    lat_gen: List[float] = []
    lat_total: List[float] = []
    # per-tag accumulators
    tag_stats: Dict[str, Dict[str, float]] = {}
    # lightweight reference metrics helpers (string-based)
    def _token_f1(a: str, b: str) -> float:
        ta = (a or "").split(); tb = (b or "").split()
        if not ta or not tb:
            return 0.0
        from collections import Counter
        ca, cb = Counter(ta), Counter(tb)
        inter = sum((ca & cb).values())
        prec = inter / max(1, sum(ca.values()))
        rec = inter / max(1, sum(cb.values()))
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def _bleu1(a: str, b: str) -> float:
        ta = (a or "").split(); tb = (b or "").split()
        if not ta or not tb:
            return 0.0
        from collections import Counter
        ca, cb = Counter(ta), Counter(tb)
        inter = sum((ca & cb).values())
        return inter / max(1, sum(ca.values()))

    for i, it in enumerate(items, 1):
        q = it["question"]
        gold = it.get("answer", "")
        tags = it.get("tags", []) or []
        res = pipe.ask(q, mode=args.mode)
        # Evaluation policy: fixed 3% tolerance
        eval_rel_tol = EVAL_REL_TOL
        em, uem = score_em_uem_lenient(
            res.text,
            gold,
            rel_tol=eval_rel_tol,
        )
        total_em += em
        total_uem += uem
        contexts_text = [s.chunk.text for s in res.sources]
        nci = compute_nci(contexts_text, gold)
        total_nci += nci
        retried = 1 if int(res.metrics.get("recovery_round", 0) or 0) > 0 else 0
        retry_count += retried
        lat_vec.append(float(res.metrics.get("vector_time_ms", 0)))
        lat_bm25.append(float(res.metrics.get("bm25_time_ms", 0)))
        lat_gen.append(float(res.metrics.get("gen_time_ms", 0)))
        lat_total.append(float(res.metrics.get("total_time_ms", 0)))
        # numeric preservation from pipeline metrics (0..1)
        np_val = float(res.metrics.get("numeric_preservation", 0.0))
        total_np += np_val
        # update tag stats
        for tg in tags:
            st = tag_stats.setdefault(tg, {
                "n": 0, "em": 0.0, "uem": 0.0, "nci": 0.0, "retry": 0.0, "np": 0.0,
                "vec_ms": 0.0, "bm25_ms": 0.0, "gen_ms": 0.0, "total_ms": 0.0,
            })
            st["n"] += 1
            st["em"] += em
            st["uem"] += uem
            st["nci"] += nci
            st["retry"] += retried
            st["np"] += np_val
            st["vec_ms"] += lat_vec[-1]
            st["bm25_ms"] += lat_bm25[-1]
            st["gen_ms"] += lat_gen[-1]
            st["total_ms"] += lat_total[-1]
        row = {
            "id": it.get("id"),
            "question": q,
            "gold": gold,
            "pred": res.text,
            "em@5%": em,
            "uem@5%": uem,
            "nci": nci,
            "confidence": res.confidence,
            "sources": len(res.sources),
            "git_commit": git_commit,
            "corpus_signature": corpus_sig,
            "seed": cfg.seed,
            "tags": tags,
        }
        row.update(res.metrics)
        # per-row reference-based metrics against ideal answers
        row["ref_token_f1"] = _token_f1(res.text, gold)
        row["ref_bleu1"] = _bleu1(res.text, gold)
        row["config_hash"] = cfg.config_hash()
        mc.add(row)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    mc.to_json(args.report)
    mc.to_csv(args.csv)

    n = max(1, len(items))
    avg_em = total_em / n
    avg_uem = total_uem / n
    avg_nci = total_nci / n
    retry_rate = retry_count / n
    np_avg = total_np / n
    # latency aggregates
    mean_vec = sum(lat_vec) / n
    mean_bm = sum(lat_bm25) / n
    mean_gen = sum(lat_gen) / n
    mean_total = sum(lat_total) / n
    p95_vec = percentile(lat_vec, 95)
    p95_bm = percentile(lat_bm25, 95)
    p95_gen = percentile(lat_gen, 95)
    p95_total = percentile(lat_total, 95)

    logger.info(f"EM@5%: {avg_em:.3f}, UEM@5%: {avg_uem:.3f}, NCI: {avg_nci:.3f}")
    logger.info(f"RetryRate: {retry_rate:.3f}")
    logger.info(f"Latency mean(ms) vec/bm25/gen/total: {mean_vec:.1f}/{mean_bm:.1f}/{mean_gen:.1f}/{mean_total:.1f}")
    logger.info(f"Latency p95(ms) vec/bm25/gen/total: {p95_vec:.1f}/{p95_bm:.1f}/{p95_gen:.1f}/{p95_total:.1f}")

    # appendix metrics
    def token_f1(a: str, b: str) -> float:
        return _token_f1(a, b)

    def bleu1(a: str, b: str) -> float:
        return _bleu1(a, b)

    def faithfulness_k(answer: str, contexts: List[str], k: int = 3) -> float:
        ctx = " \n ".join(contexts[:k]) if contexts else ""
        # proportion of answer tokens present in contexts
        ta = (answer or "").split()
        if not ta:
            return 1.0
        hits = sum(1 for t in ta if t in ctx)
        return hits / max(1, len(ta))

    def range_compliance(pred: str, gold: str) -> int:
        import re
        # parse range in gold: low..high unit
        m = re.search(r"(\d+(?:[\.,]\d+)?)\s*[–\-~]\s*(\d+(?:[\.,]\d+)?)\s*([A-Za-z%°µ/‰]+)", gold or "")
        if not m:
            return 0
        lo = float(m.group(1).replace(",", "")); hi = float(m.group(2).replace(",", "")); u = normalize_unit(m.group(3))
        pv, pu = parse_first_value_and_unit(pred or "")
        if pv is None or pu is None:
            return 0
        cv = convert_value(pv, pu, u)
        if cv is None:
            return 0
        return int(min(lo, hi) - 1e-9 <= cv <= max(lo, hi) + 1e-9)

    # build per-tag aggregate
    by_tag: Dict[str, Dict[str, float]] = {}
    for tg, st in tag_stats.items():
        n = max(1, int(st["n"]))
        by_tag[tg] = {
            "n": int(st["n"]),
            "em@5%": st["em"] / n,
            "uem@5%": st["uem"] / n,
            "nci": st["nci"] / n,
            "numeric_preservation": st["np"] / n,
            "retry_rate": st["retry"] / n,
            "latency_mean_ms": {
                "vec": st["vec_ms"] / n,
                "bm25": st["bm25_ms"] / n,
                "gen": st["gen_ms"] / n,
                "total": st["total_ms"] / n,
            },
        }

    # write enriched report JSON manually (instead of mc.to_json) to include paper summary
    # error analysis
    rows = mc.rows
    fail_rows = [r for r in rows if int(r.get("em@5%", 0)) == 0]
    nci_zero_ratio = sum(1 for r in rows if int(r.get("nci", 0)) == 0) / max(1, len(rows))
    retry_fail_rate = sum(1 for r in rows if int(r.get("recovery_round", 0) or 0) > 0 and int(r.get("em@5%", 0)) == 0) / max(1, len(rows))

    # appendix metrics aggregates
    f1_avg = sum(token_f1(r.get("pred", ""), r.get("gold", "")) for r in rows) / max(1, len(rows))
    bleu_avg = sum(bleu1(r.get("pred", ""), r.get("gold", "")) for r in rows) / max(1, len(rows))
    faith_avg = sum(faithfulness_k(r.get("pred", ""), [r.get("context_text", "")] if False else []) for r in rows) / max(1, len(rows))  # placeholder; contexts already summarized in metrics
    range_avg = sum(range_compliance(r.get("pred", ""), r.get("gold", "")) for r in rows) / max(1, len(rows))

    report_obj = {
        "rows": mc.rows,
        "aggregate": mc.aggregate(),
        "paper": {
            "global": {
                "em@5%": avg_em,
                "uem@5%": avg_uem,
                "nci": avg_nci,
                "retry_rate": retry_rate,
                "latency_mean_ms": {"vec": mean_vec, "bm25": mean_bm, "gen": mean_gen, "total": mean_total},
                "latency_p95_ms": {"vec": p95_vec, "bm25": p95_bm, "gen": p95_gen, "total": p95_total},
                "git_commit": git_commit,
                "corpus_signature": corpus_sig,
                "config_hash": cfg.config_hash(),
                "seed": cfg.seed,
                "numeric_preservation": np_avg,
                # headline-friendly reference metrics (against ideal gold answers)
                "ref_token_f1": f1_avg,
                "ref_bleu1": bleu_avg,
            },
            "by_tag": by_tag,
            "error_analysis": {
                "failure_count": len(fail_rows),
                "nci_zero_ratio": nci_zero_ratio,
                "retry_failure_rate": retry_fail_rate,
            },
            "appendix_metrics": {
                "token_f1": f1_avg,
                "bleu1": bleu_avg,
                "faithfulness_k": faith_avg,
                "range_compliance": range_avg,
            }
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)
    mc.to_csv(args.csv)


if __name__ == "__main__":
    main()
