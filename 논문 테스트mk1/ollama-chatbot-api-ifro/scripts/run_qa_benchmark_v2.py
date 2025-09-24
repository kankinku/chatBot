import argparse
import json
import sys
import logging
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

EVAL_REL_TOL: float = 0.03  # 3%


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
    return val, None


def score_numeric_match(pred: str, gold: str, rel_tol: float = EVAL_REL_TOL) -> int:
    gv, gu = parse_first_value_and_unit_lenient(gold)
    pv, pu = parse_first_value_and_unit_lenient(pred)
    if gv is None or pv is None:
        return 0
    if gu and pu:
        cv = convert_value(pv, pu, gu)
        if cv is None:
            return 0
        return int(abs(cv - gv) / max(1e-9, abs(gv)) <= rel_tol)
    return int(abs(pv - gv) / max(1e-9, abs(gv)) <= rel_tol)


def token_f1(a: str, b: str) -> float:
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


def normalize_text(s: str) -> str:
    import re
    t = (s or "").lower()
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"[\s\t\n]+", " ", t)
    t = t.replace(",", "")
    return t.strip()


def relaxed_em(gold: str, pred: str) -> int:
    g = normalize_text(gold)
    p = normalize_text(pred)
    return 1 if g and (g in p) else 0


def rouge_l_f1(a: str, b: str) -> float:
    ta = (a or "").split(); tb = (b or "").split()
    if not ta or not tb:
        return 0.0
    n, m = len(ta), len(tb)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ta[i-1] == tb[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[n][m]
    prec = lcs / max(1, m)
    rec = lcs / max(1, n)
    if prec+rec == 0:
        return 0.0
    return 2*prec*rec/(prec+rec)


def tolerant_load_items(path: str) -> List[dict]:
    txt = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(txt)
    except Exception:
        try:
            raw = json.loads("[" + txt.strip() + "]")
        except Exception:
            objs = []
            buf = ""; depth = 0
            for ch in txt:
                buf += ch
                if ch == '{': depth += 1
                elif ch == '}':
                    depth = max(0, depth-1)
                    if depth == 0:
                        try:
                            objs.append(json.loads(buf))
                        except Exception:
                            pass
                        buf = ""
            raw = objs
    if isinstance(raw, dict):
        src = raw.get("qa_pairs") or raw.get("items") or raw.get("data") or []
    else:
        src = raw if isinstance(raw, list) else []
    items: List[dict] = []
    for obj in src:
        if not isinstance(obj, dict):
            continue
        # Accept both English and Korean keys for compatibility
        q = (
            obj.get("question")
            or obj.get("question_kr")
            or obj.get("q")
            or obj.get("질문")
            or ""
        )
        a = (
            obj.get("answer")
            or obj.get("answer_kr")
            or obj.get("a")
            or obj.get("정답")
            or ""
        )
        tags = obj.get("tags", []) or []
        items.append({"id": obj.get("id"), "question": q, "answer": a, "tags": tags})
    return items


def main():
    logger.info("=== QA Benchmark (Composite Metrics) ===")

    ap = argparse.ArgumentParser(description="Run QA benchmark (Composite metrics)")
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
    # External config overrides
    if args.config:
        try:
            import yaml  # type: ignore
            overrides = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        except Exception:
            overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        def _apply(obj, kv):
            for k, v in (kv or {}).items():
                if hasattr(obj, k):
                    cur = getattr(obj, k)
                    if isinstance(v, dict) and not isinstance(cur, (str, int, float, bool)):
                        _apply(cur, v)
                    else:
                        setattr(obj, k, v)
        _apply(cfg, overrides)

    pipe = UnifiedPDFPipeline(chunks, cfg)
    items = tolerant_load_items(args.input)

    mc = MetricsCollector()
    # reproducibility
    import hashlib, subprocess
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = "nogit"
    corpus_p = Path(args.corpus)
    corpus_sig = f"{corpus_p.stat().st_size}-" + hashlib.sha1(corpus_p.read_bytes()).hexdigest()[:12]

    numeric_total = token_total = relax_total = sem_total = comp_total = 0.0
    retry_count = 0
    lat_vec: List[float] = []
    lat_bm25: List[float] = []
    lat_gen: List[float] = []
    lat_total: List[float] = []
    np_total = 0.0
    by_tag: Dict[str, Dict[str, float]] = {}

    weights = getattr(cfg, 'eval_weights', None) or {}
    w_num = float(weights.get('numeric', 0.5))
    w_tok = float(weights.get('token', 0.3))
    w_sem = float(weights.get('semantic', 0.2))

    for it in items:
        q = it.get("question", "")
        gold = it.get("answer", "")
        tags = it.get("tags", []) or []
        res = pipe.ask(q, mode=args.mode)

        nm = score_numeric_match(res.text, gold, rel_tol=EVAL_REL_TOL)
        tf1 = token_f1(res.text, gold)
        rlem = relaxed_em(gold, res.text)
        sem = rouge_l_f1(res.text, gold)
        comp = w_num*float(nm) + w_tok*tf1 + w_sem*sem

        numeric_total += float(nm)
        token_total += tf1
        relax_total += float(rlem)
        sem_total += sem
        comp_total += comp

        retried = 1 if int(res.metrics.get("recovery_round", 0) or 0) > 0 else 0
        retry_count += retried
        lat_vec.append(float(res.metrics.get("vector_time_ms", 0)))
        lat_bm25.append(float(res.metrics.get("bm25_time_ms", 0)))
        lat_gen.append(float(res.metrics.get("gen_time_ms", 0)))
        lat_total.append(float(res.metrics.get("total_time_ms", 0)))
        np_total += float(res.metrics.get("numeric_preservation", 0.0))

        for tg in tags:
            st = by_tag.setdefault(tg, {
                "n": 0, "numeric_match": 0.0, "token_f1": 0.0, "relaxed_em": 0.0,
                "semantic_sim": 0.0, "composite": 0.0, "np": 0.0,
                "vec_ms": 0.0, "bm25_ms": 0.0, "gen_ms": 0.0, "total_ms": 0.0,
            })
            st["n"] += 1
            st["numeric_match"] += float(nm)
            st["token_f1"] += tf1
            st["relaxed_em"] += float(rlem)
            st["semantic_sim"] += sem
            st["composite"] += comp
            st["np"] += float(res.metrics.get("numeric_preservation", 0.0))
            st["vec_ms"] += lat_vec[-1]
            st["bm25_ms"] += lat_bm25[-1]
            st["gen_ms"] += lat_gen[-1]
            st["total_ms"] += lat_total[-1]

        row = {
            "id": it.get("id"),
            "question": q,
            "gold": gold,
            "pred": res.text,
            "numeric_match": int(nm),
            "token_f1": tf1,
            "relaxed_em": int(rlem),
            "semantic_sim": sem,
            "composite": comp,
            "np": float(res.metrics.get("numeric_preservation", 0.0)),
        }
        row.update(res.metrics)
        row["config_hash"] = cfg.config_hash()
        row["git_commit"] = git_commit
        row["corpus_signature"] = corpus_sig
        row["seed"] = cfg.seed
        row["tags"] = tags
        mc.add(row)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    mc.to_json(args.report)
    mc.to_csv(args.csv)

    n = max(1, len(items))
    avg_numeric = numeric_total / n
    avg_token = token_total / n
    avg_relaxed = relax_total / n
    avg_sem = sem_total / n
    avg_comp = comp_total / n
    np_avg = np_total / n
    retry_rate = retry_count / n

    mean_vec = sum(lat_vec) / n
    mean_bm = sum(lat_bm25) / n
    mean_gen = sum(lat_gen) / n
    mean_total = sum(lat_total) / n
    p95_vec = percentile(lat_vec, 95)
    p95_bm = percentile(lat_bm25, 95)
    p95_gen = percentile(lat_gen, 95)
    p95_total = percentile(lat_total, 95)

    report_obj = {
        "rows": mc.rows,
        "aggregate": mc.aggregate(),
        "paper": {
            "global": {
                "numeric_match": avg_numeric,
                "token_f1": avg_token,
                "relaxed_em": avg_relaxed,
                "semantic_sim": avg_sem,
                "composite": avg_comp,
                "numeric_preservation": np_avg,
                "retry_rate": retry_rate,
                "latency_mean_ms": {"vec": mean_vec, "bm25": mean_bm, "gen": mean_gen, "total": mean_total},
                "latency_p95_ms": {"vec": p95_vec, "bm25": p95_bm, "gen": p95_gen, "total": p95_total},
                "config_hash": cfg.config_hash(),
                "seed": cfg.seed,
            },
            "by_tag": {
                tg: {
                    "n": int(st["n"]),
                    "numeric_match": st["numeric_match"] / max(1, int(st["n"])),
                    "token_f1": st["token_f1"] / max(1, int(st["n"])),
                    "relaxed_em": st["relaxed_em"] / max(1, int(st["n"])),
                    "semantic_sim": st["semantic_sim"] / max(1, int(st["n"])),
                    "composite": st["composite"] / max(1, int(st["n"])),
                    "numeric_preservation": st["np"] / max(1, int(st["n"])),
                    "latency_mean_ms": {
                        "vec": st["vec_ms"] / max(1, int(st["n"])),
                        "bm25": st["bm25_ms"] / max(1, int(st["n"])),
                        "gen": st["gen_ms"] / max(1, int(st["n"])),
                        "total": st["total_ms"] / max(1, int(st["n"])),
                    },
                }
                for tg, st in by_tag.items()
            },
        },
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)
    mc.to_csv(args.csv)


if __name__ == "__main__":
    main()

