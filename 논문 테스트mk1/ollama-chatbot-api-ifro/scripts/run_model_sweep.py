import argparse
import json
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def default_speed_config() -> dict:
    return {
        "flags": {"mode": "speed", "use_cross_reranker": False, "rerank_top_n": 0, "enable_parallel_search": True, "enable_retrieval_cache": False},
        "rrf": {"vector_weight": 0.0, "bm25_weight": 1.0, "base_rrf_k": 60},
        "context": {"k_default": 4, "k_min": 4, "k_max": 4, "allow_neighbor_from_adjacent_page": False},
        "deduplication": {"enable_semantic_dedup": False, "jaccard_threshold": 0.9},
        "thresholds": {"rerank_threshold": 1.0, "mismatch_trigger_count": 999, "qa_overlap_min": 0.0, "answer_ctx_min_overlap": 0.0},
        "llm_retries": 0,
    }


def main():
    ap = argparse.ArgumentParser(description="Sweep multiple LLM models under a fixed speed config and compare results")
    ap.add_argument("--input", required=True, help="Path to QA JSON")
    ap.add_argument("--corpus", required=True, help="Path to corpus JSONL")
    ap.add_argument("--models", required=True, help="Comma-separated model names (e.g., 'llama3.1:8b,...')")
    ap.add_argument("--outdir", default="out/model_sweep", help="Directory to store per-model reports")
    ap.add_argument("--base-config", default=None, help="Optional JSON/YAML file to override default speed config")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load base config
    cfg = default_speed_config()
    if args.base_config:
        try:
            import yaml  # type: ignore
            with open(args.base_config, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
        except Exception:
            with open(args.base_config, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)

        def _apply(obj, kv):
            for k, v in (kv or {}).items():
                if isinstance(v, dict) and isinstance(obj.get(k), dict):
                    _apply(obj[k], v)
                else:
                    obj[k] = v

        _apply(cfg, user_cfg)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    summary = {}
    for m in models:
        safe = safe_name(m)
        cfg_path = outdir / f"config_speed_{safe}.json"
        cfg_model = dict(cfg)
        # shallow copy; set model_name at top-level
        cfg_model["model_name"] = m
        save_json(cfg_path, cfg_model)

        rep_json = outdir / f"report_{safe}.json"
        rep_csv = outdir / f"report_{safe}.csv"

        cmd = [
            "python", str(HERE / "run_qa_benchmark.py"),
            "--input", args.input,
            "--corpus", args.corpus,
            "--report", str(rep_json),
            "--csv", str(rep_csv),
            "--config", str(cfg_path),
        ]
        run(cmd)

        try:
            obj = load_json(rep_json)
            g = (obj.get("paper", {}) or {}).get("global", {})
            summary[m] = {
                "em@5%": float(g.get("em@5%", 0.0)),
                "uem@5%": float(g.get("uem@5%", 0.0)),
                "nci": float(g.get("nci", 0.0)),
                "latency_ms": g.get("latency_mean_ms", {}),
                "report": str(rep_json),
                "csv": str(rep_csv),
            }
        except Exception:
            summary[m] = {"error": f"failed to parse {rep_json}"}

    save_json(outdir / "summary.json", {"models": summary, "corpus": args.corpus, "input": args.input})
    print(json.dumps({"models": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

