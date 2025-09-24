import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description="Grid search chunk parameters and report retrieval Hit@k and Split Error")
    ap.add_argument("--input", default=str(ROOT / "ollama-chatbot-api-ifro" / "data" / "tests" / "qa.json"))
    ap.add_argument("--extracted", default=str(ROOT / "ollama-chatbot-api-ifro" / "data" / "extracted_text.jsonl"))
    ap.add_argument("--chunk-size", default="600,800,1000")
    ap.add_argument("--chunk-overlap", default="100,200,300")
    ap.add_argument("--strategy", default="processor", choices=["processor", "sentence", "paragraph", "sliding"])
    ap.add_argument("--enable-numeric", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    sizes = [int(x) for x in str(args.chunk_size).split(",") if x]
    overlaps = [int(x) for x in str(args.chunk_overlap).split(",") if x]

    results = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for sz in sizes:
            for ov in overlaps:
                outp = td / f"corpus_{args.strategy}_s{sz}_o{ov}.jsonl"
                cmd = [
                    "python",
                    str(HERE / "build_corpus_from_text.py"),
                    "--input",
                    args.extracted,
                    "--out",
                    str(outp),
                    "--chunk-size",
                    str(sz),
                    "--chunk-overlap",
                    str(ov),
                    "--strategy",
                    args.strategy,
                ]
                if args.enable_numeric and args.strategy == "processor":
                    cmd.append("--enable-numeric-window")
                run(cmd)

                # Split error
                r1 = subprocess.check_output([
                    "python",
                    str(HERE / "measure_split_error.py"),
                    "--input",
                    args.extracted,
                    "--corpus",
                    str(outp),
                ])
                se = json.loads(r1.decode("utf-8", errors="ignore")).get("split_error", {})
                name = outp.stem
                se_rate = list(se.values())[0]["split_error_rate"] if se else 0.0

                # Hit@k
                r2 = subprocess.check_output([
                    "python",
                    str(HERE / "retrieval_hitk.py"),
                    "--input",
                    args.input,
                    "--corpus",
                    str(outp),
                    "--k",
                    str(args.k),
                ])
                hk = json.loads(r2.decode("utf-8", errors="ignore")).get("hitk", {})
                hitk = list(hk.values())[0]["hit@k"] if hk else 0.0

                results.append({
                    "strategy": args.strategy,
                    "chunk_size": sz,
                    "chunk_overlap": ov,
                    "numeric": bool(args.enable_numeric),
                    "split_error_rate": se_rate,
                    "hit@k": hitk,
                })

    # Pick best by low split_error then high hit@k
    best = None
    for r in results:
        key = (-(1.0 - r["split_error_rate"]), -r["hit@k"])  # minimize split error, maximize hit@k
        if best is None or key < (-(1.0 - best["split_error_rate"]), -best["hit@k"]):
            best = r

    print(json.dumps({"results": results, "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

