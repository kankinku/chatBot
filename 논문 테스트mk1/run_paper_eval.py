#!/usr/bin/env python3
"""
End-to-end paper benchmark runner (single entry).

Steps:
 1) Extract PDF text once to cache (skip if exists unless --force-extract)
 2) Build corpora from cached text:
       - A1: Fixed-size (baseline)
       - A2: Numeric-aware (no aux)
       - A3: Numeric-aware (+aux)
 3) Run benchmarks on each corpus (EM/UEM/NCI/Latency + reproducibility meta)
 4) Print concise summary table and write JSON/CSV under out/

Usage:
  python run_paper_eval.py --pdf_dir data/pdfs --qa data/tests/qa.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
API_DIR = ROOT / "ollama-chatbot-api-ifro"
SCRIPTS = API_DIR / "scripts"


def run(cmd: list[str], cwd: Path | None = None, step_name: str = "") -> None:
    start_time = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if step_name:
        print(f"\n[{timestamp}] 🔄 {step_name} 시작...")
    print(f"$ {' '.join(cmd)}")
    
    # Ensure Ollama host resolves on Windows/local by default
    env = os.environ.copy()
    env.setdefault('OLLAMA_HOST', '127.0.0.1')
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    
    elapsed = time.time() - start_time
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if r.returncode != 0:
        print(f"[{timestamp}] ❌ {step_name} 실패 (소요시간: {elapsed:.1f}초)")
        raise SystemExit(f"Command failed with code {r.returncode}: {' '.join(cmd)}")
    else:
        print(f"[{timestamp}] ✅ {step_name} 완료 (소요시간: {elapsed:.1f}초)")


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0


def summarize_report(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    rows = obj.get("rows", [])
    paper = obj.get("paper", {})
    if paper and paper.get("global"):
        g = paper["global"]
        return {
            "EM@5%": float(g.get("em@5%", 0)),
            "UEM@5%": float(g.get("uem@5%", 0)),
            "NCI": float(g.get("nci", 0)),
            "RetryRate": float(g.get("retry_rate", 0)),
            "Latency_ms": g.get("latency_mean_ms", {}),
            "ByTag": paper.get("by_tag", {}),
        }
    # Fallback to simple means
    em = mean([r.get("em@5%", 0) for r in rows])
    uem = mean([r.get("uem@5%", 0) for r in rows])
    nci = mean([r.get("nci", 0) for r in rows])
    mean_vec = mean([r.get("vector_time_ms", 0) for r in rows])
    mean_bm = mean([r.get("bm25_time_ms", 0) for r in rows])
    mean_gen = mean([r.get("gen_time_ms", 0) for r in rows])
    mean_total = mean([r.get("total_time_ms", 0) for r in rows])
    retry_rate = mean([1.0 if (r.get("recovery_round", 0) or 0) > 0 else 0.0 for r in rows])
    return {
        "EM@5%": em,
        "UEM@5%": uem,
        "NCI": nci,
        "RetryRate": retry_rate,
        "Latency_ms": {"vec": mean_vec, "bm25": mean_bm, "gen": mean_gen, "total": mean_total},
        "ByTag": {},
    }


def main():
    overall_start = time.time()
    print(f"\n🚀 논문 평가 벤치마크 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    ap = argparse.ArgumentParser(description="Paper benchmark one-click runner")
    ap.add_argument("--pdf_dir", default="ollama-chatbot-api-ifro/data/pdfs")
    ap.add_argument("--qa", default="ollama-chatbot-api-ifro/data/tests/qa.json")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--aux-per-doc", type=int, default=50)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    pdf_dir = ROOT / args.pdf_dir if not Path(args.pdf_dir).is_absolute() else Path(args.pdf_dir)
    qa_path = Path(args.qa)
    out_dir = ROOT / args.outdir
    ensure_dirs(out_dir)
    
    print(f"📁 PDF 디렉토리: {pdf_dir}")
    print(f"📋 QA 테스트 파일: {qa_path}")
    print(f"📤 출력 디렉토리: {out_dir}")
    print(f"🔧 청크 크기: {args.chunk_size}, 오버랩: {args.chunk_overlap}")
    print(f"🔄 강제 추출: {'예' if args.force_extract else '아니오'}")

    # 1) Extract text once
    print(f"\n📝 1단계: PDF 텍스트 추출")
    extracted = API_DIR / "data" / "extracted_text.jsonl"
    # Treat a zero-byte cache as invalid and re-extract
    need_extract = (
        args.force_extract
        or (not extracted.exists())
        or (extracted.exists() and extracted.stat().st_size == 0)
    )
    if need_extract:
        ensure_dirs(extracted.parent)
        pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        print(f"   📄 처리할 PDF 파일 수: {len(pdf_files)}")
        if pdf_files:
            print(f"   📄 파일 목록: {[f.name for f in pdf_files[:3]]}{'...' if len(pdf_files) > 3 else ''}")
        
        cmd = [sys.executable, str(SCRIPTS / "extract_text_from_pdfs.py"),
               "--pdf_dir", str(pdf_dir), "--out", str(extracted), "--ocr", "off"]
        run(cmd, step_name="PDF 텍스트 추출")
    else:
        print(f"   ⏭️  텍스트 추출 건너뜀 (이미 존재): {extracted}")

    # 2) Build corpora from cached text
    print(f"\n📚 2단계: 코퍼스 구축")
    corpus_A1 = API_DIR / "data" / "corpus_A1.jsonl"
    corpus_A2 = API_DIR / "data" / "corpus_A2.jsonl"
    corpus_A3 = API_DIR / "data" / "corpus_A3.jsonl"
    corpus_A4 = API_DIR / "data" / "corpus_A4.jsonl"

    # A1: fixed-size baseline
    print(f"   📖 A1 코퍼스 구축 (고정 크기 기준)")
    cmd = [sys.executable, str(SCRIPTS / "build_corpus_from_text.py"),
           "--input", str(extracted), "--out", str(corpus_A1),
           "--chunk-size", str(args.chunk_size), "--chunk-overlap", str(args.chunk_overlap),
           "--disable-numeric-window"]
    run(cmd, step_name="A1 코퍼스 구축")

    # A2: numeric-aware (no aux)
    print(f"   📖 A2 코퍼스 구축 (숫자 인식, 보조 없음)")
    cmd = [sys.executable, str(SCRIPTS / "build_corpus_from_text.py"),
           "--input", str(extracted), "--out", str(corpus_A2),
           "--chunk-size", str(args.chunk_size), "--chunk-overlap", str(args.chunk_overlap),
           "--enable-numeric-window", "--aux-per-doc", "0"]
    run(cmd, step_name="A2 코퍼스 구축")

    # A3: numeric-aware (+aux)
    print(f"   📖 A3 코퍼스 구축 (숫자 인식, 보조 {args.aux_per_doc}개)")
    cmd = [sys.executable, str(SCRIPTS / "build_corpus_from_text.py"),
           "--input", str(extracted), "--out", str(corpus_A3),
           "--chunk-size", str(args.chunk_size), "--chunk-overlap", str(args.chunk_overlap),
           "--enable-numeric-window", "--aux-per-doc", str(args.aux_per_doc)]
    run(cmd, step_name="A3 코퍼스 구축")

    # A4: numeric-aware (+basic OCR correction)
    print(f"   📖 A4 코퍼스 구축 (숫자 인식, OCR 보정)")
    cmd = [sys.executable, str(SCRIPTS / "build_corpus_from_text.py"),
           "--input", str(extracted), "--out", str(corpus_A4),
           "--chunk-size", str(args.chunk_size), "--chunk-overlap", str(args.chunk_overlap),
           "--enable-numeric-window", "--aux-per-doc", str(args.aux_per_doc), "--ocr-correct", "basic"]
    run(cmd, step_name="A4 코퍼스 구축")

    # 3) Run benchmarks
    print(f"\n🧪 3단계: QA 벤치마크 실행")
    rep_A1 = out_dir / "report_A1.json"
    rep_A2 = out_dir / "report_A2.json"
    rep_A3 = out_dir / "report_A3.json"
    rep_A4 = out_dir / "report_A4.json"
    rep_A5 = out_dir / "report_A5.json"
    csv_A1 = out_dir / "report_A1.csv"
    csv_A2 = out_dir / "report_A2.csv"
    csv_A3 = out_dir / "report_A3.csv"
    csv_A4 = out_dir / "report_A4.csv"
    csv_A5 = out_dir / "report_A5.csv"

    # QA 테스트 정보 표시
    if qa_path.exists():
        with qa_path.open("r", encoding="utf-8") as f:
            qa_data = json.load(f)
            qa_count = len(qa_data) if isinstance(qa_data, list) else 1
        print(f"   📋 QA 테스트 질문 수: {qa_count}")

    benchmarks = [
        (corpus_A1, rep_A1, csv_A1, "A1 (고정 크기 기준)"),
        (corpus_A2, rep_A2, csv_A2, "A2 (숫자 인식)"),
        (corpus_A3, rep_A3, csv_A3, "A3 (숫자 인식 + 보조)"),
        (corpus_A4, rep_A4, csv_A4, "A4 (숫자 인식 + OCR 보정)"),
    ]
    
    for i, (corpus, rep, csv, name) in enumerate(benchmarks, 1):
        print(f"   🧪 {name} 벤치마크 실행 ({i}/{len(benchmarks)})")
        cmd = [sys.executable, str(SCRIPTS / "run_qa_benchmark.py"),
               "--input", str(qa_path), "--corpus", str(corpus),
               "--report", str(rep), "--csv", str(csv)]
        run(cmd, step_name=f"{name} 벤치마크")

    # A5: numeric-aware + verification loop (config override)
    print(f"   🧪 A5 (숫자 인식 + 검증 루프) 벤치마크 실행")
    cfg_A5 = out_dir / "config_A5.json"
    cfg_A5.write_text(json.dumps({
        "thresholds": {
            "numeric_preservation_min": 0.9,
            "mismatch_trigger_count": 1
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   ⚙️  A5 설정 파일 생성: {cfg_A5}")
    cmd = [sys.executable, str(SCRIPTS / "run_qa_benchmark.py"),
           "--input", str(qa_path), "--corpus", str(corpus_A2),
           "--report", str(rep_A5), "--csv", str(csv_A5), "--config", str(cfg_A5)]
    run(cmd, step_name="A5 벤치마크 (검증 루프)")

    # 4) Summarize
    print(f"\n📊 4단계: 결과 요약 및 분석")
    print(f"   📈 각 벤치마크 결과 분석 중...")
    sum_A1 = summarize_report(rep_A1)
    sum_A2 = summarize_report(rep_A2)
    sum_A3 = summarize_report(rep_A3)
    sum_A4 = summarize_report(rep_A4)
    sum_A5 = summarize_report(rep_A5)

    def fmt(s):
        return f"{s:.3f}"

    print("\n=== Summary (EM/UEM/NCI, Retry) ===")
    print(f"A1 Fixed       EM {fmt(sum_A1['EM@5%'])}  UEM {fmt(sum_A1['UEM@5%'])}  NCI {fmt(sum_A1['NCI'])}  Retry {fmt(sum_A1.get('RetryRate', 0.0))}")
    print(f"A2 Numeric     EM {fmt(sum_A2['EM@5%'])}  UEM {fmt(sum_A2['UEM@5%'])}  NCI {fmt(sum_A2['NCI'])}  Retry {fmt(sum_A2.get('RetryRate', 0.0))}")
    print(f"A3 Numeric+Aux EM {fmt(sum_A3['EM@5%'])}  UEM {fmt(sum_A3['UEM@5%'])}  NCI {fmt(sum_A3['NCI'])}  Retry {fmt(sum_A3.get('RetryRate', 0.0))}")
    print(f"A4 +OCRCorr    EM {fmt(sum_A4['EM@5%'])}  UEM {fmt(sum_A4['UEM@5%'])}  NCI {fmt(sum_A4['NCI'])}  Retry {fmt(sum_A4.get('RetryRate', 0.0))}")
    print(f"A5 +VerifyLoop EM {fmt(sum_A5['EM@5%'])}  UEM {fmt(sum_A5['UEM@5%'])}  NCI {fmt(sum_A5['NCI'])}  Retry {fmt(sum_A5.get('RetryRate', 0.0))}")

    def delta(a, b):
        return f"{(b - a):+.3f}"

    print("\n=== Improvements vs A1 ===")
    print(f"A2 ΔEM {delta(sum_A1['EM@5%'], sum_A2['EM@5%'])}  ΔUEM {delta(sum_A1['UEM@5%'], sum_A2['UEM@5%'])}  ΔNCI {delta(sum_A1['NCI'], sum_A2['NCI'])}  ΔRetry {delta(sum_A1.get('RetryRate',0.0), sum_A2.get('RetryRate',0.0))}")
    print(f"A3 ΔEM {delta(sum_A1['EM@5%'], sum_A3['EM@5%'])}  ΔUEM {delta(sum_A1['UEM@5%'], sum_A3['UEM@5%'])}  ΔNCI {delta(sum_A1['NCI'], sum_A3['NCI'])}  ΔRetry {delta(sum_A1.get('RetryRate',0.0), sum_A3.get('RetryRate',0.0))}")
    print(f"A4 ΔEM {delta(sum_A1['EM@5%'], sum_A4['EM@5%'])}  ΔUEM {delta(sum_A1['UEM@5%'], sum_A4['UEM@5%'])}  ΔNCI {delta(sum_A1['NCI'], sum_A4['NCI'])}  ΔRetry {delta(sum_A1.get('RetryRate',0.0), sum_A4.get('RetryRate',0.0))}")
    print(f"A5 ΔEM {delta(sum_A1['EM@5%'], sum_A5['EM@5%'])}  ΔUEM {delta(sum_A1['UEM@5%'], sum_A5['UEM@5%'])}  ΔNCI {delta(sum_A1['NCI'], sum_A5['NCI'])}  ΔRetry {delta(sum_A1.get('RetryRate',0.0), sum_A5.get('RetryRate',0.0))}")

    # Save combined summary
    combined = {"A1": sum_A1, "A2": sum_A2, "A3": sum_A3, "A4": sum_A4, "A5": sum_A5,
                "deltas": {"A2_vs_A1": {"EM": sum_A2['EM@5%']-sum_A1['EM@5%'], "UEM": sum_A2['UEM@5%']-sum_A1['UEM@5%'], "NCI": sum_A2['NCI']-sum_A1['NCI'], "Retry": sum_A2.get('RetryRate',0.0)-sum_A1.get('RetryRate',0.0)},
                           "A3_vs_A1": {"EM": sum_A3['EM@5%']-sum_A1['EM@5%'], "UEM": sum_A3['UEM@5%']-sum_A1['UEM@5%'], "NCI": sum_A3['NCI']-sum_A1['NCI'], "Retry": sum_A3.get('RetryRate',0.0)-sum_A1.get('RetryRate',0.0)},
                           "A4_vs_A1": {"EM": sum_A4['EM@5%']-sum_A1['EM@5%'], "UEM": sum_A4['UEM@5%']-sum_A1['UEM@5%'], "NCI": sum_A4['NCI']-sum_A1['NCI'], "Retry": sum_A4.get('RetryRate',0.0)-sum_A1.get('RetryRate',0.0)},
                           "A5_vs_A1": {"EM": sum_A5['EM@5%']-sum_A1['EM@5%'], "UEM": sum_A5['UEM@5%']-sum_A1['UEM@5%'], "NCI": sum_A5['NCI']-sum_A1['NCI'], "Retry": sum_A5.get('RetryRate',0.0)-sum_A1.get('RetryRate',0.0)}}}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    # 전체 완료 메시지
    total_elapsed = time.time() - overall_start
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n🎉 논문 평가 벤치마크 완료!")
    print(f"⏱️  총 소요시간: {total_elapsed/60:.1f}분")
    print(f"📁 결과 파일 위치: {out_dir}")
    print(f"🕐 완료 시간: {timestamp}")
    print("=" * 60)


if __name__ == "__main__":
    main()
