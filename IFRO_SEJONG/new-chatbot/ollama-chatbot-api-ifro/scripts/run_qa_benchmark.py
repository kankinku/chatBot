import argparse
import json
import sys
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

# Ensure 'src' is on sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from unifiedpdf.config import PipelineConfig
from unifiedpdf.facade import UnifiedPDFPipeline
from unifiedpdf.metrics import MetricsCollector
from unifiedpdf.types import Chunk
from unifiedpdf.measurements import verify_answer_numeric, extract_measurements, units_equivalent


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


def normalize_text(s: str) -> str:
    import re
    t = s.lower()
    t = t.replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def score_answer(pred: str, gold: str, tags: List[str]) -> float:
    import re
    p = normalize_text(pred)
    g = normalize_text(gold)
    if g in {"없음", "없다", "없습니다", "none", "no"}:
        # strict: only treat explicit negation as correct
        return 1.0 if p.startswith("문서에서 해당 정보를 확인할 수 없습니다") else 0.0
    # keyword match with numeric + unit weighting
    keywords = set(re.findall(r"[\w\-/%°℃]+", g))
    nums = set(re.findall(r"\d+(?:[\.,]\d+)?", g))
    units = set(re.findall(r"[a-z%°℃/]+", g))
    hit = 0.0
    total = 0.0
    # numeric
    if nums:
        total += 1.5
        hit += 1.5 if any(n in p for n in nums) else 0.0
    # units
    if units:
        total += 1.3
        uh = 0.0
        for u in units:
            if u in p:
                uh = 1.3
                break
        if uh == 0.0:
            # try unit synonym mapping
            for u in units:
                # reuse shared units_equivalent
                for v in ["mg/l", "ppm"]:
                    if v in p and units_equivalent(u, v):
                        uh = 1.3
                        break
        hit += uh
    # general keywords
    kw = {k for k in keywords if k not in nums and k not in units and len(k) >= 2}
    if kw:
        total += 1.0
        hit += 1.0 if any(k in p for k in kw) else 0.0
    return (hit / total) if total > 0 else 0.0


def main():
    logger.info("=== QA 벤치마크 시작 ===")
    
    ap = argparse.ArgumentParser(description="Run QA benchmark for UnifiedPDFPipeline")
    ap.add_argument("--input", default="data/tests/qa.json", help="Path to input QA JSON")
    ap.add_argument("--corpus", default="data/corpus_v1.jsonl", help="Path to JSONL corpus")
    ap.add_argument("--mode", default="accuracy", choices=["accuracy", "speed"])
    ap.add_argument("--k", default="auto")
    ap.add_argument("--report", default="out/report.json")
    ap.add_argument("--csv", default="out/report.csv")
    ap.add_argument("--store-backend", default="auto", choices=["auto", "faiss", "hnsw"]) 
    ap.add_argument("--vector-store-dir", default="vector_store", help="Directory of prebuilt vector index")
    ap.add_argument("--use-cross-reranker", action="store_true")
    ap.add_argument("--rerank-top-n", type=int, default=50)
    ap.add_argument("--thr-base", type=float, default=None)
    ap.add_argument("--thr-numeric", type=float, default=None)
    ap.add_argument("--thr-long", type=float, default=None)
    args = ap.parse_args()

    logger.info(f"설정: 입력={args.input}, 코퍼스={args.corpus}, 모드={args.mode}")
    logger.info(f"벡터스토어: {args.vector_store_dir}, 백엔드={args.store_backend}")

    logger.info("코퍼스 로딩 중...")
    chunks = load_corpus(args.corpus)
    logger.info(f"코퍼스 로딩 완료: {len(chunks)}개 청크")
    
    logger.info("파이프라인 설정 중...")
    cfg = PipelineConfig()
    cfg.flags.store_backend = args.store_backend
    cfg.vector_store_dir = args.vector_store_dir
    cfg.flags.use_cross_reranker = args.use_cross_reranker
    cfg.flags.rerank_top_n = args.rerank_top_n
    if args.thr_base is not None:
        cfg.thresholds.confidence_threshold = args.thr_base
    if args.thr_numeric is not None:
        cfg.thresholds.confidence_threshold_numeric = args.thr_numeric
    if args.thr_long is not None:
        cfg.thresholds.confidence_threshold_long = args.thr_long
    
    logger.info("파이프라인 초기화 중...")
    pipe = UnifiedPDFPipeline(chunks, cfg)
    logger.info("파이프라인 초기화 완료")

    logger.info("QA 데이터 로딩 중...")
    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)
    logger.info(f"QA 데이터 로딩 완료: {len(items)}개 질문")

    logger.info("=== 질문 처리 시작 ===")
    mc = MetricsCollector()
    for i, it in enumerate(items, 1):
        q = it["question"]
        gold = it.get("answer", "")
        tags = it.get("tags", [])
        
        logger.info(f"[{i}/{len(items)}] 질문 처리 중: {q[:50]}...")
        
        res = pipe.ask(q, mode=args.mode)
        s = score_answer(res.text, gold, tags)
        
        logger.info(f"[{i}/{len(items)}] 완료 - 점수: {s:.3f}, 신뢰도: {res.confidence:.3f}")
        
        row = {
            "id": it.get("id"),
            "question": q,
            "gold": gold,
            "pred": res.text,
            "score": s,
            "confidence": res.confidence,
            "sources": len(res.sources),
        }
        row.update(res.metrics)
        row["config_hash"] = cfg.config_hash()
        mc.add(row)

    logger.info("=== 결과 저장 중 ===")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    mc.to_json(args.report)
    mc.to_csv(args.csv)
    
    # 간단한 형태의 결과 파일도 생성
    simple_report_path = str(Path(args.report).parent / "report_simple.json")
    simple_results = []
    for row in mc.rows:
        simple_results.append({
            "번호": row["id"],
            "질문": row["question"],
            "정답": row["gold"],
            "답변": row["pred"]
        })
    
    with open(simple_report_path, "w", encoding="utf-8") as f:
        json.dump(simple_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"상세 보고서 저장: {args.report}")
    logger.info(f"CSV 보고서 저장: {args.csv}")
    logger.info(f"간단 보고서 저장: {simple_report_path}")
    
    # 전체 성능 요약
    avg_score = sum(row["score"] for row in mc.rows) / len(mc.rows)
    avg_confidence = sum(row["confidence"] for row in mc.rows) / len(mc.rows)
    logger.info(f"=== 벤치마크 완료 ===")
    logger.info(f"평균 점수: {avg_score:.3f}")
    logger.info(f"평균 신뢰도: {avg_confidence:.3f}")
    logger.info(f"총 처리 시간: {sum(row['total_time_ms'] for row in mc.rows)/1000:.1f}초")


if __name__ == "__main__":
    main()
