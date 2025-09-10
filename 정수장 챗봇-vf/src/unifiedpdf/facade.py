from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .analyzer import analyze_question
from .config import PipelineConfig
from .context import add_neighbors, choose_k
from .filtering import calibrate_and_filter
from .guardrail import guard_check
from .llm import generate_answer
from .merger import merge_then_dedup
from .retriever import HybridRetriever
from .reranker import rerank_spans
from .measurements import build_context_measure_map, verify_answer_numeric
from .types import Answer, Chunk, RetrievedSpan
from .utils import now_ms


class UnifiedPDFPipeline:
    def __init__(self, chunks: List[Chunk], cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.retriever = HybridRetriever(chunks, cfg=self.cfg)

    def ask(self, question: str, mode: str | None = None) -> Answer:
        cfg = self.cfg
        if mode:
            cfg.flags.mode = mode
        t0 = now_ms()
        metrics: Dict[str, float | int | str] = {}

        # Analyze
        analysis = analyze_question(question, cfg)

        # Retrieve hybrid
        vbm_start = now_ms()
        spans, timings = self.retriever.retrieve(
            question,
            topk_each=50,
            rrf_k=cfg.rrf.base_rrf_k,
            rrf_weights=(analysis.rrf_vector_weight, analysis.rrf_bm25_weight),
        )
        metrics.update(timings)
        metrics["merged"] = len(spans)

        # Merge + dedup
        spans = merge_then_dedup(spans)
        metrics["deduped"] = len(spans)

        # Filter calibration
        # Threshold override by type/length
        thr_eff = cfg.thresholds.confidence_threshold
        if analysis.qtype == "numeric":
            thr_eff = min(thr_eff, cfg.thresholds.confidence_threshold_numeric)
        elif analysis.length >= 20:
            thr_eff = min(thr_eff, cfg.thresholds.confidence_threshold_long)

        spans, fstats = calibrate_and_filter(spans, cfg, threshold_override=thr_eff)
        metrics.update(fstats)

        # Optional reranker (accuracy mode only) before selecting k
        if cfg.flags.mode == "accuracy":
            use_cross = cfg.flags.use_cross_reranker
            spans, rerank_ms = rerank_spans(question, spans[: cfg.flags.rerank_top_n], use_cross)
            metrics["rerank_time_ms"] = rerank_ms

        # Build context with dynamic k + neighbor + final dedup
        k = choose_k(analysis, cfg)
        contexts = spans[:k]
        context_filled = 1 if len(contexts) > 0 else 0
        metrics["k_total"] = k
        metrics["context_filled"] = context_filled
        nb_added = 0
        if contexts:
            before = len(contexts)
            contexts = add_neighbors(contexts, cfg)
            nb_added = max(0, len(contexts) - before)
        metrics["neighbor_added"] = 1 if nb_added > 0 else 0

        # Guardrail
        guard = guard_check(question, contexts, cfg)
        metrics.update(guard)

        # If hard blocked or no context, attempt fallback path: relax threshold once
        fallback_used = "none"
        if (guard["hard_blocked"] == 1) or not contexts:
            # Relax threshold by 20% and re-run filter quickly
            orig_th = cfg.thresholds.confidence_threshold
            cfg.thresholds.confidence_threshold = max(0.2, orig_th * 0.8)
            spans2, _ = calibrate_and_filter(spans, cfg)
            cfg.thresholds.confidence_threshold = orig_th
            if spans2:
                contexts = spans2[:k]
                fallback_used = "low_conf_retry"
            elif spans:
                # Single top span fallback
                contexts = spans[:1]
                fallback_used = "single_span"
        metrics["fallback_used"] = fallback_used

        # Generate
        gen_start = now_ms()
        answer_text = generate_answer(question, contexts, cfg, qtype=analysis.qtype)
        metrics["gen_time_ms"] = now_ms() - gen_start
        metrics["model_name"] = cfg.model_name

        # Retry once with shorter context if empty
        if not answer_text and contexts:
            answer_text = generate_answer(question, contexts[:1], cfg, qtype=analysis.qtype)

        # If model produced nothing and we have context, do extractive fallback
        if not answer_text and contexts:
            answer_text = contexts[0].chunk.text.strip()[:500]
        if not answer_text:
            answer_text = "문서에서 해당 정보를 확인할 수 없습니다."
            metrics["no_answer"] = 1
        else:
            metrics["no_answer"] = 0

        # Numeric/unit verification against context with tolerance
        ctx_map = build_context_measure_map([s.chunk.text for s in contexts])
        metrics["numeric_preservation"] = verify_answer_numeric(answer_text, ctx_map, tol_ratio=0.05)

        metrics["total_time_ms"] = now_ms() - t0
        metrics["config_hash"] = self.cfg.config_hash()

        # Confidence heuristic: combine top-k calibrated conf and guard overlap
        if contexts:
            vals = [c.calibrated_conf for c in contexts[:3] if c.calibrated_conf is not None]
            ctx_conf = sum(vals) / len(vals) if vals else self.cfg.thresholds.base_no_answer_confidence
        else:
            ctx_conf = self.cfg.thresholds.base_no_answer_confidence
        guard_overlap = float(metrics.get("overlap_ratio", 0.0))
        confidence = max(0.0, min(1.0, 0.7 * ctx_conf + 0.3 * guard_overlap))

        return Answer(
            text=answer_text,
            confidence=confidence,
            sources=contexts,
            metrics=metrics,
            fallback_used=fallback_used,
        )


def _numeric_preservation_score(answer: str, context: str) -> float:
    import re

    nums_ans = re.findall(r"\d+(?:[\.,]\d+)?", answer)
    nums_ctx = set(re.findall(r"\d+(?:[\.,]\d+)?", context))
    if not nums_ans:
        return 1.0  # nothing numeric to preserve
    hits = sum(1 for n in nums_ans if n in nums_ctx)
    return hits / len(nums_ans)
