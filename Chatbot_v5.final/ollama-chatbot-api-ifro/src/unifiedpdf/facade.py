from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .analyzer import analyze_question
from .config import PipelineConfig
from .context import add_neighbors, choose_k
from .filtering import calibrate_and_filter, annotate_and_pre_filter, apply_rerank_threshold
from .guardrail import guard_check
from .llm import generate_answer
from .merger import merge_then_dedup
from .retriever import HybridRetriever
from .reranker import rerank_spans
from .measurements import build_context_measure_map, verify_answer_numeric
from .types import Answer, Chunk, RetrievedSpan
from .utils import now_ms, overlap_ratio, key_tokens, contains_any_token


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
        spans = merge_then_dedup(
            spans, 
            jaccard_thr=cfg.deduplication.jaccard_threshold,
            semantic_thr=cfg.deduplication.semantic_threshold if cfg.deduplication.enable_semantic_dedup else 0.0
        )
        metrics["deduped"] = len(spans)

        # Context quality pre-filter: overlap + keyword hit
        spans, qstats = annotate_and_pre_filter(question, spans, cfg)
        metrics.update(qstats)
        metrics["quality_kept"] = len(spans)

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
            # Apply stricter rerank threshold
            spans, rst = apply_rerank_threshold(spans, cfg)
            metrics.update(rst)

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
            # Relax threshold by 30% and re-run filter quickly
            orig_th = cfg.thresholds.confidence_threshold
            cfg.thresholds.confidence_threshold = max(0.15, orig_th * 0.7)
            spans2, _ = calibrate_and_filter(spans, cfg)
            cfg.thresholds.confidence_threshold = orig_th
            if spans2:
                contexts = spans2[:k]
                fallback_used = "low_conf_retry"
            elif spans:
                # Single top span fallback with more context
                contexts = spans[:min(2, len(spans))]
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

        # QA consistency: overlap with question and key-token hits
        q_tokens = key_tokens(question)
        qa_overlap = overlap_ratio(question, answer_text)
        qa_token_hit = contains_any_token(answer_text, q_tokens) if q_tokens else True
        metrics["qa_overlap"] = qa_overlap
        metrics["qa_token_match"] = 1 if qa_token_hit else 0
        metrics["qa_consistent"] = 1 if (qa_overlap >= 0.05 and qa_token_hit) else 0

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
        # penalize confidence if QA consistency is poor
        if metrics.get("qa_consistent", 1) == 0:
            confidence = max(0.0, confidence - 0.1)

        # Strong mismatch recovery loop: up to 2 attempts with stricter prompt and reduced context
        def _detect_mismatch() -> bool:
            # Do not recover for explicit "no answer" responses
            if metrics.get("no_answer", 0) == 1:
                return False
            # thresholds
            thr_qa = cfg.thresholds.qa_overlap_min
            thr_token = cfg.thresholds.qa_token_hit_min_ratio
            thr_ctx = cfg.thresholds.answer_ctx_min_overlap
            thr_num = cfg.thresholds.numeric_preservation_min
            thr_num_sev = cfg.thresholds.numeric_preservation_severe
            # signals
            viol = 0
            # qa overlap
            if metrics.get("qa_overlap", 0.0) < thr_qa:
                viol += 1
            # token coverage ratio
            from .utils import key_tokens
            toks = key_tokens(question)
            if toks:
                ans = (answer_text or "").lower()
                hits = sum(1 for t in toks if t in ans)
                ratio = hits / max(1, len(toks))
                metrics["qa_token_hit_ratio"] = ratio
                if ratio < thr_token:
                    viol += 1
            # answer-context alignment (max overlap against any context)
            from .utils import overlap_ratio as _ov
            if contexts:
                mx = max((_ov(answer_text, s.chunk.text) for s in contexts), default=0.0)
            else:
                mx = 0.0
            metrics["answer_ctx_overlap_max"] = mx
            if mx < thr_ctx:
                viol += 1
            # numeric preservation
            num_pres = float(metrics.get("numeric_preservation", 1.0))
            if num_pres < thr_num:
                viol += 1
            # severe condition: numeric very low and answer contains number
            import re as _re
            has_num = bool(_re.search(r"\d", answer_text or ""))
            if has_num and (num_pres < thr_num_sev):
                viol += 1  # severe -> count extra
            return viol >= int(getattr(cfg.thresholds, "mismatch_trigger_count", 2))

        recovery_attempts = 0
        if _detect_mismatch():
            recovery_attempts += 1
            # pick stricter contexts: top-1 by highest overlap with question or calibrated_conf
            def _score_span(sp):
                return sp.aux_scores.get("ovlp", 0.0) if sp.aux_scores else (sp.calibrated_conf or 0.0)
            strict_contexts = sorted(contexts, key=_score_span, reverse=True)[:1] or contexts[:1]
            answer_text = generate_answer(question, strict_contexts, cfg, qtype=analysis.qtype, recovery=True)
            # recompute QA metrics after recovery
            q_tokens = key_tokens(question)
            qa_overlap = overlap_ratio(question, answer_text)
            qa_token_hit = contains_any_token(answer_text, q_tokens) if q_tokens else True
            metrics["qa_overlap"] = qa_overlap
            metrics["qa_token_match"] = 1 if qa_token_hit else 0
            metrics["qa_consistent"] = 1 if (qa_overlap >= 0.05 and qa_token_hit) else 0
            ctx_map = build_context_measure_map([s.chunk.text for s in strict_contexts])
            metrics["numeric_preservation"] = verify_answer_numeric(answer_text, ctx_map, tol_ratio=0.05)
            metrics["recovery_round"] = 1
            metrics["fallback_used"] = "qa_recover1"

            if _detect_mismatch() and len(contexts) > 1:
                recovery_attempts += 1
                strict_contexts2 = strict_contexts[:1]
                answer_text = generate_answer(question, strict_contexts2, cfg, qtype=analysis.qtype, recovery=True)
                q_tokens = key_tokens(question)
                qa_overlap = overlap_ratio(question, answer_text)
                qa_token_hit = contains_any_token(answer_text, q_tokens) if q_tokens else True
                metrics["qa_overlap"] = qa_overlap
                metrics["qa_token_match"] = 1 if qa_token_hit else 0
                metrics["qa_consistent"] = 1 if (qa_overlap >= 0.05 and qa_token_hit) else 0
                ctx_map = build_context_measure_map([s.chunk.text for s in strict_contexts2])
                metrics["numeric_preservation"] = verify_answer_numeric(answer_text, ctx_map, tol_ratio=0.05)
                metrics["recovery_round"] = 2
                metrics["fallback_used"] = "qa_recover2"

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
