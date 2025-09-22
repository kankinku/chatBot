from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from .config import PipelineConfig
from .types import RetrievedSpan
from .utils import zscore_clip_to_unit, overlap_ratio, key_tokens, contains_any_token


def calibrate_and_filter(
    spans: List[RetrievedSpan], cfg: PipelineConfig, threshold_override: Optional[float] = None
) -> Tuple[List[RetrievedSpan], Dict[str, float]]:
    if not spans:
        return [], {"filter_in": 0, "filter_out": 0, "filter_pass_rate": 0.0}

    # Group scores by source type for min-max normalization
    per_source_scores: Dict[str, List[float]] = defaultdict(list)
    for s in spans:
        for k, v in s.aux_scores.items():
            # Only process numeric values
            if isinstance(v, (int, float)):
                per_source_scores[k].append(v)

    per_source_minmax: Dict[str, Tuple[float, float]] = {}
    for src, vals in per_source_scores.items():
        vmin, vmax = min(vals), max(vals)
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
        per_source_minmax[src] = (vmin, vmax)

    # Compute per-source normalized scores and z-score capped projection
    all_norms: List[float] = []
    per_span_norms: List[float] = []
    for s in spans:
        norm_components: List[float] = []
        for src, val in s.aux_scores.items():
            # Only process numeric values
            if isinstance(val, (int, float)):
                vmin, vmax = per_source_minmax.get(src, (0.0, 1.0))
                s_prime = (val - vmin) / (vmax - vmin + 1e-9)
                norm_components.append(s_prime)
        # if no aux scores, fallback to base
        if not norm_components:
            norm_components = [cfg.thresholds.base_no_answer_confidence]
        avg_norm = sum(norm_components) / len(norm_components)
        per_span_norms.append(avg_norm)
        all_norms.append(avg_norm)

    mu = statistics.mean(all_norms)
    sigma = statistics.pstdev(all_norms) if len(all_norms) > 1 else 0.0

    passed: List[RetrievedSpan] = []
    thr = cfg.thresholds.confidence_threshold if threshold_override is None else float(threshold_override)
    for s, avg_norm in zip(spans, per_span_norms):
        s.calibrated_conf = zscore_clip_to_unit(avg_norm, mu, sigma, zmax=3.0)
        if (s.calibrated_conf or 0.0) >= thr:
            passed.append(s)

    # page/paragraph diversity constraint: no more than 1 per identical page/start
    seen_pagepos = set()
    diversified: List[RetrievedSpan] = []
    for s in passed:
        key = (s.chunk.filename, s.chunk.page, s.chunk.start_offset)
        if key in seen_pagepos:
            continue
        seen_pagepos.add(key)
        diversified.append(s)

    stats = {
        "filter_in": len(spans),
        "filter_out": len(spans) - len(diversified),
        "filter_pass_rate": (len(diversified) / len(spans)) if spans else 0.0,
    }
    return diversified, stats


def annotate_and_pre_filter(
    question: str, spans: List[RetrievedSpan], cfg: PipelineConfig
) -> Tuple[List[RetrievedSpan], Dict[str, float]]:
    """Annotate spans with overlap/keyword scores and drop low-quality ones early.

    - Adds aux_scores["ovlp"]: overlap_ratio(question, span.text)
    - Adds aux_scores["kw"]: 1.0 if any key token from question appears in span.text else 0.0
    - Filters out spans with ovlp < cfg.thresholds.context_min_overlap
    - If question has tokens, filters out spans with kw == 0 when keyword_filter_min >= 1
    """
    if not spans:
        return [], {"pre_in": 0, "pre_out": 0, "pre_pass_rate": 0.0}

    toks = key_tokens(question)
    require_kw = (len(toks) > 0) and (cfg.thresholds.keyword_filter_min >= 1)

    kept: List[RetrievedSpan] = []
    removed = 0
    for s in spans:
        ctx = s.chunk.text
        ov = overlap_ratio(question, ctx)
        kw_hit = contains_any_token(ctx, toks) if toks else False
        s.aux_scores["ovlp"] = float(ov)
        s.aux_scores["kw"] = 1.0 if kw_hit else 0.0
        if ov < cfg.thresholds.context_min_overlap:
            removed += 1
            continue
        if require_kw and not kw_hit:
            removed += 1
            continue
        kept.append(s)

    stats = {
        "pre_in": len(spans),
        "pre_out": removed,
        "pre_pass_rate": (len(kept) / len(spans)) if spans else 0.0,
    }
    return kept, stats


def apply_rerank_threshold(spans: List[RetrievedSpan], cfg: PipelineConfig) -> Tuple[List[RetrievedSpan], Dict[str, float]]:
    """Drop spans whose normalized rerank score falls below threshold.

    Normalizes using per-batch min-max over available aux_scores["rerank"].
    """
    if not spans:
        return [], {"rerank_filtered": 0}

    scores = [s.aux_scores.get("rerank", 0.0) for s in spans]
    vmin, vmax = min(scores), max(scores)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    thr = cfg.thresholds.rerank_threshold
    kept: List[RetrievedSpan] = []
    removed = 0
    for s in spans:
        r = s.aux_scores.get("rerank", 0.0)
        rn = (r - vmin) / (vmax - vmin + 1e-9)
        s.aux_scores["rerank_norm"] = rn
        if rn >= thr:
            kept.append(s)
        else:
            removed += 1
    return kept, {"rerank_filtered": removed}
