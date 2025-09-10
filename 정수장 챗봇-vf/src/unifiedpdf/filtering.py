from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from .config import PipelineConfig
from .types import RetrievedSpan
from .utils import zscore_clip_to_unit


def calibrate_and_filter(
    spans: List[RetrievedSpan], cfg: PipelineConfig, threshold_override: Optional[float] = None
) -> Tuple[List[RetrievedSpan], Dict[str, float]]:
    if not spans:
        return [], {"filter_in": 0, "filter_out": 0, "filter_pass_rate": 0.0}

    # Group scores by source type for min-max normalization
    per_source_scores: Dict[str, List[float]] = defaultdict(list)
    for s in spans:
        for k, v in s.aux_scores.items():
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
