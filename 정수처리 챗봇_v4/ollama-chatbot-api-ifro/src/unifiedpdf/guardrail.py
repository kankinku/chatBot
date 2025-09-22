from __future__ import annotations

from typing import List

from .config import PipelineConfig
from .types import RetrievedSpan
from .utils import overlap_ratio, key_tokens, contains_any_token


def guard_check(question: str, contexts: List[RetrievedSpan], cfg: PipelineConfig) -> dict:
    ctx_text = "\n".join(s.chunk.text for s in contexts)
    ov = overlap_ratio(question, ctx_text)
    tokens = key_tokens(question)
    has_key = contains_any_token(ctx_text, tokens) if tokens else False
    # snippet warn: count sentences with very low overlap
    sentences = []
    for s in contexts:
        sentences.extend([t.strip() for t in s.chunk.text.split(".") if t.strip()])
    low_sim = 0
    for sent in sentences:
        if overlap_ratio(question, sent) < 0.05:
            low_sim += 1
    passed = (ov >= cfg.thresholds.guard_overlap_threshold) and (len(tokens) >= cfg.thresholds.guard_key_tokens_min) and has_key
    return {
        "overlap_ratio": ov,
        "key_tokens": tokens,
        "snippet_warn_count": low_sim,
        "hard_blocked": 0 if passed else 1,
    }
