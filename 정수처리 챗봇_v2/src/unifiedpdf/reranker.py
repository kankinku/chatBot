from __future__ import annotations

import time
from typing import List, Tuple

from .timeouts import run_with_timeout, RERANK_TIMEOUT_S
from .types import RetrievedSpan


def _safe_import_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        return CrossEncoder
    except Exception:
        return None


class LightweightReranker:
    def score(self, query: str, texts: List[str]) -> List[float]:
        # Simple char n-gram overlap score as a fallback.
        from .utils import overlap_ratio
        return [overlap_ratio(query, t) for t in texts]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        CrossEncoder = _safe_import_cross_encoder()
        self.model = CrossEncoder(model_name) if CrossEncoder else None
        self.use_fp16 = use_fp16

    def score(self, query: str, texts: List[str]) -> List[float]:
        if self.model is None:
            return LightweightReranker().score(query, texts)
        pairs = [[query, t] for t in texts]
        try:
            scores = self.model.predict(pairs, convert_to_tensor=False, show_progress_bar=False)
            return [float(s) for s in scores]
        except Exception:
            return LightweightReranker().score(query, texts)


def rerank_spans(query: str, spans: List[RetrievedSpan], use_cross: bool) -> Tuple[List[RetrievedSpan], int]:
    start = time.time()
    texts = [s.chunk.text for s in spans]
    reranker = CrossEncoderReranker() if use_cross else LightweightReranker()
    def _run():
        scores = reranker.score(query, texts)
        return scores
    scores = run_with_timeout(_run, timeout_s=RERANK_TIMEOUT_S, default=[0.0] * len(texts))
    for s, r in zip(spans, scores):
        s.aux_scores["rerank"] = float(r)
    spans = sorted(spans, key=lambda x: x.aux_scores.get("rerank", 0.0), reverse=True)
    took_ms = int((time.time() - start) * 1000)
    return spans, took_ms

