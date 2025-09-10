from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from .types import Chunk, RetrievedSpan
from .utils import char_ngrams, safe_div
from .vector_store import make_vector_store
from .embedding import get_embedder
from .config import PipelineConfig
from .timeouts import run_with_timeout, SEARCH_TIMEOUT_S


class InMemoryBM25:
    def __init__(self, docs: List[Chunk], n_min: int = 3, n_max: int = 5, k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.n_min = n_min
        self.n_max = n_max
        self.k1 = k1
        self.b = b
        self.doc_terms: List[List[str]] = [char_ngrams(c.text, n_min, n_max) for c in docs]
        self.doc_len = [len(t) for t in self.doc_terms]
        self.avgdl = safe_div(sum(self.doc_len), len(self.doc_len), 0.0) if self.doc_len else 0.0
        self.df: Dict[str, int] = defaultdict(int)
        for terms in self.doc_terms:
            for t in set(terms):
                self.df[t] += 1
        self.N = len(docs)
        self.idf: Dict[str, float] = {}
        for t, dfi in self.df.items():
            # BM25 IDF with log + correction
            self.idf[t] = math.log(1 + (self.N - dfi + 0.5) / (dfi + 0.5))

    def query(self, q: str, topk: int = 50) -> List[Tuple[int, float]]:
        q_terms = char_ngrams(q, self.n_min, self.n_max)
        q_tf = Counter(q_terms)
        scores = [0.0] * self.N
        for i, terms in enumerate(self.doc_terms):
            tf = Counter(terms)
            dl = self.doc_len[i]
            s = 0.0
            for t, qf in q_tf.items():
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
                s += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
            scores[i] = s
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:topk]


class InMemoryTFIDFVector:
    pass  # kept for backward import safety; moved to vector_store


def rrf_merge(
    vector_hits: List[Tuple[int, float]],
    bm25_hits: List[Tuple[int, float]],
    k: int,
    weight_vector: float = 0.6,
    weight_bm25: float = 0.4,
) -> Dict[int, float]:
    # RRF scoring: sum weights * 1/(k + rank)
    scores: Dict[int, float] = defaultdict(float)
    for r, (idx, _) in enumerate(vector_hits):
        scores[idx] += weight_vector * (1.0 / (k + r + 1))
    for r, (idx, _) in enumerate(bm25_hits):
        scores[idx] += weight_bm25 * (1.0 / (k + r + 1))
    return scores


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], cfg: PipelineConfig):
        self.chunks = chunks
        self.cfg = cfg
        self.bm25 = InMemoryBM25(chunks)
        embedder = get_embedder(cfg.embedding_model, use_gpu=cfg.flags.use_gpu) if cfg.flags.store_backend in {"faiss", "hnsw", "auto"} else None
        self.vec = make_vector_store(chunks, backend=cfg.flags.store_backend, index_dir=cfg.vector_store_dir, embedder=embedder)

    def retrieve(
        self,
        query: str,
        topk_each: int = 50,
        rrf_k: int = 60,
        rrf_weights: Tuple[float, float] = (0.6, 0.4),
    ) -> Tuple[List[RetrievedSpan], Dict[str, int]]:
        import time
        t0 = time.time()
        v_hits = run_with_timeout(lambda: self.vec.query(query, topk_each), timeout_s=SEARCH_TIMEOUT_S, default=[])
        vt = int((time.time() - t0) * 1000)
        t1 = time.time()
        b_hits = run_with_timeout(lambda: self.bm25.query(query, topk_each), timeout_s=SEARCH_TIMEOUT_S, default=[])
        bt = int((time.time() - t1) * 1000)
        rrf_scores = rrf_merge(v_hits, b_hits, rrf_k, rrf_weights[0], rrf_weights[1])

        spans: List[RetrievedSpan] = []
        aux_v = {i: s for i, s in v_hits}
        aux_b = {i: s for i, s in b_hits}
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (idx, rrf_s) in enumerate(ranked, start=1):
            ch = self.chunks[idx]
            spans.append(
                RetrievedSpan(
                    chunk=ch,
                    source="hybrid_rrf",
                    score=rrf_s,
                    rank=rank,
                    aux_scores={
                        "rrf": rrf_s,
                        "vector": aux_v.get(idx, 0.0),
                        "bm25": aux_b.get(idx, 0.0),
                    },
                )
            )
        return spans, {"vector_time_ms": vt, "bm25_time_ms": bt}
