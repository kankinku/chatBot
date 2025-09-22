from __future__ import annotations

import math
import concurrent.futures
import time
from collections import Counter, defaultdict, OrderedDict
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
        # Retrieval cache (LRU)
        self._cache_enabled = bool(getattr(cfg.flags, "enable_retrieval_cache", True))
        self._cache_max = int(getattr(cfg.flags, "retrieval_cache_size", 256)) or 0
        self._cache: "OrderedDict[tuple, dict]" = OrderedDict()
        # Simple corpus signature to mitigate stale cache across corpora
        try:
            total_len = sum(len(c.text) for c in chunks)
        except Exception:
            total_len = len(chunks)
        self._corpus_sig = (len(chunks), int(total_len))
        self._parallel_enabled = bool(getattr(cfg.flags, "enable_parallel_search", True))

    def _cache_key(self, query: str, topk_each: int) -> tuple:
        return (query, int(topk_each), self._corpus_sig, self.cfg.config_hash())

    def _cache_get(self, key: tuple):
        if not (self._cache_enabled and self._cache_max > 0):
            return None
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    def _cache_set(self, key: tuple, value: dict) -> None:
        if not (self._cache_enabled and self._cache_max > 0):
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def retrieve(
        self,
        query: str,
        topk_each: int = 50,
        rrf_k: int = 60,
        rrf_weights: Tuple[float, float] = (0.6, 0.4),
    ) -> Tuple[List[RetrievedSpan], Dict[str, int]]:
        # Try cache first
        cache_key = self._cache_key(query, topk_each)
        cached = self._cache_get(cache_key)
        if cached is not None:
            v_hits = list(cached.get("v_hits", []))
            b_hits = list(cached.get("b_hits", []))
            vt, bt = 0, 0
            timings: Dict[str, int] = {"vector_time_ms": vt, "bm25_time_ms": bt, "cache_hit": 1}
        else:
            timings = {"cache_hit": 0}
            if self._parallel_enabled:
                # Run vector and BM25 in parallel with shared timeout budget
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    start = time.time()
                    f_vec = ex.submit(self.vec.query, query, topk_each)
                    f_bm = ex.submit(self.bm25.query, query, topk_each)
                    v_hits: List[Tuple[int, float]] = []
                    b_hits: List[Tuple[int, float]] = []
                    # vector
                    t0 = time.time()
                    try:
                        v_hits = f_vec.result(timeout=SEARCH_TIMEOUT_S) or []
                    except Exception:
                        v_hits = []
                    vt = int((time.time() - t0) * 1000)
                    # bm25 (respect remaining budget best-effort)
                    t1 = time.time()
                    try:
                        remaining = max(0.0, SEARCH_TIMEOUT_S - (time.time() - start))
                        b_hits = f_bm.result(timeout=remaining) or []
                    except Exception:
                        b_hits = []
                    bt = int((time.time() - t1) * 1000)
            else:
                # Sequential (fallback to previous behavior)
                t0 = time.time()
                v_hits = run_with_timeout(lambda: self.vec.query(query, topk_each), timeout_s=SEARCH_TIMEOUT_S, default=[])
                vt = int((time.time() - t0) * 1000)
                t1 = time.time()
                b_hits = run_with_timeout(lambda: self.bm25.query(query, topk_each), timeout_s=SEARCH_TIMEOUT_S, default=[])
                bt = int((time.time() - t1) * 1000)

            timings.update({"vector_time_ms": vt, "bm25_time_ms": bt})
            # Save cache
            self._cache_set(cache_key, {"v_hits": v_hits, "b_hits": b_hits})
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
        return spans, timings
