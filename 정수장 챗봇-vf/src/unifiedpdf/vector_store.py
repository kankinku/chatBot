from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from .types import Chunk
from .utils import char_ngrams
from .embedding import EmbeddingModel


class VectorStore:
    def add(self, chunks: List[Chunk]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def query(self, q: str, topk: int) -> List[Tuple[int, float]]:  # returns (chunk_index, score)
        raise NotImplementedError


class InMemoryTFIDFVector(VectorStore):
    def __init__(self, chunks: List[Chunk], n_min: int = 3, n_max: int = 5):
        import math
        from collections import Counter, defaultdict

        self.chunks = chunks
        self.n_min = n_min
        self.n_max = n_max
        self.doc_terms: List[List[str]] = [char_ngrams(c.text, n_min, n_max) for c in chunks]
        self.df = defaultdict(int)
        for terms in self.doc_terms:
            for t in set(terms):
                self.df[t] += 1
        self.N = len(chunks)
        # Precompute tf-idf norms
        self.doc_vecs: List[dict] = []
        self.doc_norms: List[float] = []
        for terms in self.doc_terms:
            tf = Counter(terms)
            vec = {}
            for t, f in tf.items():
                idf = math.log((self.N + 1) / (self.df[t] + 1)) + 1.0
                vec[t] = f * idf
            norm = math.sqrt(sum(v * v for v in vec.values())) if vec else 1.0
            self.doc_vecs.append(vec)
            self.doc_norms.append(norm)

    def add(self, chunks: List[Chunk]) -> None:
        # In-memory: all added at init; no-op for now
        return

    def query(self, q: str, topk: int) -> List[Tuple[int, float]]:
        import math
        from collections import Counter

        q_terms = char_ngrams(q, self.n_min, self.n_max)
        tf = Counter(q_terms)
        qvec = {}
        for t, f in tf.items():
            idf = math.log((self.N + 1) / (self.df.get(t, 0) + 1)) + 1.0
            qvec[t] = f * idf
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        sims: List[Tuple[int, float]] = []
        for i, dvec in enumerate(self.doc_vecs):
            dot = 0.0
            for t, qv in qvec.items():
                dv = dvec.get(t)
                if dv:
                    dot += qv * dv
            sim = dot / (qnorm * (self.doc_norms[i] or 1.0))
            sims.append((i, sim))
        ranked = sorted(sims, key=lambda x: x[1], reverse=True)
        return ranked[:topk]


class FaissVectorStore(VectorStore):
    def __init__(self, index_dir: str, embedder: Optional[EmbeddingModel]):
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.ok = False
        self.index = None
        self.idmap: List[int] = []
        self.dim = 0
        try:
            import faiss  # type: ignore
            self.faiss = faiss
        except Exception:
            self.faiss = None
        self._load()

    def _load(self):
        if self.faiss is None:
            return
        idx_path = self.index_dir / "index.faiss"
        map_path = self.index_dir / "mapping.json"
        meta_path = self.index_dir / "meta.json"
        if not (idx_path.exists() and map_path.exists() and meta_path.exists()):
            return
        import json
        self.index = self.faiss.read_index(str(idx_path))
        with map_path.open("r", encoding="utf-8") as f:
            self.idmap = json.load(f)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            self.dim = int(meta.get("dim", 0))
        self.ok = True

    def add(self, chunks: List[Chunk]) -> None:
        # Built offline via build_vector_index.py
        return

    def query(self, q: str, topk: int) -> List[Tuple[int, float]]:
        if not (self.ok and self.index and self.embedder):
            return []
        import numpy as np
        qv = self.embedder.embed_query(q).reshape(1, -1)
        if qv.shape[1] != self.dim:
            return []
        D, I = self.index.search(qv, topk)
        out: List[Tuple[int, float]] = []
        for i, d in zip(I[0].tolist(), D[0].tolist()):
            if i < 0 or i >= len(self.idmap):
                continue
            out.append((self.idmap[i], float(d)))
        return out


class HnswVectorStore(VectorStore):
    def __init__(self, index_dir: str, embedder: Optional[EmbeddingModel]):
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.ok = False
        self.p = None
        self.idmap: List[int] = []
        self.dim = 0
        try:
            import hnswlib  # type: ignore
            self.hnswlib = hnswlib
        except Exception:
            self.hnswlib = None
        self._load()

    def _load(self):
        if self.hnswlib is None:
            return
        idx_path = self.index_dir / "index.hnsw"
        map_path = self.index_dir / "mapping.json"
        meta_path = self.index_dir / "meta.json"
        if not (idx_path.exists() and map_path.exists() and meta_path.exists()):
            return
        import json
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            self.dim = int(meta.get("dim", 0))
            space = meta.get("space", "cosine")
        self.p = self.hnswlib.Index(space=space, dim=self.dim)
        self.p.load_index(str(idx_path))
        with map_path.open("r", encoding="utf-8") as f:
            self.idmap = json.load(f)
        try:
            self.p.set_ef(64)
        except Exception:
            pass
        self.ok = True

    def add(self, chunks: List[Chunk]) -> None:
        return

    def query(self, q: str, topk: int) -> List[Tuple[int, float]]:
        if not (self.ok and self.p and self.embedder):
            return []
        import numpy as np
        qv = self.embedder.embed_query(q).reshape(1, -1)
        if qv.shape[1] != self.dim:
            return []
        labels, distances = self.p.knn_query(qv, k=topk)
        out: List[Tuple[int, float]] = []
        for i, d in zip(labels[0].tolist(), distances[0].tolist()):
            if i < 0 or i >= len(self.idmap):
                continue
            out.append((self.idmap[i], float(-d)))  # convert distance to similarity-ish
        return out


def make_vector_store(
    chunks: List[Chunk], backend: str = "auto", index_dir: str = "vector_store", embedder: Optional[EmbeddingModel] = None
) -> VectorStore:
    if backend == "faiss":
        vs = FaissVectorStore(index_dir=index_dir, embedder=embedder)
        return vs if vs.ok else InMemoryTFIDFVector(chunks)
    if backend == "hnsw":
        vs = HnswVectorStore(index_dir=index_dir, embedder=embedder)
        return vs if vs.ok else InMemoryTFIDFVector(chunks)
    # auto: try faiss, then hnsw, else tfidf
    vf = FaissVectorStore(index_dir=index_dir, embedder=embedder)
    if vf.ok:
        return vf
    vh = HnswVectorStore(index_dir=index_dir, embedder=embedder)
    if vh.ok:
        return vh
    return InMemoryTFIDFVector(chunks)
