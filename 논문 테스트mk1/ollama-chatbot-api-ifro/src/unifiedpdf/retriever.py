from __future__ import annotations

import json
import math
import concurrent.futures
import time
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .types import Chunk, RetrievedSpan
from .utils import char_ngrams, safe_div
from .vector_store import make_vector_store
from .embedding import get_embedder
from .config import PipelineConfig
from .timeouts import run_with_timeout, SEARCH_TIMEOUT_S
from .question_classifier import QuestionClassifier, QuestionType


class InMemoryBM25:
    def __init__(self, docs: List[Chunk], n_min: int = 3, n_max: int = 5, k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.n_min = n_min
        self.n_max = n_max
        self.k1 = k1
        self.b = b
        self.doc_terms: List[List[str]] = [char_ngrams(c.text, n_min, n_max) for c in docs if hasattr(c, 'text')]
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
        
        # 정확한 키워드 매칭 강화
        exact_match_boost = 2.0
        q_lower = q.lower()
        
        for i, terms in enumerate(self.doc_terms):
            tf = Counter(terms)
            dl = self.doc_len[i]
            s = 0.0
            
            # BM25 점수 계산
            for t, qf in q_tf.items():
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
                s += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
            
            # 정확한 키워드 매칭 보너스
            doc_text = self.docs[i].text.lower()
            if q_lower in doc_text:
                s *= exact_match_boost
            
            scores[i] = s
        
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:topk]


class InMemoryTFIDFVector:
    pass  # kept for backward import safety; moved to vector_store


def normalized_weighted_merge(
    vector_hits: List[Tuple[int, float]],
    bm25_hits: List[Tuple[int, float]],
    question_type: str = "general",
) -> Dict[int, float]:
    """
    점수 정규화 + 동적 가중치를 사용한 이상적인 병합 방법
    """
    if not vector_hits and not bm25_hits:
        return {}
    
    # 질문 유형별 가중치 설정
    weights = {
        "definition": (0.7, 0.3),      # 정의형: 의미적 유사성 우선
        "procedural": (0.7, 0.3),      # 절차형: 의미적 유사성 우선
        "numeric": (0.4, 0.6),         # 수치형: 정확한 키워드 매칭 우선
        "technical_spec": (0.4, 0.6),  # 기술사양: 정확한 키워드 매칭 우선
        "system_info": (0.3, 0.7),     # 시스템정보: 정확한 키워드 매칭 우선
        "comparative": (0.6, 0.4),     # 비교형: 균형
        "problem": (0.6, 0.4),         # 문제해결형: 균형
        "operational": (0.7, 0.3),     # 운영형: 의미적 유사성 우선
        "general": (0.6, 0.4),         # 일반형: 균형
    }
    
    w_vector, w_bm25 = weights.get(question_type, (0.6, 0.4))
    
    # 점수 정규화 (Min-Max Normalization)
    vector_scores = [score for _, score in vector_hits] if vector_hits else []
    bm25_scores = [score for _, score in bm25_hits] if bm25_hits else []
    
    # 정규화 범위 계산
    vector_min = min(vector_scores) if vector_scores else 0
    vector_max = max(vector_scores) if vector_scores else 1
    bm25_min = min(bm25_scores) if bm25_scores else 0
    bm25_max = max(bm25_scores) if bm25_scores else 1
    
    # 범위가 0인 경우 처리
    vector_range = vector_max - vector_min if vector_max > vector_min else 1
    bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1
    
    # 병합된 점수 계산
    scores: Dict[int, float] = defaultdict(float)
    
    # 벡터 결과 처리
    for idx, score in vector_hits:
        normalized_score = (score - vector_min) / vector_range
        scores[idx] = w_vector * normalized_score
    
    # BM25 결과 처리
    for idx, score in bm25_hits:
        normalized_score = (score - bm25_min) / bm25_range
        if idx in scores:
            scores[idx] += w_bm25 * normalized_score
        else:
            scores[idx] = w_bm25 * normalized_score
    
    return scores


# 기존 RRF 함수는 호환성을 위해 유지 (deprecated)
def rrf_merge(
    vector_hits: List[Tuple[int, float]],
    bm25_hits: List[Tuple[int, float]],
    k: int,
    weight_vector: float = 0.2,  # 벡터 가중치 대폭 감소
    weight_bm25: float = 0.8,    # BM25 가중치 대폭 증가
) -> Dict[int, float]:
    """
    DEPRECATED: 기존 RRF 병합 방법 (호환성 유지)
    새로운 normalized_weighted_merge 함수 사용 권장
    """
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
        
        # 질문 분류기 초기화
        from .question_classifier import QuestionClassifier
        self._question_classifier = QuestionClassifier(cfg.domain.domain_dict_path)
    
    def _expand_query(self, query: str) -> str:
        """쿼리 확장 (동의어 추가 등)"""
        # 현재는 단순히 원본 쿼리 반환
        # 향후 동의어 사전이나 확장 로직 추가 가능
        return query
    
    def _expand_query_with_domain_dict(self, query: str, domain_dict: dict = None) -> str:
        """Domain Dictionary를 활용한 쿼리 확장"""
        if not domain_dict:
            return query
        
        expanded_terms = []
        
        # 동의어 확장
        synonyms = domain_dict.get("synonyms", {})
        for main_term, synonym_list in synonyms.items():
            if main_term.lower() in query.lower():
                expanded_terms.extend(synonym_list)
        
        # 질문 유형별 관련 키워드 추가
        question_lower = query.lower()
        
        # 시스템 정보 관련 확장
        if any(term in question_lower for term in ["url", "주소", "접속", "대시보드"]):
            system_keywords = domain_dict.get("system_info", [])
            expanded_terms.extend([kw for kw in system_keywords if kw.lower() in question_lower])
        
        # 기술사양 관련 확장
        if any(term in question_lower for term in ["입력변수", "수질", "인자", "모델"]):
            tech_keywords = domain_dict.get("technical_spec", [])
            expanded_terms.extend([kw for kw in tech_keywords if kw.lower() in question_lower])
        
        # 확장된 용어들을 원본 쿼리에 추가
        if expanded_terms:
            expanded_query = query + " " + " ".join(set(expanded_terms))
            return expanded_query
        
        return query
    
    def _determine_dynamic_k(self, query: str, question_type: str, domain_dict: dict = None) -> int:
        """질문 유형과 복잡도에 따른 동적 k 값 결정"""
        base_k = 50
        
        # 질문 유형별 조정
        if question_type in ["technical_spec", "definition"]:
            base_k = 80  # 기술사양은 더 많은 검색 결과 필요
        elif question_type in ["system_info", "numeric"]:
            base_k = 60  # 시스템 정보는 정확한 매칭 중요
        else:
            base_k = 50
        
        # 도메인 키워드 매칭에 따른 추가 조정
        if domain_dict:
            domain_keywords = domain_dict.get("keywords", [])
            matched_keywords = sum(1 for kw in domain_keywords if kw.lower() in query.lower())
            if matched_keywords >= 3:
                base_k += 20  # 많은 도메인 키워드 매칭 시 더 많은 결과 필요
        
        return min(base_k, 100)  # 최대 100개로 제한

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
        # Classify question type and enhance query
        question_type, confidence, type_scores = self._question_classifier.classify_question(query)
        enhanced_query = self._question_classifier.enhance_query_for_type(query, question_type)
        
        # Domain Dictionary 로드
        domain_dict = None
        try:
            import json
            domain_dict_path = self.cfg.domain.domain_dict_path
            if domain_dict_path and Path(domain_dict_path).exists():
                with open(domain_dict_path, 'r', encoding='utf-8') as f:
                    domain_dict = json.load(f)
        except Exception:
            pass
        
        # 동적 k 값 결정
        dynamic_topk = self._determine_dynamic_k(query, question_type.value, domain_dict)
        actual_topk = max(topk_each, dynamic_topk)
        
        # Domain Dictionary를 활용한 쿼리 확장
        expanded_query = self._expand_query_with_domain_dict(enhanced_query, domain_dict)
        
        # Try cache first
        cache_key = self._cache_key(expanded_query, actual_topk)
        cached = self._cache_get(cache_key)
        if cached is not None:
            v_hits = list(cached.get("v_hits", []))
            b_hits = list(cached.get("b_hits", []))
            vt, bt = 0, 0
            timings: Dict[str, int] = {
                "vector_time_ms": vt, 
                "bm25_time_ms": bt, 
                "cache_hit": 1,
                "question_type": question_type.value,
                "type_confidence": confidence,
                "enhanced_query": enhanced_query,
                "dynamic_topk": actual_topk,
                "expanded_query": expanded_query
            }
        else:
            timings = {"cache_hit": 0}
            if self._parallel_enabled:
                # Run vector and BM25 in parallel with shared timeout budget
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    start = time.time()
                    f_vec = ex.submit(self.vec.query, expanded_query, actual_topk)
                    f_bm = ex.submit(self.bm25.query, expanded_query, actual_topk)
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
                v_hits = run_with_timeout(lambda: self.vec.query(expanded_query, actual_topk), timeout_s=SEARCH_TIMEOUT_S, default=[])
                vt = int((time.time() - t0) * 1000)
                t1 = time.time()
                b_hits = run_with_timeout(lambda: self.bm25.query(expanded_query, actual_topk), timeout_s=SEARCH_TIMEOUT_S, default=[])
                bt = int((time.time() - t1) * 1000)

            timings.update({
                "vector_time_ms": vt, 
                "bm25_time_ms": bt,
                "question_type": question_type.value,
                "type_confidence": confidence,
                "enhanced_query": enhanced_query,
                "dynamic_topk": actual_topk,
                "expanded_query": expanded_query
            })
            # Save cache
            self._cache_set(cache_key, {"v_hits": v_hits, "b_hits": b_hits})
        # 새로운 정규화된 가중치 병합 방법 사용
        merged_scores = normalized_weighted_merge(v_hits, b_hits, question_type.value)

        spans: List[RetrievedSpan] = []
        aux_v = {i: s for i, s in v_hits}
        aux_b = {i: s for i, s in b_hits}
        # Apply simple numeric bonuses and tie-break priority when question is numeric
        aux_bonus = 0.08
        density_bonus = 0.04
        is_numeric_q = (question_type.value == "numeric") if hasattr(question_type, 'value') else (question_type == "numeric")
        # Build priority map: base_numeric(2) > aux_numeric(1) > non_numeric(0)
        priority: Dict[int, int] = {}
        adjusted: Dict[int, float] = {}
        for idx, sc in merged_scores.items():
            ch = self.chunks[idx]
            pri = 0
            if ch.extra.get("numeric_anchor"):
                pri = 2
            if ch.extra.get("aux_numeric"):
                pri = 1 if pri == 0 else 2  # aux only -> 1, anchor has priority
            priority[idx] = pri
            bonus = 0.0
            if is_numeric_q:
                if ch.extra.get("aux_numeric"):
                    bonus += aux_bonus
                den = float(ch.extra.get("anchor_density", 0.0) or 0.0)
                # scale density to a reasonable range
                bonus += density_bonus * min(1.0, den * 500.0)
            adjusted[idx] = sc + bonus

        # Read bonuses from config
        try:
            aux_bns = float(getattr(self.cfg, "retriever_aux_bonus", 0.08))
            dens_bns = float(getattr(self.cfg, "retriever_density_bonus", 0.04))
        except Exception:
            aux_bns, dens_bns = 0.08, 0.04

        adjusted2: Dict[int, float] = {}
        for idx, sc in merged_scores.items():
            ch = self.chunks[idx]
            bonus = 0.0
            if is_numeric_q:
                if ch.extra.get("aux_numeric"):
                    bonus += aux_bns
                den = float(ch.extra.get("anchor_density", 0.0) or 0.0)
                bonus += dens_bns * min(1.0, den * 500.0)
            adjusted2[idx] = sc + bonus

        ranked = sorted(adjusted2.items(), key=lambda x: (x[1], priority.get(x[0], 0)), reverse=True)
        for rank, (idx, merged_score) in enumerate(ranked, start=1):
            ch = self.chunks[idx]
            spans.append(
                RetrievedSpan(
                    chunk=ch,
                    source="hybrid_normalized",
                    score=merged_score,
                    rank=rank,
                    aux_scores={
                        "merged": merged_score,
                        "vector": aux_v.get(idx, 0.0),
                        "bm25": aux_b.get(idx, 0.0),
                        "question_type": question_type.value,
                        "type_confidence": confidence,
                        "type_scores": type_scores,
                    },
                )
            )
        
        # Add question classification info to timings
        timings.update({
            "question_type": question_type.value,
            "type_confidence": confidence,
            "enhanced_query": enhanced_query,
        })
        
        return spans, timings
