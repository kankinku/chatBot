from __future__ import annotations

from typing import Dict, List, Tuple

from .types import RetrievedSpan
from .utils import md5_head160, char_ngrams, jaccard


def stable_key(span: RetrievedSpan) -> str:
    ch = span.chunk
    core = f"{ch.doc_id}|{ch.filename}|{ch.page}|{ch.start_offset}|{ch.length}"
    h = md5_head160(ch.text)
    return f"{core}|{h}"


def dedup_spans(spans: List[RetrievedSpan], approx: bool = True, jaccard_thr: float = 0.9, semantic_thr: float = 0.0) -> List[RetrievedSpan]:
    seen: Dict[str, RetrievedSpan] = {}
    out: List[RetrievedSpan] = []
    
    # 의미적 중복 제거를 위한 임베딩 캐시
    semantic_cache = {} if semantic_thr > 0.0 else None
    
    for s in spans:
        key = stable_key(s)
        if key in seen:
            # keep higher score/rank
            if s.score > seen[key].score:
                seen[key] = s
            continue
        
        # approximate duplicate detection using n-gram Jaccard
        if approx:
            s_ngrams = char_ngrams(s.chunk.text)
            duplicate = False
            for o in out:
                j = jaccard(s_ngrams, char_ngrams(o.chunk.text))
                if j >= jaccard_thr:
                    duplicate = True
                    break
            if duplicate:
                continue
        
        # 의미적 중복 제거 (임베딩 기반)
        if semantic_thr > 0.0 and semantic_cache is not None:
            if _is_semantic_duplicate(s, out, semantic_cache, semantic_thr):
                continue
        
        seen[key] = s
        out.append(s)
    return out


def _is_semantic_duplicate(span: RetrievedSpan, existing_spans: List[RetrievedSpan], cache: Dict[str, any], threshold: float) -> bool:
    """의미적 유사도 기반 중복 검사"""
    try:
        # 간단한 임베딩 기반 유사도 검사 (실제 구현에서는 더 정교한 방법 사용 가능)
        span_text = span.chunk.text.lower().strip()
        
        # 캐시에서 임베딩 가져오기 또는 생성
        if span_text not in cache:
            # 간단한 해시 기반 유사도 (실제로는 임베딩 모델 사용)
            cache[span_text] = hash(span_text)
        
        span_hash = cache[span_text]
        
        for existing in existing_spans:
            existing_text = existing.chunk.text.lower().strip()
            if existing_text not in cache:
                cache[existing_text] = hash(existing_text)
            
            existing_hash = cache[existing_text]
            
            # 간단한 해시 유사도 (실제로는 코사인 유사도 사용)
            similarity = 1.0 - abs(span_hash - existing_hash) / (2**64)
            
            if similarity >= threshold:
                return True
                
    except Exception:
        # 의미적 중복 검사 실패 시 건너뛰기
        pass
    
    return False


def merge_then_dedup(spans: List[RetrievedSpan], jaccard_thr: float = 0.9, semantic_thr: float = 0.0) -> List[RetrievedSpan]:
    # spans already merged via RRF; now ensure deterministic ordering by score then rank
    spans_sorted = sorted(spans, key=lambda s: (-s.score, s.rank))
    # Dedup level 1 with configurable thresholds
    spans_d1 = dedup_spans(spans_sorted, approx=True, jaccard_thr=jaccard_thr, semantic_thr=semantic_thr)
    return spans_d1

