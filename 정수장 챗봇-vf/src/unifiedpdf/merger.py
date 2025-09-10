from __future__ import annotations

from typing import Dict, List, Tuple

from .types import RetrievedSpan
from .utils import md5_head160, char_ngrams, jaccard


def stable_key(span: RetrievedSpan) -> str:
    ch = span.chunk
    core = f"{ch.doc_id}|{ch.filename}|{ch.page}|{ch.start_offset}|{ch.length}"
    h = md5_head160(ch.text)
    return f"{core}|{h}"


def dedup_spans(spans: List[RetrievedSpan], approx: bool = True, jaccard_thr: float = 0.9) -> List[RetrievedSpan]:
    seen: Dict[str, RetrievedSpan] = {}
    out: List[RetrievedSpan] = []
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
        seen[key] = s
        out.append(s)
    return out


def merge_then_dedup(spans: List[RetrievedSpan]) -> List[RetrievedSpan]:
    # spans already merged via RRF; now ensure deterministic ordering by score then rank
    spans_sorted = sorted(spans, key=lambda s: (-s.score, s.rank))
    # Dedup level 1
    spans_d1 = dedup_spans(spans_sorted, approx=True)
    return spans_d1

