from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


def now_ms() -> int:
    return int(time.time() * 1000)


def clean_text_for_hash(text: str) -> str:
    # Normalize whitespace, hyphens, and newlines for stable hashing
    t = re.sub(r"[\s\u00A0]+", " ", text.strip())
    t = t.replace("-", "-")
    return t


def md5_head160(text: str) -> str:
    cleaned = clean_text_for_hash(text)[:160]
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> List[str]:
    t = re.sub(r"\s+", " ", text.lower())
    ngrams: List[str] = []
    for n in range(n_min, n_max + 1):
        for i in range(max(0, len(t) - n + 1)):
            ngrams.append(t[i : i + n])
    return ngrams


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def overlap_ratio(query: str, context: str) -> float:
    q = set(char_ngrams(query, 3, 5))
    c = set(char_ngrams(context, 3, 5))
    if not q or not c:
        return 0.0
    return len(q & c) / len(q)


def zscore_clip_to_unit(x: float, mu: float, sigma: float, zmax: float = 3.0) -> float:
    if sigma <= 1e-9:
        return 0.5
    z = (x - mu) / sigma
    z = max(-zmax, min(zmax, z))
    # Project z in [-zmax,zmax] into [0,1]
    return (z + zmax) / (2 * zmax)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-12 else default


def key_tokens(query: str) -> List[str]:
    # Simple heuristic: Korean/English alphanumerics and common unit symbols, length>=2
    toks = re.findall(r"[0-9A-Za-z가-힣\.\-/%]+", query)
    toks = [t.lower() for t in toks if len(t) >= 2]
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def contains_any_token(text: str, tokens: Sequence[str]) -> bool:
    tl = text.lower()
    return any(t in tl for t in tokens)


@dataclass
class TimeoutBudget:
    ms: int

    def remaining(self, start_ms: int) -> int:
        return max(0, self.ms - (now_ms() - start_ms))
