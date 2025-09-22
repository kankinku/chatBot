from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Chunk:
    doc_id: str
    filename: str
    page: Optional[int]
    start_offset: int
    length: int
    text: str
    # Optional neighbor hints for ±1 page/position
    neighbor_hint: Optional[Tuple[str, Optional[int], int]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedSpan:
    chunk: Chunk
    source: str  # "vector" | "bm25" | "rerank" | etc.
    score: float
    rank: int
    aux_scores: Dict[str, float] = field(default_factory=dict)
    calibrated_conf: Optional[float] = None


@dataclass
class Answer:
    text: str
    confidence: float
    sources: List[RetrievedSpan]
    metrics: Dict[str, Any]
    fallback_used: str = "none"

