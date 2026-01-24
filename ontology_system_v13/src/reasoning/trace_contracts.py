from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime


@dataclass
class EdgeTrace:
    head_id: str
    tail_id: str
    relation_type: str
    polarity: str
    final_weight: float
    domain_conf: float
    semantic_score: float
    evidence_count: int = 0


@dataclass
class PathTrace:
    nodes: List[str]
    edges: List[EdgeTrace]
    path_weight: float
    sign_product: str


@dataclass
class ReasoningTrace:
    query_entities: Dict[str, str]
    candidate_paths: List[PathTrace]
    selected_path: Optional[PathTrace]
    as_of: datetime
    trace_id: str
