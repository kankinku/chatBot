from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class EdgeTrace:
    head_id: str
    tail_id: str
    relation_type: str
    polarity: str  # "+", "-", "0"
    final_weight: float
    domain_conf: float
    pcs: float
    semantic_score: float
    conflict_flag: bool = False
    drift_flag: bool = False
    supporting_fragment_ids: List[str] = field(default_factory=list)

@dataclass
class PathTrace:
    nodes: List[str]
    edges: List[EdgeTrace]
    path_weight: float
    sign_product: str
    why_selected: Optional[str] = None

@dataclass
class ReasoningTrace:
    query_entities: Dict[str, str]
    candidate_paths: List[PathTrace]
    selected_path: Optional[PathTrace]
    policy_id: Optional[str]
    as_of: datetime
    trace_id: str
    snapshot_id: Optional[str] = None
    reward: Optional[float] = None
    reward_breakdown: Optional[Dict] = None
