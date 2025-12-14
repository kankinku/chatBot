from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Delta Report ---
@dataclass
class ScalarDeltaItem:
    id: str  # series_id or entity_id
    feature_name: str
    prev_value: Optional[float]
    curr_value: Optional[float]
    delta_val: Optional[float]
    delta_pct: Optional[float]
    severity: str  # LOW, MEDIUM, HIGH

@dataclass
class TopologyDeltaItem:
    change_type: str  # NODE_ADDED, NODE_REMOVED, EDGE_ADDED, EDGE_REMOVED, HUB_CHANGED
    target_id: str
    details: str

@dataclass
class DeltaReport:
    scalar_items: List[ScalarDeltaItem] = field(default_factory=list)
    topology_items: List[TopologyDeltaItem] = field(default_factory=list)
    summary: str = ""

# --- Chain, Evidence, Conclusion, Break ---

@dataclass
class ChainLink:
    source_entity: str
    target_entity: str
    relation: str
    weight: float
    sign: str

@dataclass
class ChainReport:
    links: List[ChainLink] = field(default_factory=list)
    description: str = ""

@dataclass
class EvidenceItem:
    support_type: str  # POSITIVE, NEGATIVE
    score: float
    description: str
    source_id: str

@dataclass
class EvidenceReport:
    positive_items: List[EvidenceItem] = field(default_factory=list)
    negative_items: List[EvidenceItem] = field(default_factory=list)
    total_score: float = 0.0

@dataclass
class ConclusionReport:
    direction: str  # UP, DOWN, FLAT
    confidence: float
    text: str
    rationale: str

@dataclass
class BreakCondition:
    trigger_type: str  # EDGE_WEIGHT, EVIDENCE_DROP, FEATURE_REVERSAL
    threshold: float
    current_value: float
    target_component: str
    rationale: str

@dataclass
class BreakConditionsReport:
    conditions: List[BreakCondition] = field(default_factory=list)

@dataclass
class MetaReport:
    policy_id: str
    policy_version: str
    as_of: datetime
    snapshot_id: Optional[str] = None
    seed: Optional[int] = None
    trace_id: Optional[str] = None

# --- Main Scenario Report ---
@dataclass
class ScenarioReport:
    delta: DeltaReport
    chain: ChainReport
    evidence: EvidenceReport
    conclusion: ConclusionReport
    break_conditions: BreakConditionsReport
    meta: MetaReport
