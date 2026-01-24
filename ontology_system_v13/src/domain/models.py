"""
Domain models for ontology management.
"""
from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class DomainAction(str, Enum):
    """Domain processing actions."""
    STRENGTHEN_STATIC = "strengthen_static_evidence"
    REJECT_TO_LOG = "reject_to_log"
    CREATE_NEW = "create_new_relation"
    UPDATE_EXISTING = "update_existing"


class DomainCandidate(BaseModel):
    """Normalized edge ready for domain evaluation."""
    candidate_id: str = Field(default_factory=lambda: generate_id("DC"))

    raw_edge_id: str
    head_canonical_id: str
    head_canonical_name: str
    tail_canonical_id: str
    tail_canonical_name: str

    relation_type: str
    polarity: str  # +, -, neutral

    semantic_tag: str
    combined_conf: float
    student_conf: float

    timestamp: datetime = Field(default_factory=datetime.now)
    freq_count: int = Field(default=1)
    evidence_source: str = Field(default="student")

    fragment_text: Optional[str] = None


class StaticGuardResult(BaseModel):
    """Static domain guard result."""
    candidate_id: str
    static_pass: bool
    static_conflict: bool = False
    action: DomainAction

    conflict_rule_id: Optional[str] = None
    expected_polarity: Optional[str] = None
    actual_polarity: Optional[str] = None
    conflict_reason: Optional[str] = None


class DynamicRelation(BaseModel):
    """Persisted dynamic relation."""
    relation_id: str = Field(default_factory=lambda: generate_id("DYN"))

    head_id: str
    head_name: str
    tail_id: str
    tail_name: str
    relation_type: str
    sign: str

    domain_conf: float = Field(default=0.5)
    evidence_count: int = Field(default=1)
    conflict_count: int = Field(default=0)

    created_at: datetime = Field(default_factory=datetime.now)
    last_update: datetime = Field(default_factory=datetime.now)

    origin: str = Field(default="student")
    semantic_tags: List[str] = Field(default_factory=list)
    decay_applied: bool = Field(default=False)


class DynamicUpdateResult(BaseModel):
    """Dynamic update result."""
    candidate_id: str
    relation_id: str
    action: DomainAction

    domain_conf: float
    evidence_count: int
    decayed: bool = False
    is_new: bool = False

    previous_conf: Optional[float] = None
    previous_evidence_count: Optional[int] = None


class DomainProcessResult(BaseModel):
    """Final processing result for a domain candidate."""
    candidate_id: str
    raw_edge_id: str

    final_destination: str  # "domain" | "log"

    intake_result: Optional[DomainCandidate] = None
    static_result: Optional[StaticGuardResult] = None
    dynamic_result: Optional[DynamicUpdateResult] = None

    domain_relation_id: Optional[str] = None
