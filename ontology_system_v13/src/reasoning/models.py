"""Reasoning models."""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class QueryType(str, Enum):
    DIRECT_RELATION = "direct_relation"
    CONDITIONED = "conditioned"
    CAUSAL = "causal"
    PREDICTIVE = "predictive"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


class ReasoningDirection(str, Enum):
    POSITIVE = "+"
    NEGATIVE = "-"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class ParsedQuery(BaseModel):
    query_id: str = Field(default_factory=lambda: generate_id("Q"))
    original_query: str

    query_entities: List[str] = Field(default_factory=list)
    entity_names: Dict[str, str] = Field(default_factory=dict)

    query_type: QueryType = QueryType.UNKNOWN

    head_entity: Optional[str] = None
    tail_entity: Optional[str] = None
    condition_entities: List[str] = Field(default_factory=list)

    fragments: List[str] = Field(default_factory=list)


class RetrievedPath(BaseModel):
    path_id: str = Field(default_factory=lambda: generate_id("PATH"))
    nodes: List[str]
    node_names: List[str]
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    source: str = "domain"
    path_length: int = 0


class RetrievalResult(BaseModel):
    query_id: str
    direct_paths: List[RetrievedPath] = Field(default_factory=list)
    indirect_paths: List[RetrievedPath] = Field(default_factory=list)
    domain_paths_count: int = 0
    total_edges_retrieved: int = 0


class FusedEdge(BaseModel):
    edge_id: str
    head_id: str
    tail_id: str
    relation_type: str
    sign: str

    final_weight: float = 0.0
    domain_conf: float = 0.0
    decay_factor: float = 0.0
    semantic_score: float = 0.0
    evidence_count: int = 0


class FusedPath(BaseModel):
    path_id: str
    nodes: List[str]
    fused_edges: List[FusedEdge] = Field(default_factory=list)
    path_weight: float = 0.0
    path_sign: str = "+"


class PathReasoningResult(BaseModel):
    path_id: str
    nodes: List[str]
    node_names: List[str]
    combined_sign: str
    path_strength: float
    edge_signs: List[str] = Field(default_factory=list)
    edge_weights: List[float] = Field(default_factory=list)


class ReasoningResult(BaseModel):
    query_id: str
    direction: ReasoningDirection
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    paths_used: List[PathReasoningResult] = Field(default_factory=list)
    strongest_path: Optional[PathReasoningResult] = None
    positive_evidence: float = 0.0
    negative_evidence: float = 0.0
    conflicting_paths: int = 0


class ReasoningConclusion(BaseModel):
    query_id: str
    original_query: str
    conclusion_text: str
    explanation_text: str
    direction: ReasoningDirection
    confidence: float
    strongest_path_description: str
    evidence_summary: str
    reasoning_result: Optional[ReasoningResult] = None
