"""
Shared models for extraction, validation, and reasoning pipelines.
"""
from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


def generate_id(prefix: str) -> str:
    """Generate a short unique ID with a prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class QualityTag(str, Enum):
    """Fragment quality label."""
    INFORMATIVE = "informative"
    NOISY = "noisy"
    UNCLEAR = "unclear"
    EMOTIONAL = "emotional"
    INCOMPLETE = "incomplete"


class ResolutionMode(str, Enum):
    """Entity resolution outcome."""
    DICTIONARY_MATCH = "dictionary_match"
    STATIC_DOMAIN = "static_domain"
    DYNAMIC_DOMAIN = "dynamic_domain"
    CUSTOM_ALIAS = "custom_alias"
    FUZZY_MATCH = "fuzzy_match"
    AMBIGUOUS = "ambiguous"
    NEW_ENTITY = "new_entity"


class Polarity(str, Enum):
    """Relation polarity."""
    POSITIVE = "+"
    NEGATIVE = "-"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class Fragment(BaseModel):
    """Minimal fragment extracted from raw text."""
    fragment_id: str = Field(default_factory=lambda: generate_id("F"))
    text: str = Field(..., description="Fragment text.")
    doc_id: str = Field(..., description="Document ID.")
    timestamp: datetime = Field(default_factory=datetime.now)
    quality_tag: QualityTag = Field(default=QualityTag.INFORMATIVE)

    source_start: Optional[int] = Field(default=None, description="Start offset in source text.")
    source_end: Optional[int] = Field(default=None, description="End offset in source text.")

    class Config:
        use_enum_values = True


class EntityCandidate(BaseModel):
    """Raw entity candidate extracted from a fragment."""
    entity_id: str = Field(default_factory=lambda: generate_id("E_temp"))
    surface_text: str = Field(..., description="Surface string from the fragment.")
    type_guess: str = Field(..., description="Guessed entity type.")
    normalized_name_guess: Optional[str] = Field(default=None, description="Guessed normalized name.")

    span_start: int = Field(..., description="Start offset in fragment.")
    span_end: int = Field(..., description="End offset in fragment.")

    student_conf: float = Field(default=0.0, ge=0.0, le=1.0)
    fragment_id: str = Field(..., description="Origin fragment ID.")


class ResolvedEntity(BaseModel):
    """Entity candidate resolved to a canonical entity."""
    entity_id: str = Field(..., description="Original temporary entity ID.")

    canonical_id: Optional[str] = Field(default=None, description="Canonical entity ID.")
    canonical_name: Optional[str] = Field(default=None, description="Canonical name.")
    canonical_type: Optional[str] = Field(default=None, description="Canonical type.")

    resolution_mode: ResolutionMode
    resolution_conf: float = Field(default=0.0, ge=0.0, le=1.0)

    candidate_ids: Optional[List[str]] = Field(default=None)
    candidate_confs: Optional[List[float]] = Field(default=None)

    is_new_entity_candidate: bool = Field(default=False)

    surface_text: str = Field(..., description="Original surface text.")
    fragment_id: str = Field(..., description="Origin fragment ID.")


class RawEdge(BaseModel):
    """Raw relation extracted from a fragment."""
    raw_edge_id: str = Field(default_factory=lambda: generate_id("R"))

    head_entity_id: str = Field(..., description="Head entity ID.")
    head_canonical_name: Optional[str] = Field(default=None)
    tail_entity_id: str = Field(..., description="Tail entity ID.")
    tail_canonical_name: Optional[str] = Field(default=None)

    relation_type: str = Field(..., description="Relation type.")
    polarity_guess: Polarity = Field(default=Polarity.UNKNOWN)

    student_conf: float = Field(default=0.0, ge=0.0, le=1.0)

    condition_text: Optional[str] = Field(default=None, description="Conditional context.")

    fragment_id: str = Field(..., description="Origin fragment ID.")
    fragment_text: Optional[str] = Field(default=None, description="Fragment text.")

    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class ExtractionResult(BaseModel):
    """Full extraction result for a document."""
    doc_id: str
    fragments: List[Fragment]
    entity_candidates: List[EntityCandidate]
    resolved_entities: List[ResolvedEntity]
    raw_edges: List[RawEdge]

    processing_time_ms: float = Field(default=0.0)
    error_count: int = Field(default=0)
    warning_messages: List[str] = Field(default_factory=list)
