"""Validation models."""
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class SignTag(str, Enum):
    CONFIDENT = "confident"
    AMBIGUOUS = "ambiguous"
    SUSPECT = "suspect"
    UNKNOWN = "unknown"


class SemanticTag(str, Enum):
    SEM_CONFIDENT = "sem_confident"
    SEM_WEAK = "sem_weak"
    SEM_SPURIOUS = "sem_spurious"
    SEM_WRONG = "sem_wrong"
    SEM_AMBIGUOUS = "sem_ambiguous"


class ValidationDestination(str, Enum):
    DOMAIN_CANDIDATE = "domain_candidate"
    DROP_LOG = "drop_log"


class SchemaValidationResult(BaseModel):
    edge_id: str
    schema_valid: bool
    schema_errors: List[str] = Field(default_factory=list)

    has_required_fields: bool = True
    relation_type_valid: bool = True
    entity_pair_valid: bool = True
    no_self_loop: bool = True


class SignValidationResult(BaseModel):
    edge_id: str
    polarity_final: str
    sign_tag: SignTag
    sign_consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)

    pattern_polarity: Optional[str] = None
    domain_polarity: Optional[str] = None
    llm_polarity: Optional[str] = None
    conflict_with_static: bool = False


class SemanticValidationResult(BaseModel):
    edge_id: str
    semantic_tag: SemanticTag
    semantic_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    has_exaggeration: bool = False
    is_correlation_as_causation: bool = False
    has_weak_evidence: bool = False
    domain_conflict: bool = False
    llm_judgement: Optional[str] = None


class ValidationResult(BaseModel):
    """Final validation output."""
    edge_id: str
    validation_passed: bool
    destination: ValidationDestination

    combined_conf: float = Field(default=0.0, ge=0.0, le=1.0)
    student_conf: float = 0.0
    sign_score: float = 0.0
    semantic_conf: float = 0.0

    schema_result: Optional[SchemaValidationResult] = None
    sign_result: Optional[SignValidationResult] = None
    semantic_result: Optional[SemanticValidationResult] = None

    rejection_reason: Optional[str] = None
    rejection_details: List[str] = Field(default_factory=list)
