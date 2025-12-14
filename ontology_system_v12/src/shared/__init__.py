# shared package
from .models import (
    Fragment,
    EntityCandidate,
    ResolvedEntity,
    RawEdge,
    QualityTag,
    ResolutionMode,
    Polarity,
)
from .exceptions import (
    ExtractionError,
    FragmentExtractionError,
    NERError,
    EntityResolutionError,
    RelationExtractionError,
    LLMError,
    ConfigError,
)
from .schemas import (
    # Enums
    RegimeType,
    EvidenceDirection,
    FeatureType,
    SourceType,
    DeltaMethod,
    # Time Series
    Observation,
    TimeSeriesMetadata,
    # Event
    Event,
    # Feature
    FeatureSpec,
    FeatureValue,
    # Evidence
    EvidenceSpec,
    EvidenceScore,
    AccumulatedEvidence,
    # Regime
    RegimeCondition,
    RegimeSpec,
    RegimeDetectionResult,
    # Source
    SourceSpec,
    FetchState,
    # Dependency
    DependencyEdge,
    DependencyGraph,
)

__all__ = [
    # Models
    "Fragment",
    "EntityCandidate",
    "ResolvedEntity",
    "RawEdge",
    "QualityTag",
    "ResolutionMode",
    "Polarity",
    # Exceptions
    "ExtractionError",
    "FragmentExtractionError",
    "NERError",
    "EntityResolutionError",
    "RelationExtractionError",
    "LLMError",
    "ConfigError",
    # Schemas - Enums
    "RegimeType",
    "EvidenceDirection",
    "FeatureType",
    "SourceType",
    "DeltaMethod",
    # Schemas - Time Series
    "Observation",
    "TimeSeriesMetadata",
    # Schemas - Event
    "Event",
    # Schemas - Feature
    "FeatureSpec",
    "FeatureValue",
    # Schemas - Evidence
    "EvidenceSpec",
    "EvidenceScore",
    "AccumulatedEvidence",
    # Schemas - Regime
    "RegimeCondition",
    "RegimeSpec",
    "RegimeDetectionResult",
    # Schemas - Source
    "SourceSpec",
    "FetchState",
    # Schemas - Dependency
    "DependencyEdge",
    "DependencyGraph",
]
