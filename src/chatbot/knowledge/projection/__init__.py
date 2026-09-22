"""Read-only deterministic scenario/regime projection layer."""

from .base_view import ProjectionBaseResolver, ProjectionBaseState
from .engine import ScenarioProjectionEngine
from .models import (
    PROJECTION_ENGINE_VERSION,
    PROJECTION_SCHEMA_VERSION,
    InjectRelationAssumption,
    NodeImpactSummary,
    ProjectedImpact,
    ProjectedRelation,
    ProjectionBase,
    ProjectionDependency,
    ProjectionTrace,
    RegimeRule,
    RegimeSpec,
    RelationDisableAssumption,
    RelationScaleAssumption,
    RelationSelector,
    RelationSignOverrideAssumption,
    ScenarioProjection,
    ScenarioShock,
    ScenarioSpec,
    SensitivityResult,
    SensitivityThreshold,
)
from .relation_provider import (
    DynamicRelationProvider,
    ProjectedRelationProvider,
    RelationProvider,
)
from .service import ProjectionService
from .store import ProjectionStore

__all__ = [
    "PROJECTION_ENGINE_VERSION",
    "PROJECTION_SCHEMA_VERSION",
    "DynamicRelationProvider",
    "InjectRelationAssumption",
    "NodeImpactSummary",
    "ProjectedImpact",
    "ProjectedRelation",
    "ProjectedRelationProvider",
    "ProjectionBase",
    "ProjectionBaseResolver",
    "ProjectionBaseState",
    "ProjectionDependency",
    "ProjectionService",
    "ProjectionStore",
    "ProjectionTrace",
    "RegimeRule",
    "RegimeSpec",
    "RelationDisableAssumption",
    "RelationProvider",
    "RelationScaleAssumption",
    "RelationSelector",
    "RelationSignOverrideAssumption",
    "ScenarioProjection",
    "ScenarioProjectionEngine",
    "ScenarioShock",
    "ScenarioSpec",
    "SensitivityResult",
    "SensitivityThreshold",
]
