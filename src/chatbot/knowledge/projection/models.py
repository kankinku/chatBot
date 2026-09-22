"""Deterministic contracts for scenario/regime knowledge projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from chatbot.knowledge.evidence import EvidenceScoreSummary
from chatbot.knowledge.workspace.hashing import hash_value
from chatbot.knowledge.workspace.models import SourceRef


PROJECTION_SCHEMA_VERSION = 1
PROJECTION_ENGINE_VERSION = "scenario-projection-v1"

RelationSign = Literal["+", "-", "neutral", "unknown"]
ShockDirection = Literal["+", "-"]


def _bounded(value: float, *, low: float, high: float, field_name: str) -> float:
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f"{field_name} must be within {low}..{high}")
    return value


def _require_sign(value: str, *, allow_unknown: bool = False) -> str:
    allowed = {"+", "-", "neutral"}
    if allow_unknown:
        allowed.add("unknown")
    if value not in allowed:
        raise ValueError(f"unsupported relation sign: {value}")
    return value


@dataclass(frozen=True)
class RelationSelector:
    relation_node_id: str | None = None
    head_id: str | None = None
    tail_id: str | None = None
    relation_type: str | None = None

    def __post_init__(self) -> None:
        node_mode = self.relation_node_id is not None
        key_values = (self.head_id, self.tail_id, self.relation_type)
        key_mode = all(value is not None for value in key_values)
        type_mode = (
            self.relation_type is not None
            and self.head_id is None
            and self.tail_id is None
        )
        if sum(bool(item) for item in (node_mode, key_mode, type_mode)) != 1:
            raise ValueError(
                "relation selector must use exactly one of relation_node_id, "
                "complete head/tail/type key, or relation_type-only"
            )

    def semantic_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "relation_node_id": self.relation_node_id,
                "head_id": self.head_id,
                "tail_id": self.tail_id,
                "relation_type": self.relation_type,
            }.items()
            if value is not None
        }

    def to_dict(self) -> dict[str, Any]:
        return self.semantic_dict()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationSelector":
        return cls(
            relation_node_id=value.get("relation_node_id"),
            head_id=value.get("head_id"),
            tail_id=value.get("tail_id"),
            relation_type=value.get("relation_type"),
        )

    def matches(
        self,
        *,
        relation_node_id: str | None,
        head_id: str,
        tail_id: str,
        relation_type: str,
    ) -> bool:
        if self.relation_node_id is not None:
            return relation_node_id == self.relation_node_id
        if self.head_id is not None:
            return (
                head_id == self.head_id
                and tail_id == self.tail_id
                and relation_type == self.relation_type
            )
        return relation_type == self.relation_type


@dataclass(frozen=True)
class RelationScaleAssumption:
    selector: RelationSelector
    multiplier: float
    kind: str = field(default="relation_scale", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "multiplier",
            _bounded(
                self.multiplier,
                low=0.0,
                high=2.0,
                field_name="scenario multiplier",
            ),
        )

    @property
    def assumption_id(self) -> str:
        return "asm_" + hash_value(self.semantic_dict())[:24]

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "selector": self.selector.semantic_dict(),
            "multiplier": self.multiplier,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "assumption_id": self.assumption_id}


@dataclass(frozen=True)
class RelationDisableAssumption:
    selector: RelationSelector
    kind: str = field(default="relation_disable", init=False)

    @property
    def assumption_id(self) -> str:
        return "asm_" + hash_value(self.semantic_dict())[:24]

    def semantic_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "selector": self.selector.semantic_dict()}

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "assumption_id": self.assumption_id}


@dataclass(frozen=True)
class RelationSignOverrideAssumption:
    selector: RelationSelector
    sign: str
    kind: str = field(default="relation_sign_override", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sign", _require_sign(self.sign))

    @property
    def assumption_id(self) -> str:
        return "asm_" + hash_value(self.semantic_dict())[:24]

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "selector": self.selector.semantic_dict(),
            "sign": self.sign,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "assumption_id": self.assumption_id}


@dataclass(frozen=True)
class InjectRelationAssumption:
    head_id: str
    tail_id: str
    relation_type: str
    sign: str
    assumed_weight: float
    kind: str = field(default="inject_relation", init=False)

    def __post_init__(self) -> None:
        if not self.head_id or not self.tail_id or not self.relation_type:
            raise ValueError("injected relation requires head, tail, and relation_type")
        object.__setattr__(self, "sign", _require_sign(self.sign))
        object.__setattr__(
            self,
            "assumed_weight",
            _bounded(
                self.assumed_weight,
                low=0.0,
                high=1.0,
                field_name="assumed_weight",
            ),
        )

    @property
    def stable_key(self) -> tuple[str, str, str]:
        return (self.head_id, self.tail_id, self.relation_type)

    @property
    def assumption_id(self) -> str:
        return "asm_" + hash_value(self.semantic_dict())[:24]

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "head_id": self.head_id,
            "tail_id": self.tail_id,
            "relation_type": self.relation_type,
            "sign": self.sign,
            "assumed_weight": self.assumed_weight,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "assumption_id": self.assumption_id}


ScenarioAssumption = (
    RelationScaleAssumption
    | RelationDisableAssumption
    | RelationSignOverrideAssumption
    | InjectRelationAssumption
)


def assumption_from_dict(value: dict[str, Any]) -> ScenarioAssumption:
    kind = value.get("kind")
    if kind == "relation_scale":
        return RelationScaleAssumption(
            selector=RelationSelector.from_dict(value["selector"]),
            multiplier=value["multiplier"],
        )
    if kind == "relation_disable":
        return RelationDisableAssumption(
            selector=RelationSelector.from_dict(value["selector"]),
        )
    if kind == "relation_sign_override":
        return RelationSignOverrideAssumption(
            selector=RelationSelector.from_dict(value["selector"]),
            sign=value["sign"],
        )
    if kind == "inject_relation":
        return InjectRelationAssumption(
            head_id=value["head_id"],
            tail_id=value["tail_id"],
            relation_type=value["relation_type"],
            sign=value["sign"],
            assumed_weight=value["assumed_weight"],
        )
    raise ValueError(f"unsupported scenario assumption kind: {kind}")


@dataclass(frozen=True)
class ScenarioShock:
    target_entity_id: str
    direction: str
    magnitude: float

    def __post_init__(self) -> None:
        if not self.target_entity_id:
            raise ValueError("shock target entity must not be empty")
        if self.direction not in {"+", "-"}:
            raise ValueError("shock direction must be + or -")
        object.__setattr__(
            self,
            "magnitude",
            _bounded(
                self.magnitude,
                low=0.0,
                high=1.0,
                field_name="shock magnitude",
            ),
        )

    @property
    def shock_id(self) -> str:
        return "shk_" + hash_value(self.semantic_dict())[:24]

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "target_entity_id": self.target_entity_id,
            "direction": self.direction,
            "magnitude": self.magnitude,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "shock_id": self.shock_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScenarioShock":
        return cls(
            target_entity_id=value["target_entity_id"],
            direction=value["direction"],
            magnitude=value["magnitude"],
        )


@dataclass(frozen=True)
class SensitivityThreshold:
    selector: RelationSelector
    operator: str
    threshold: float

    def __post_init__(self) -> None:
        if self.operator not in {"<", "<=", ">", ">="}:
            raise ValueError("unsupported sensitivity operator")
        object.__setattr__(
            self,
            "threshold",
            _bounded(
                self.threshold,
                low=0.0,
                high=1.0,
                field_name="sensitivity threshold",
            ),
        )

    @property
    def threshold_id(self) -> str:
        return "thr_" + hash_value(self.semantic_dict())[:24]

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector.semantic_dict(),
            "operator": self.operator,
            "threshold": self.threshold,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "threshold_id": self.threshold_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SensitivityThreshold":
        return cls(
            selector=RelationSelector.from_dict(value["selector"]),
            operator=value["operator"],
            threshold=value["threshold"],
        )


@dataclass(frozen=True)
class ScenarioSpec:
    assumptions: tuple[ScenarioAssumption, ...] = ()
    shocks: tuple[ScenarioShock, ...] = ()
    sensitivity_thresholds: tuple[SensitivityThreshold, ...] = ()
    max_depth: int = 4
    max_paths: int = 100
    label: str | None = None
    description: str | None = None
    schema_version: int = PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROJECTION_SCHEMA_VERSION:
            raise ValueError("unsupported scenario schema")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.max_paths < 1:
            raise ValueError("max_paths must be positive")
        assumption_ids = [item.assumption_id for item in self.assumptions]
        if len(set(assumption_ids)) != len(assumption_ids):
            raise ValueError("duplicate scenario assumption")
        shock_ids = [item.shock_id for item in self.shocks]
        if len(set(shock_ids)) != len(shock_ids):
            raise ValueError("duplicate scenario shock")
        object.__setattr__(
            self,
            "assumptions",
            tuple(sorted(self.assumptions, key=lambda item: item.assumption_id)),
        )
        object.__setattr__(
            self,
            "shocks",
            tuple(sorted(self.shocks, key=lambda item: item.shock_id)),
        )
        object.__setattr__(
            self,
            "sensitivity_thresholds",
            tuple(
                sorted(
                    self.sensitivity_thresholds,
                    key=lambda item: item.threshold_id,
                )
            ),
        )

    @property
    def scenario_spec_id(self) -> str:
        return "scn_" + hash_value(self.semantic_dict())

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assumptions": [item.semantic_dict() for item in self.assumptions],
            "shocks": [item.semantic_dict() for item in self.shocks],
            "sensitivity_thresholds": [
                item.semantic_dict() for item in self.sensitivity_thresholds
            ],
            "max_depth": self.max_depth,
            "max_paths": self.max_paths,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_dict(),
            "scenario_spec_id": self.scenario_spec_id,
            "label": self.label,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScenarioSpec":
        return cls(
            assumptions=tuple(
                assumption_from_dict(item)
                for item in value.get("assumptions", [])
            ),
            shocks=tuple(
                ScenarioShock.from_dict(item)
                for item in value.get("shocks", [])
            ),
            sensitivity_thresholds=tuple(
                SensitivityThreshold.from_dict(item)
                for item in value.get("sensitivity_thresholds", [])
            ),
            max_depth=int(value.get("max_depth", 4)),
            max_paths=int(value.get("max_paths", 100)),
            label=value.get("label"),
            description=value.get("description"),
            schema_version=int(
                value.get("schema_version", PROJECTION_SCHEMA_VERSION)
            ),
        )


@dataclass(frozen=True)
class RegimeRule:
    selector: RelationSelector
    multiplier: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "multiplier",
            _bounded(
                self.multiplier,
                low=0.0,
                high=2.0,
                field_name="regime multiplier",
            ),
        )

    @property
    def rule_id(self) -> str:
        return "rgr_" + hash_value(self.semantic_dict())[:24]

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector.semantic_dict(),
            "multiplier": self.multiplier,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "rule_id": self.rule_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RegimeRule":
        return cls(
            selector=RelationSelector.from_dict(value["selector"]),
            multiplier=value["multiplier"],
        )


@dataclass(frozen=True)
class RegimeSpec:
    rules: tuple[RegimeRule, ...] = ()
    label: str | None = None
    description: str | None = None
    schema_version: int = PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROJECTION_SCHEMA_VERSION:
            raise ValueError("unsupported regime schema")
        rule_ids = [item.rule_id for item in self.rules]
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("duplicate regime rule")
        object.__setattr__(
            self,
            "rules",
            tuple(sorted(self.rules, key=lambda item: item.rule_id)),
        )

    @property
    def regime_spec_id(self) -> str:
        return "rgm_" + hash_value(self.semantic_dict())

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rules": [item.semantic_dict() for item in self.rules],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_dict(),
            "regime_spec_id": self.regime_spec_id,
            "label": self.label,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RegimeSpec":
        return cls(
            rules=tuple(
                RegimeRule.from_dict(item)
                for item in value.get("rules", [])
            ),
            label=value.get("label"),
            description=value.get("description"),
            schema_version=int(
                value.get("schema_version", PROJECTION_SCHEMA_VERSION)
            ),
        )


@dataclass(frozen=True)
class ProjectionBase:
    state_digest: str
    origin: str
    snapshot_id: str | None = None
    committed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.origin not in {"replay_snapshot", "current_state"}:
            raise ValueError("unsupported projection base origin")
        if len(self.state_digest) != 64:
            raise ValueError("projection base state_digest must be SHA-256")
        try:
            int(self.state_digest, 16)
        except ValueError as exc:
            raise ValueError(
                "projection base state_digest must be SHA-256"
            ) from exc

        if self.origin == "replay_snapshot":
            if self.snapshot_id is None:
                raise ValueError("replay projection base requires snapshot_id")
            if (
                self.committed_at is None
                or self.committed_at.tzinfo is None
                or self.committed_at.utcoffset() is None
            ):
                raise ValueError(
                    "replay projection base requires timezone-aware committed_at"
                )
            object.__setattr__(
                self,
                "committed_at",
                self.committed_at.astimezone(timezone.utc),
            )
        else:
            if self.snapshot_id is not None:
                raise ValueError(
                    "current projection base must not carry snapshot_id"
                )
            if self.committed_at is not None:
                raise ValueError(
                    "current projection base must not carry committed_at"
                )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "state_digest": self.state_digest,
            "snapshot_id": self.snapshot_id,
            "origin": self.origin,
            "committed_at": (
                self.committed_at.isoformat().replace("+00:00", "Z")
                if self.committed_at is not None
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.identity_dict()


def score_summary_to_dict(
    value: EvidenceScoreSummary | None,
) -> dict[str, Any] | None:
    return asdict(value) if value is not None else None


@dataclass(frozen=True)
class ProjectedRelation:
    relation_id: str
    head_id: str
    head_name: str
    tail_id: str
    tail_name: str
    relation_type: str
    origin: str
    base_relation_node_id: str | None
    base_sign: str | None
    projected_sign: str
    evidence_score: EvidenceScoreSummary | None
    base_weight: float
    regime_multiplier: float
    scenario_multiplier: float
    active: bool
    projected_weight: float
    applied_regime_rule_ids: tuple[str, ...] = ()
    applied_assumption_ids: tuple[str, ...] = ()
    support_assertion_ids: tuple[str, ...] = ()
    conflict_assertion_ids: tuple[str, ...] = ()
    source_refs: tuple[SourceRef, ...] = ()

    @property
    def stable_key(self) -> tuple[str, str, str]:
        return (self.head_id, self.tail_id, self.relation_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "head_id": self.head_id,
            "head_name": self.head_name,
            "tail_id": self.tail_id,
            "tail_name": self.tail_name,
            "relation_type": self.relation_type,
            "origin": self.origin,
            "base_relation_node_id": self.base_relation_node_id,
            "base_sign": self.base_sign,
            "projected_sign": self.projected_sign,
            "evidence_score": score_summary_to_dict(self.evidence_score),
            "base_weight": self.base_weight,
            "regime_multiplier": self.regime_multiplier,
            "scenario_multiplier": self.scenario_multiplier,
            "active": self.active,
            "projected_weight": self.projected_weight,
            "applied_regime_rule_ids": list(self.applied_regime_rule_ids),
            "applied_assumption_ids": list(self.applied_assumption_ids),
            "support_assertion_ids": list(self.support_assertion_ids),
            "conflict_assertion_ids": list(self.conflict_assertion_ids),
            "source_refs": [item.to_dict() for item in self.source_refs],
        }


@dataclass(frozen=True)
class ProjectedImpact:
    shock_id: str
    target_entity_id: str
    direction: str
    impact_value: float
    path_entities: tuple[str, ...]
    path_relation_ids: tuple[str, ...]
    applied_rule_ids: tuple[str, ...] = ()
    applied_assumption_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shock_id": self.shock_id,
            "target_entity_id": self.target_entity_id,
            "direction": self.direction,
            "impact_value": self.impact_value,
            "path_entities": list(self.path_entities),
            "path_relation_ids": list(self.path_relation_ids),
            "applied_rule_ids": list(self.applied_rule_ids),
            "applied_assumption_ids": list(self.applied_assumption_ids),
        }


@dataclass(frozen=True)
class NodeImpactSummary:
    entity_id: str
    strongest_positive: float
    strongest_negative: float
    net_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectionDependency:
    kind: str
    relation_id: str | None
    input_id: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SensitivityResult:
    threshold_id: str
    relation_id: str
    projected_weight: float
    operator: str
    threshold: float
    triggered: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectionTrace:
    projection_id: str
    engine_version: str
    base: ProjectionBase
    scenario_spec_id: str
    regime_spec_id: str | None
    evidence_score_versions: tuple[str, ...]
    dependencies: tuple[ProjectionDependency, ...]
    warnings: tuple[str, ...]
    output_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "engine_version": self.engine_version,
            "base": self.base.to_dict(),
            "scenario_spec_id": self.scenario_spec_id,
            "regime_spec_id": self.regime_spec_id,
            "evidence_score_versions": list(self.evidence_score_versions),
            "dependencies": [item.to_dict() for item in self.dependencies],
            "warnings": list(self.warnings),
            "output_digest": self.output_digest,
        }


@dataclass(frozen=True)
class ScenarioProjection:
    projection_id: str
    engine_version: str
    base: ProjectionBase
    scenario_spec_id: str
    regime_spec_id: str | None
    relations: tuple[ProjectedRelation, ...]
    impacts: tuple[ProjectedImpact, ...]
    node_summaries: tuple[NodeImpactSummary, ...]
    sensitivity: tuple[SensitivityResult, ...]
    trace: ProjectionTrace
    output_digest: str
    schema_version: int = PROJECTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "projection_id": self.projection_id,
            "engine_version": self.engine_version,
            "base": self.base.to_dict(),
            "scenario_spec_id": self.scenario_spec_id,
            "regime_spec_id": self.regime_spec_id,
            "relations": [item.to_dict() for item in self.relations],
            "impacts": [item.to_dict() for item in self.impacts],
            "node_summaries": [item.to_dict() for item in self.node_summaries],
            "sensitivity": [item.to_dict() for item in self.sensitivity],
            "trace": self.trace.to_dict(),
            "output_digest": self.output_digest,
        }
