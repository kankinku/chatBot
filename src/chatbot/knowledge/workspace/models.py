"""Deterministic relationship/provenance workspace models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = 1


class NodeKind(str, Enum):
    SOURCE = "source"
    CONFIG = "config"
    ENTITY_TYPE = "entity_type"
    RELATION_TYPE = "relation_type"
    ENTITY = "entity"
    ENTITY_REF = "entity_ref"
    DOCUMENT = "document"
    ASSERTION = "assertion"
    SYSTEM = "system"


class MetaRelation(str, Enum):
    """Closed meta-edge vocabulary.

    Domain semantic relations such as Affect and Cause are carried by
    semantic_type on DOMAIN_RELATION edges rather than expanding this set.
    """

    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    PRODUCES = "produces"
    CONFIGURES = "configures"
    VALIDATES = "validates"
    DERIVED_FROM = "derived_from"
    SUPPORTED_BY = "supported_by"
    CONTRADICTED_BY = "contradicted_by"
    DOMAIN_RELATION = "domain_relation"


class Confidence(str, Enum):
    DECLARED = "declared"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    DERIVED = "derived"


@dataclass(frozen=True, order=True)
class SourceRef:
    path: str
    hash: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "hash": self.hash}


@dataclass
class WorkspaceNode:
    id: str
    label: str
    kind: NodeKind
    content_hash: str
    sources: list[SourceRef] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind.value,
            "content_hash": self.content_hash,
            "sources": [source.to_dict() for source in sorted(self.sources)],
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkspaceNode":
        return cls(
            id=value["id"],
            label=value["label"],
            kind=NodeKind(value["kind"]),
            content_hash=value["content_hash"],
            sources=[SourceRef(**item) for item in value.get("sources", [])],
            properties=value.get("properties", {}),
        )


@dataclass
class WorkspaceEdge:
    source: str
    target: str
    relation: MetaRelation
    confidence: Confidence
    content_hash: str
    sources: list[SourceRef] = field(default_factory=list)
    semantic_type: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (
            self.source,
            self.relation.value,
            self.target,
            self.semantic_type or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
            "confidence": self.confidence.value,
            "content_hash": self.content_hash,
            "sources": [source.to_dict() for source in sorted(self.sources)],
            "semantic_type": self.semantic_type,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkspaceEdge":
        return cls(
            source=value["source"],
            target=value["target"],
            relation=MetaRelation(value["relation"]),
            confidence=Confidence(value["confidence"]),
            content_hash=value["content_hash"],
            sources=[SourceRef(**item) for item in value.get("sources", [])],
            semantic_type=value.get("semantic_type"),
            properties=value.get("properties", {}),
        )


@dataclass
class WorkspaceGraph:
    nodes: list[WorkspaceNode]
    edges: list[WorkspaceEdge]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in sorted(self.nodes, key=lambda item: item.id)],
            "edges": [
                edge.to_dict()
                for edge in sorted(self.edges, key=lambda item: item.identity)
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkspaceGraph":
        return cls(
            schema_version=value["schema_version"],
            nodes=[WorkspaceNode.from_dict(item) for item in value.get("nodes", [])],
            edges=[WorkspaceEdge.from_dict(item) for item in value.get("edges", [])],
        )


@dataclass
class DriftReport:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    generator_changed: bool = False

    @property
    def clean(self) -> bool:
        return not (
            self.added
            or self.changed
            or self.removed
            or self.generator_changed
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
