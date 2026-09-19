"""Evidence provenance models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chatbot.knowledge.domain.models import DomainProcessResult
from chatbot.knowledge.shared.models import ExtractionResult
from chatbot.knowledge.validation.models import ValidationResult
from chatbot.knowledge.workspace.models import WorkspaceGraph


@dataclass
class EvidenceProjection:
    source_uri: str
    source_hash: str
    document_node_id: str
    projection_hash: str
    graph: WorkspaceGraph

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "source_hash": self.source_hash,
            "document_node_id": self.document_node_id,
            "projection_hash": self.projection_hash,
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceProjection":
        return cls(
            source_uri=value["source_uri"],
            source_hash=value["source_hash"],
            document_node_id=value["document_node_id"],
            projection_hash=value["projection_hash"],
            graph=WorkspaceGraph.from_dict(value["graph"]),
        )


@dataclass
class InvalidationPlan:
    source_uri: str
    previous_source_hash: str | None
    current_source_hash: str | None
    invalidated_node_ids: list[str] = field(default_factory=list)
    invalidated_edge_ids: list[str] = field(default_factory=list)
    affected_relation_ids: list[str] = field(default_factory=list)
    reason: str = "no_change"

    @property
    def changed(self) -> bool:
        return self.reason != "no_change"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "previous_source_hash": self.previous_source_hash,
            "current_source_hash": self.current_source_hash,
            "invalidated_node_ids": self.invalidated_node_ids,
            "invalidated_edge_ids": self.invalidated_edge_ids,
            "affected_relation_ids": self.affected_relation_ids,
            "reason": self.reason,
        }


@dataclass
class EvidenceProcessResult:
    extraction: ExtractionResult
    validation_results: list[ValidationResult]
    domain_results: list[DomainProcessResult]
    projection: EvidenceProjection
