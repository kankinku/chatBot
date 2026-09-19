"""Source-scoped evidence projections and invalidation planning."""

from __future__ import annotations

import json
from pathlib import Path

from chatbot.knowledge.workspace.hashing import hash_value
from chatbot.knowledge.workspace.impact import blast_radius
from chatbot.knowledge.workspace.invariants import validate_graph
from chatbot.knowledge.workspace.models import (
    NodeKind,
    SourceRef,
    WorkspaceEdge,
    WorkspaceGraph,
    WorkspaceNode,
)

from .models import EvidenceProjection, InvalidationPlan


def _merge_sources(*groups: list[SourceRef]) -> list[SourceRef]:
    merged = {
        (source.path, source.hash): source
        for group in groups
        for source in group
    }
    return [merged[key] for key in sorted(merged)]


def _edge_id(edge: WorkspaceEdge) -> str:
    source, relation, target, semantic = edge.identity
    return "|".join((source, relation, target, semantic))


class EvidenceLedger:
    """Keep one current derived projection per logical source."""

    def __init__(self):
        self._projections: dict[str, EvidenceProjection] = {}

    def get(self, source_uri: str) -> EvidenceProjection | None:
        return self._projections.get(source_uri)

    def sources(self) -> list[str]:
        return sorted(self._projections)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "projections": [
                self._projections[source_uri].to_dict()
                for source_uri in sorted(self._projections)
            ],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "EvidenceLedger":
        if value.get("schema_version") != 1:
            raise ValueError("unsupported evidence ledger schema")
        ledger = cls()
        for item in value.get("projections", []):
            projection = EvidenceProjection.from_dict(item)
            if projection.source_uri in ledger._projections:
                raise ValueError(
                    f"duplicate evidence source: {projection.source_uri}"
                )
            validate_graph(projection.graph)
            if projection.projection_hash != hash_value(
                projection.graph.to_dict()
            ):
                raise ValueError(
                    f"evidence projection hash mismatch: {projection.source_uri}"
                )
            document = next(
                (
                    node
                    for node in projection.graph.nodes
                    if node.id == projection.document_node_id
                ),
                None,
            )
            if (
                document is None
                or document.kind != NodeKind.DOCUMENT
                or document.content_hash != projection.source_hash
            ):
                raise ValueError(
                    f"invalid evidence document root: {projection.source_uri}"
                )
            ledger._projections[projection.source_uri] = projection
        return ledger

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.tmp")
        temp.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceLedger":
        target = Path(path)
        if not target.exists():
            return cls()
        return cls.from_dict(
            json.loads(target.read_text(encoding="utf-8"))
        )

    def upsert(self, projection: EvidenceProjection) -> InvalidationPlan:
        previous = self._projections.get(projection.source_uri)

        if (
            previous is not None
            and previous.source_hash == projection.source_hash
            and previous.projection_hash == projection.projection_hash
        ):
            self._projections[projection.source_uri] = projection
            return InvalidationPlan(
                source_uri=projection.source_uri,
                previous_source_hash=previous.source_hash,
                current_source_hash=projection.source_hash,
                reason="no_change",
            )

        plan = self._invalidation_plan(
            previous,
            current_source_hash=projection.source_hash,
            reason=(
                "source_changed"
                if previous is not None
                and previous.source_hash != projection.source_hash
                else "projection_changed"
                if previous is not None
                else "source_added"
            ),
            source_uri=projection.source_uri,
        )
        self._projections[projection.source_uri] = projection
        return plan

    def remove(self, source_uri: str) -> InvalidationPlan:
        previous = self._projections.pop(source_uri, None)
        if previous is None:
            return InvalidationPlan(
                source_uri=source_uri,
                previous_source_hash=None,
                current_source_hash=None,
                reason="no_change",
            )
        return self._invalidation_plan(
            previous,
            current_source_hash=None,
            reason="source_removed",
            source_uri=source_uri,
        )

    def merged_graph(self) -> WorkspaceGraph:
        node_map: dict[str, WorkspaceNode] = {}
        edge_map: dict[tuple[str, str, str, str], WorkspaceEdge] = {}

        for source_uri in sorted(self._projections):
            graph = self._projections[source_uri].graph
            for node in graph.nodes:
                current = node_map.get(node.id)
                if current is None:
                    node_map[node.id] = WorkspaceNode.from_dict(
                        node.to_dict()
                    )
                    continue
                if (
                    current.kind != node.kind
                    or current.content_hash != node.content_hash
                    or current.properties != node.properties
                ):
                    raise ValueError(
                        f"conflicting evidence node across sources: {node.id}"
                    )
                current.sources = _merge_sources(
                    current.sources,
                    node.sources,
                )

            for edge in graph.edges:
                current = edge_map.get(edge.identity)
                if current is None:
                    edge_map[edge.identity] = WorkspaceEdge.from_dict(
                        edge.to_dict()
                    )
                    continue
                if (
                    current.content_hash != edge.content_hash
                    or current.confidence != edge.confidence
                    or current.properties != edge.properties
                ):
                    raise ValueError(
                        "conflicting evidence edge across sources: "
                        f"{edge.identity}"
                    )
                current.sources = _merge_sources(
                    current.sources,
                    edge.sources,
                )

        graph = WorkspaceGraph(
            nodes=[node_map[key] for key in sorted(node_map)],
            edges=[edge_map[key] for key in sorted(edge_map)],
        )
        validate_graph(graph)
        return graph

    def digest(self) -> str:
        return hash_value(self.merged_graph().to_dict())

    @staticmethod
    def _invalidation_plan(
        previous: EvidenceProjection | None,
        *,
        current_source_hash: str | None,
        reason: str,
        source_uri: str,
    ) -> InvalidationPlan:
        if previous is None:
            return InvalidationPlan(
                source_uri=source_uri,
                previous_source_hash=None,
                current_source_hash=current_source_hash,
                reason=reason,
            )

        node_by_id = {
            node.id: node
            for node in previous.graph.nodes
        }
        hits = blast_radius(
            previous.graph,
            [previous.document_node_id],
        )
        affected = {
            previous.document_node_id,
            *(hit.node_id for hit in hits),
        }
        relation_ids = sorted(
            node_id
            for node_id in affected
            if (
                node_id in node_by_id
                and node_by_id[node_id].kind == NodeKind.DOMAIN_RELATION
            )
        )
        invalidated_nodes = sorted(
            node_id
            for node_id in affected
            if node_id not in relation_ids
        )

        return InvalidationPlan(
            source_uri=source_uri,
            previous_source_hash=previous.source_hash,
            current_source_hash=current_source_hash,
            invalidated_node_ids=invalidated_nodes,
            invalidated_edge_ids=sorted(
                _edge_id(edge)
                for edge in previous.graph.edges
            ),
            affected_relation_ids=relation_ids,
            reason=reason,
        )
