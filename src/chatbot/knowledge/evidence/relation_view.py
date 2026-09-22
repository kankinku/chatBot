"""Pure evidence-backed relation views shared by reconciliation and projection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from chatbot.knowledge.workspace.models import (
    MetaRelation,
    NodeKind,
    SourceRef,
    WorkspaceGraph,
)

from .scoring import EvidenceScoreAggregator, EvidenceScoreSummary


@dataclass(frozen=True)
class EvidenceBackedRelation:
    relation_node_id: str
    head_id: str
    head_name: str
    tail_id: str
    tail_name: str
    relation_type: str
    sign: str
    score: EvidenceScoreSummary
    support_assertion_ids: tuple[str, ...]
    conflict_assertion_ids: tuple[str, ...]
    source_refs: tuple[SourceRef, ...]
    semantic_tags: tuple[str, ...]

    @property
    def support_count(self) -> int:
        return len(self.support_assertion_ids)

    @property
    def conflict_count(self) -> int:
        return len(self.conflict_assertion_ids)

    @property
    def stable_key(self) -> tuple[str, str, str]:
        return (self.head_id, self.tail_id, self.relation_type)


class RelationEvidenceViewBuilder:
    """Derive relation evidence semantics without reading or mutating live state."""

    def __init__(self, scorer: EvidenceScoreAggregator | None = None):
        self.scorer = scorer or EvidenceScoreAggregator()

    def build(self, graph: WorkspaceGraph) -> list[EvidenceBackedRelation]:
        node_map = {node.id: node for node in graph.nodes}
        outgoing: dict[str, list] = {}
        for edge in graph.edges:
            outgoing.setdefault(edge.source, []).append(edge)

        result: list[EvidenceBackedRelation] = []
        for node in sorted(graph.nodes, key=lambda item: item.id):
            if node.kind != NodeKind.DOMAIN_RELATION:
                continue
            props = node.properties
            head = props.get("head")
            tail = props.get("tail")
            relation_type = props.get("relation_type")
            if not head or not tail or not relation_type:
                continue

            support_assertions = []
            conflict_assertions = []
            for edge in outgoing.get(node.id, []):
                assertion = node_map.get(edge.target)
                if assertion is None or assertion.kind != NodeKind.ASSERTION:
                    continue
                if edge.relation == MetaRelation.SUPPORTED_BY:
                    support_assertions.append(assertion)
                elif edge.relation == MetaRelation.CONTRADICTED_BY:
                    conflict_assertions.append(assertion)

            support_assertions.sort(key=lambda item: item.id)
            conflict_assertions.sort(key=lambda item: item.id)
            score = self.scorer.score(
                support_assertions,
                conflict_assertions,
            )
            refs = {
                ref
                for assertion in [*support_assertions, *conflict_assertions]
                for ref in assertion.sources
            }
            semantic_tags = sorted(
                {
                    str(assertion.properties["semantic_tag"])
                    for assertion in support_assertions
                    if assertion.properties.get("semantic_tag")
                }
            )
            result.append(
                EvidenceBackedRelation(
                    relation_node_id=node.id,
                    head_id=str(head),
                    head_name=self._entity_name(graph, str(head)),
                    tail_id=str(tail),
                    tail_name=self._entity_name(graph, str(tail)),
                    relation_type=str(relation_type),
                    sign=self._majority_sign(support_assertions),
                    score=score,
                    support_assertion_ids=tuple(
                        assertion.id for assertion in support_assertions
                    ),
                    conflict_assertion_ids=tuple(
                        assertion.id for assertion in conflict_assertions
                    ),
                    source_refs=tuple(sorted(refs)),
                    semantic_tags=tuple(semantic_tags),
                )
            )
        return result

    def by_node_id(
        self,
        graph: WorkspaceGraph,
    ) -> dict[str, EvidenceBackedRelation]:
        return {item.relation_node_id: item for item in self.build(graph)}

    @staticmethod
    def _majority_sign(assertions) -> str:
        values = []
        for assertion in assertions:
            value = (
                assertion.properties.get("polarity_final")
                or assertion.properties.get("polarity_guess")
            )
            if value in {"+", "-", "neutral"}:
                values.append(value)
        if not values:
            return "unknown"
        counts = Counter(values)
        maximum = max(counts.values())
        winners = sorted(
            value
            for value, count in counts.items()
            if count == maximum
        )
        return winners[0] if len(winners) == 1 else "unknown"

    @staticmethod
    def _entity_name(graph: WorkspaceGraph, stable_key: str) -> str:
        exact_id = f"entity:{stable_key}"
        for node in graph.nodes:
            if node.id == exact_id:
                return (
                    node.properties.get("canonical_name")
                    or node.label
                    or stable_key
                )
        for node in graph.nodes:
            if node.properties.get("stable_key") == stable_key:
                return (
                    node.properties.get("canonical_name")
                    or node.label
                    or stable_key
                )
        return stable_key
