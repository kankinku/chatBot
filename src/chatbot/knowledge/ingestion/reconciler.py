"""Deterministically reconcile affected domain relations from evidence."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from chatbot.knowledge.bootstrap import get_domain_kg_adapter
from chatbot.knowledge.domain.kg_adapter import DomainKGAdapter
from chatbot.knowledge.domain.models import DynamicRelation
from chatbot.knowledge.evidence import (
    EvidenceLedger,
    EvidenceScoreAggregator,
    EvidenceScorePolicy,
)
from chatbot.knowledge.workspace.hashing import hash_value
from chatbot.knowledge.workspace.models import (
    MetaRelation,
    NodeKind,
    WorkspaceGraph,
)

from .models import RelationReconcileResult, RelationSpec


MANAGED_ORIGIN = "evidence_reconciled"


def relation_specs_from_graph(
    graph: WorkspaceGraph,
) -> dict[str, RelationSpec]:
    specs: dict[str, RelationSpec] = {}
    for node in graph.nodes:
        if node.kind != NodeKind.DOMAIN_RELATION:
            continue
        props = node.properties
        head = props.get("head")
        tail = props.get("tail")
        relation_type = props.get("relation_type")
        if not head or not tail or not relation_type:
            continue
        specs[node.id] = RelationSpec(
            node_id=node.id,
            head_id=str(head),
            tail_id=str(tail),
            relation_type=str(relation_type),
        )
    return specs


class EvidenceRelationReconciler:
    """Rebuild managed dynamic relations from the current evidence ledger."""

    def __init__(
        self,
        adapter: DomainKGAdapter | None = None,
        *,
        scorer: EvidenceScoreAggregator | None = None,
        initial_conf: float = 0.5,
        support_gain: float = 0.4,
        conflict_penalty: float = 0.45,
        support_rate: float | None = None,
        conflict_rate: float | None = None,
    ):
        self.adapter = adapter or get_domain_kg_adapter()
        if support_rate is not None:
            support_gain = support_rate
        if conflict_rate is not None:
            conflict_penalty = conflict_rate
        self.scorer = scorer or EvidenceScoreAggregator(
            EvidenceScorePolicy(
                baseline=initial_conf,
                support_gain=support_gain,
                conflict_penalty=conflict_penalty,
            )
        )

    def reconcile(
        self,
        ledger: EvidenceLedger,
        relation_specs: dict[str, RelationSpec],
    ) -> list[RelationReconcileResult]:
        if not relation_specs:
            return []

        graph = ledger.merged_graph()
        node_map = {node.id: node for node in graph.nodes}
        outgoing: dict[str, list] = {}
        for edge in graph.edges:
            outgoing.setdefault(edge.source, []).append(edge)

        operations = []
        results: list[RelationReconcileResult] = []

        for node_id in sorted(relation_specs):
            spec = relation_specs[node_id]
            current = node_map.get(node_id)
            if current is not None and current.kind == NodeKind.DOMAIN_RELATION:
                props = current.properties
                spec = RelationSpec(
                    node_id=node_id,
                    head_id=str(props.get("head", spec.head_id)),
                    tail_id=str(props.get("tail", spec.tail_id)),
                    relation_type=str(
                        props.get("relation_type", spec.relation_type)
                    ),
                )

            support_assertions = []
            conflict_assertions = []
            for edge in outgoing.get(node_id, []):
                assertion = node_map.get(edge.target)
                if assertion is None or assertion.kind != NodeKind.ASSERTION:
                    continue
                if edge.relation == MetaRelation.SUPPORTED_BY:
                    support_assertions.append(assertion)
                elif edge.relation == MetaRelation.CONTRADICTED_BY:
                    conflict_assertions.append(assertion)

            score = self.scorer.score(
                support_assertions,
                conflict_assertions,
            )

            existing = self.adapter.get_relation(
                spec.head_id,
                spec.tail_id,
                spec.relation_type,
            )

            if not support_assertions:
                if existing is not None and existing.origin == MANAGED_ORIGIN:
                    operations.append(("delete", spec, None))
                    results.append(
                        RelationReconcileResult(
                            relation_node_id=node_id,
                            action="deleted",
                            evidence_count=0,
                            conflict_count=len(conflict_assertions),
                            domain_conf=None,
                            support_score=score.support_score,
                            conflict_score=score.conflict_score,
                            support_source_count=score.support_source_count,
                            conflict_source_count=score.conflict_source_count,
                            evidence_score_version=score.version,
                        )
                    )
                else:
                    results.append(
                        RelationReconcileResult(
                            relation_node_id=node_id,
                            action=(
                                "preserved_external"
                                if existing is not None
                                else "absent"
                            ),
                            evidence_count=0,
                            conflict_count=len(conflict_assertions),
                            domain_conf=(
                                existing.domain_conf
                                if existing is not None
                                else None
                            ),
                            support_score=score.support_score,
                            conflict_score=score.conflict_score,
                            support_source_count=score.support_source_count,
                            conflict_source_count=score.conflict_source_count,
                            evidence_score_version=score.version,
                        )
                    )
                continue

            if existing is not None and existing.origin != MANAGED_ORIGIN:
                results.append(
                    RelationReconcileResult(
                        relation_node_id=node_id,
                        action="preserved_external",
                        evidence_count=len(support_assertions),
                        conflict_count=len(conflict_assertions),
                        domain_conf=existing.domain_conf,
                        support_score=score.support_score,
                        conflict_score=score.conflict_score,
                        support_source_count=score.support_source_count,
                        conflict_source_count=score.conflict_source_count,
                        evidence_score_version=score.version,
                    )
                )
                continue

            support_count = len(support_assertions)
            conflict_count = len(conflict_assertions)
            confidence = score.score
            sign = self._majority_sign(support_assertions)
            semantic_tags = sorted(
                {
                    str(assertion.properties["semantic_tag"])
                    for assertion in support_assertions
                    if assertion.properties.get("semantic_tag")
                }
            )
            relation = DynamicRelation(
                relation_id=(
                    "EVD_"
                    + hash_value(
                        {
                            "head": spec.head_id,
                            "tail": spec.tail_id,
                            "relation_type": spec.relation_type,
                        }
                    )[:20]
                ),
                head_id=spec.head_id,
                head_name=self._entity_name(
                    graph,
                    spec.head_id,
                ),
                tail_id=spec.tail_id,
                tail_name=self._entity_name(
                    graph,
                    spec.tail_id,
                ),
                relation_type=spec.relation_type,
                sign=sign,
                domain_conf=confidence,
                evidence_count=support_count,
                conflict_count=conflict_count,
                support_score=score.support_score,
                conflict_score=score.conflict_score,
                support_source_count=score.support_source_count,
                conflict_source_count=score.conflict_source_count,
                evidence_score_version=score.version,
                created_at=(
                    existing.created_at
                    if existing is not None
                    else datetime.now()
                ),
                last_update=datetime.now(),
                origin=MANAGED_ORIGIN,
                semantic_tags=semantic_tags,
                drift_flag=conflict_count > 0,
            )
            operations.append(("upsert", spec, relation))
            results.append(
                RelationReconcileResult(
                    relation_node_id=node_id,
                    action=(
                        "updated"
                        if existing is not None
                        else "created"
                    ),
                    evidence_count=support_count,
                    conflict_count=conflict_count,
                    domain_conf=confidence,
                    support_score=score.support_score,
                    conflict_score=score.conflict_score,
                    support_source_count=score.support_source_count,
                    conflict_source_count=score.conflict_source_count,
                    evidence_score_version=score.version,
                )
            )

        if operations:
            with self.adapter.with_transaction() as tx:
                for action, spec, relation in operations:
                    if action == "delete":
                        self.adapter.delete_relation(
                            spec.head_id,
                            spec.tail_id,
                            spec.relation_type,
                            tx=tx,
                        )
                    else:
                        self.adapter.upsert_relation(
                            relation,
                            tx=tx,
                        )

        return results

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
        if len(winners) != 1:
            return "unknown"
        return winners[0]

    @staticmethod
    def _entity_name(
        graph: WorkspaceGraph,
        stable_key: str,
    ) -> str:
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
