"""Deterministically reconcile affected domain relations from evidence."""

from __future__ import annotations

from datetime import datetime

from chatbot.knowledge.bootstrap import get_domain_kg_adapter
from chatbot.knowledge.domain.kg_adapter import DomainKGAdapter
from chatbot.knowledge.domain.models import DynamicRelation
from chatbot.knowledge.evidence import (
    EvidenceLedger,
    EvidenceScoreAggregator,
    EvidenceScorePolicy,
    RelationEvidenceViewBuilder,
)
from chatbot.knowledge.workspace.hashing import hash_value
from chatbot.knowledge.workspace.models import NodeKind, WorkspaceGraph

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
        self.view_builder = RelationEvidenceViewBuilder(self.scorer)

    def reconcile(
        self,
        ledger: EvidenceLedger,
        relation_specs: dict[str, RelationSpec],
    ) -> list[RelationReconcileResult]:
        if not relation_specs:
            return []

        graph = ledger.merged_graph()
        views = self.view_builder.by_node_id(graph)
        operations = []
        results: list[RelationReconcileResult] = []

        for node_id in sorted(relation_specs):
            spec = relation_specs[node_id]
            view = views.get(node_id)
            if view is not None:
                spec = RelationSpec(
                    node_id=node_id,
                    head_id=view.head_id,
                    tail_id=view.tail_id,
                    relation_type=view.relation_type,
                )

            existing = self.adapter.get_relation(
                spec.head_id,
                spec.tail_id,
                spec.relation_type,
            )

            support_count = view.support_count if view is not None else 0
            conflict_count = view.conflict_count if view is not None else 0
            score = (
                view.score
                if view is not None
                else self.scorer.score([], [])
            )

            if support_count == 0:
                if existing is not None and existing.origin == MANAGED_ORIGIN:
                    operations.append(("delete", spec, None))
                    results.append(
                        RelationReconcileResult(
                            relation_node_id=node_id,
                            action="deleted",
                            evidence_count=0,
                            conflict_count=conflict_count,
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
                            conflict_count=conflict_count,
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
                        evidence_count=support_count,
                        conflict_count=conflict_count,
                        domain_conf=existing.domain_conf,
                        support_score=score.support_score,
                        conflict_score=score.conflict_score,
                        support_source_count=score.support_source_count,
                        conflict_source_count=score.conflict_source_count,
                        evidence_score_version=score.version,
                    )
                )
                continue

            assert view is not None
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
                head_name=view.head_name,
                tail_id=spec.tail_id,
                tail_name=view.tail_name,
                relation_type=spec.relation_type,
                sign=view.sign,
                domain_conf=score.score,
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
                semantic_tags=list(view.semantic_tags),
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
                    domain_conf=score.score,
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
