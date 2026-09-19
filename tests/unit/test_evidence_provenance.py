"""Evidence provenance chain and source-scoped invalidation contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chatbot.knowledge.domain.models import (
    DomainAction,
    DomainProcessResult,
    DynamicUpdateResult,
    StaticGuardResult,
)
from chatbot.knowledge.evidence import EvidenceLedger, EvidenceProjector
from chatbot.knowledge.shared.models import (
    ExtractionResult,
    Fragment,
    Polarity,
    RawEdge,
    ResolvedEntity,
    ResolutionMode,
)
from chatbot.knowledge.validation.models import (
    SemanticTag,
    SemanticValidationResult,
    ValidationDestination,
    ValidationResult,
)
from chatbot.knowledge.workspace.models import MetaRelation, NodeKind


RAW_TEXT = "배포 이후 오류율이 증가했다."


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extraction(
    *,
    source_uri: str = "file://incident.txt",
    raw_text: str = RAW_TEXT,
    fragment_id: str = "runtime-fragment-a",
    raw_edge_id: str = "runtime-edge-a",
    head_runtime_id: str = "runtime-head-a",
    tail_runtime_id: str = "runtime-tail-a",
) -> ExtractionResult:
    fragment = Fragment(
        fragment_id=fragment_id,
        text=raw_text,
        doc_id="incident-doc",
        source_start=0,
        source_end=len(raw_text),
    )
    resolved = [
        ResolvedEntity(
            entity_id=head_runtime_id,
            canonical_id="event:deployment",
            canonical_name="Deployment",
            canonical_type="Event",
            resolution_mode=ResolutionMode.STATIC_DOMAIN,
            resolution_conf=1.0,
            surface_text="배포",
            fragment_id=fragment_id,
        ),
        ResolvedEntity(
            entity_id=tail_runtime_id,
            canonical_id="indicator:error-rate",
            canonical_name="Error rate",
            canonical_type="Indicator",
            resolution_mode=ResolutionMode.STATIC_DOMAIN,
            resolution_conf=1.0,
            surface_text="오류율",
            fragment_id=fragment_id,
        ),
    ]
    raw_edge = RawEdge(
        raw_edge_id=raw_edge_id,
        head_entity_id=head_runtime_id,
        head_canonical_name="Deployment",
        tail_entity_id=tail_runtime_id,
        tail_canonical_name="Error rate",
        relation_type="Cause",
        polarity_guess=Polarity.POSITIVE,
        student_conf=0.9,
        fragment_id=fragment_id,
        fragment_text=raw_text,
    )
    return ExtractionResult(
        doc_id="incident-doc",
        source_uri=source_uri,
        source_hash=_sha(raw_text),
        fragments=[fragment],
        entity_candidates=[],
        resolved_entities=resolved,
        raw_edges=[raw_edge],
    )


def _validation(
    raw_edge_id: str,
    *,
    passed: bool = True,
    domain_conflict: bool = False,
) -> ValidationResult:
    return ValidationResult(
        edge_id=raw_edge_id,
        validation_passed=passed,
        destination=(
            ValidationDestination.DOMAIN_CANDIDATE
            if passed
            else ValidationDestination.DROP_LOG
        ),
        combined_conf=0.85 if passed else 0.2,
        student_conf=0.9,
        semantic_conf=0.9 if passed else 0.2,
        semantic_result=SemanticValidationResult(
            edge_id=raw_edge_id,
            semantic_tag=(
                SemanticTag.SEM_CONFIDENT
                if passed
                else SemanticTag.SEM_WRONG
            ),
            semantic_confidence=0.9 if passed else 0.2,
            domain_conflict=domain_conflict,
        ),
        rejection_reason=None if passed else "semantic_rejected",
    )


def _domain(
    raw_edge_id: str,
    *,
    accepted: bool = True,
    static_conflict: bool = False,
) -> DomainProcessResult:
    if static_conflict:
        return DomainProcessResult(
            candidate_id="runtime-candidate",
            raw_edge_id=raw_edge_id,
            final_destination="log",
            static_result=StaticGuardResult(
                candidate_id="runtime-candidate",
                static_pass=False,
                static_conflict=True,
                action=DomainAction.REJECT_TO_LOG,
            ),
        )
    if not accepted:
        return DomainProcessResult(
            candidate_id="runtime-candidate",
            raw_edge_id=raw_edge_id,
            final_destination="log",
        )
    return DomainProcessResult(
        candidate_id="runtime-candidate",
        raw_edge_id=raw_edge_id,
        final_destination="domain",
        dynamic_result=DynamicUpdateResult(
            candidate_id="runtime-candidate",
            relation_id="runtime-domain-relation",
            action=DomainAction.CREATE_NEW,
            domain_conf=0.55,
            evidence_count=1,
            is_new=True,
        ),
        domain_relation_id="runtime-domain-relation",
    )


def _project(
    *,
    source_uri: str = "file://incident.txt",
    raw_text: str = RAW_TEXT,
    fragment_id: str = "runtime-fragment-a",
    raw_edge_id: str = "runtime-edge-a",
    head_runtime_id: str = "runtime-head-a",
    tail_runtime_id: str = "runtime-tail-a",
    passed: bool = True,
    static_conflict: bool = False,
):
    extraction = _extraction(
        source_uri=source_uri,
        raw_text=raw_text,
        fragment_id=fragment_id,
        raw_edge_id=raw_edge_id,
        head_runtime_id=head_runtime_id,
        tail_runtime_id=tail_runtime_id,
    )
    return EvidenceProjector().project(
        extraction,
        validation_results=[
            _validation(
                raw_edge_id,
                passed=passed,
                domain_conflict=static_conflict,
            )
        ],
        domain_results=[
            _domain(
                raw_edge_id,
                accepted=passed and not static_conflict,
                static_conflict=static_conflict,
            )
        ],
    )


def test_projection_is_stable_across_runtime_uuid_changes():
    first = _project(
        fragment_id="fragment-run-1",
        raw_edge_id="edge-run-1",
        head_runtime_id="head-run-1",
        tail_runtime_id="tail-run-1",
    )
    second = _project(
        fragment_id="fragment-run-2",
        raw_edge_id="edge-run-2",
        head_runtime_id="head-run-2",
        tail_runtime_id="tail-run-2",
    )

    assert first.source_hash == second.source_hash
    assert first.projection_hash == second.projection_hash
    assert first.graph.to_dict() == second.graph.to_dict()


def test_projection_links_document_fragment_assertion_and_domain_relation():
    projection = _project()
    graph = projection.graph
    nodes = {node.id: node for node in graph.nodes}

    document_ids = [
        node.id for node in graph.nodes
        if node.kind == NodeKind.DOCUMENT
    ]
    fragment_ids = [
        node.id for node in graph.nodes
        if node.kind == NodeKind.FRAGMENT
    ]
    assertion_ids = [
        node.id for node in graph.nodes
        if node.kind == NodeKind.ASSERTION
    ]
    relation_ids = [
        node.id for node in graph.nodes
        if node.kind == NodeKind.DOMAIN_RELATION
    ]

    assert document_ids == [projection.document_node_id]
    assert len(fragment_ids) == 1
    assert len(assertion_ids) == 1
    assert len(relation_ids) == 1

    triples = {
        (edge.source, edge.relation, edge.target)
        for edge in graph.edges
    }
    assert (
        document_ids[0],
        MetaRelation.PRODUCES,
        fragment_ids[0],
    ) in triples
    assert (
        fragment_ids[0],
        MetaRelation.PRODUCES,
        assertion_ids[0],
    ) in triples
    assert (
        relation_ids[0],
        MetaRelation.SUPPORTED_BY,
        assertion_ids[0],
    ) in triples

    semantic_edges = [
        edge
        for edge in graph.edges
        if edge.relation == MetaRelation.DOMAIN_RELATION
    ]
    assert len(semantic_edges) == 1
    assert semantic_edges[0].semantic_type == "Cause"
    assert nodes[assertion_ids[0]].properties["validation_state"] == "passed"
    assert nodes[assertion_ids[0]].properties["domain_state"] == "domain"


def test_conflicting_evidence_is_preserved_without_promoting_semantic_edge():
    projection = _project(passed=False, static_conflict=True)

    relation_ids = [
        node.id for node in projection.graph.nodes
        if node.kind == NodeKind.DOMAIN_RELATION
    ]
    assert len(relation_ids) == 1

    contradicted = [
        edge
        for edge in projection.graph.edges
        if edge.relation == MetaRelation.CONTRADICTED_BY
    ]
    semantic = [
        edge
        for edge in projection.graph.edges
        if edge.relation == MetaRelation.DOMAIN_RELATION
    ]

    assert len(contradicted) == 1
    assert contradicted[0].source == relation_ids[0]
    assert semantic == []


def test_source_change_invalidates_only_its_document_chain_and_recomputes_relation():
    ledger = EvidenceLedger()
    first = _project()
    initial = ledger.upsert(first)

    assert initial.reason == "source_added"
    assert initial.invalidated_node_ids == []

    changed = _project(
        raw_text="배포 이후 오류율이 크게 증가했다.",
        fragment_id="new-runtime-fragment",
        raw_edge_id="new-runtime-edge",
        head_runtime_id="new-head",
        tail_runtime_id="new-tail",
    )
    plan = ledger.upsert(changed)

    assert plan.reason == "source_changed"
    assert first.document_node_id in plan.invalidated_node_ids
    assert any(
        node_id.startswith("fragment:")
        for node_id in plan.invalidated_node_ids
    )
    assert any(
        node_id.startswith("assertion:")
        for node_id in plan.invalidated_node_ids
    )
    assert len(plan.affected_relation_ids) == 1
    assert all(
        not node_id.startswith("entity:")
        for node_id in plan.invalidated_node_ids
    )

    merged = ledger.merged_graph()
    assert changed.document_node_id in {node.id for node in merged.nodes}
    assert first.source_hash != changed.source_hash


def test_replacing_one_source_does_not_remove_other_source_support():
    ledger = EvidenceLedger()
    one = _project(source_uri="file://one.txt")
    two = _project(
        source_uri="file://two.txt",
        fragment_id="fragment-two",
        raw_edge_id="edge-two",
        head_runtime_id="head-two",
        tail_runtime_id="tail-two",
    )
    ledger.upsert(one)
    ledger.upsert(two)

    merged = ledger.merged_graph()
    relation = next(
        node
        for node in merged.nodes
        if node.kind == NodeKind.DOMAIN_RELATION
    )
    assert {source.path for source in relation.sources} == {
        "file://one.txt",
        "file://two.txt",
    }

    plan = ledger.remove("file://one.txt")
    assert plan.reason == "source_removed"
    assert plan.affected_relation_ids == [relation.id]

    merged_after = ledger.merged_graph()
    relation_after = next(
        node
        for node in merged_after.nodes
        if node.kind == NodeKind.DOMAIN_RELATION
    )
    assert [source.path for source in relation_after.sources] == [
        "file://two.txt"
    ]
    assert all(
        source.path != "file://one.txt"
        for node in merged_after.nodes
        for source in node.sources
    )


def test_same_source_and_projection_is_idempotent():
    ledger = EvidenceLedger()
    projection = _project()

    ledger.upsert(projection)
    digest = ledger.digest()
    plan = ledger.upsert(projection)

    assert plan.reason == "no_change"
    assert not plan.changed
    assert ledger.digest() == digest


def test_evidence_ledger_round_trip_preserves_projection_state(tmp_path: Path):
    ledger = EvidenceLedger()
    projection = _project()
    ledger.upsert(projection)

    path = tmp_path / "knowledge-workspace/evidence-ledger.json"
    ledger.save(path)
    restored = EvidenceLedger.load(path)

    assert restored.sources() == ["file://incident.txt"]
    assert restored.digest() == ledger.digest()
    assert restored.get("file://incident.txt").projection_hash == (
        projection.projection_hash
    )


def test_domain_aggregate_state_does_not_change_evidence_identity():
    extraction = _extraction(raw_edge_id="edge-stable")
    validation = _validation("edge-stable")

    created = DomainProcessResult(
        candidate_id="candidate-a",
        raw_edge_id="edge-stable",
        final_destination="domain",
        dynamic_result=DynamicUpdateResult(
            candidate_id="candidate-a",
            relation_id="runtime-relation-a",
            action=DomainAction.CREATE_NEW,
            domain_conf=0.50,
            evidence_count=1,
            is_new=True,
        ),
    )
    updated = DomainProcessResult(
        candidate_id="candidate-b",
        raw_edge_id="edge-stable",
        final_destination="domain",
        dynamic_result=DynamicUpdateResult(
            candidate_id="candidate-b",
            relation_id="runtime-relation-b",
            action=DomainAction.UPDATE_EXISTING,
            domain_conf=0.91,
            evidence_count=37,
            is_new=False,
        ),
    )

    projector = EvidenceProjector()
    first = projector.project(
        extraction,
        validation_results=[validation],
        domain_results=[created],
    )
    second = projector.project(
        extraction,
        validation_results=[validation],
        domain_results=[updated],
    )

    assert first.projection_hash == second.projection_hash
    assert first.graph.to_dict() == second.graph.to_dict()


def test_projector_rejects_non_sha256_source_hash():
    extraction = _extraction()
    extraction.source_hash = "not-a-digest"

    try:
        EvidenceProjector().project(extraction)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("invalid source hash must be rejected")


def test_ledger_rejects_tampered_persisted_projection(tmp_path: Path):
    ledger = EvidenceLedger()
    ledger.upsert(_project())
    path = tmp_path / "evidence-ledger.json"
    ledger.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["projections"][0]["projection_hash"] = "0" * 64
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        EvidenceLedger.load(path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered ledger must be rejected")


def test_domain_log_without_conflict_does_not_promote_relation():
    extraction = _extraction(raw_edge_id="edge-log")
    validation = _validation("edge-log", passed=True)
    domain = _domain("edge-log", accepted=False, static_conflict=False)

    projection = EvidenceProjector().project(
        extraction,
        validation_results=[validation],
        domain_results=[domain],
    )

    assert all(
        node.kind != NodeKind.DOMAIN_RELATION
        for node in projection.graph.nodes
    )
    assert all(
        edge.relation not in {
            MetaRelation.SUPPORTED_BY,
            MetaRelation.CONTRADICTED_BY,
            MetaRelation.DOMAIN_RELATION,
        }
        for edge in projection.graph.edges
    )
