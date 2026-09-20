"""Domain evaluation and transaction rollback safety contracts."""

from __future__ import annotations

import pytest

from chatbot.knowledge.domain.kg_adapter import DomainKGAdapter
from chatbot.knowledge.domain.models import (
    DomainAction,
    DomainCandidate,
    DynamicRelation,
    StaticGuardResult,
)
from chatbot.knowledge.domain.pipeline import DomainPipeline
from chatbot.knowledge.shared.models import Polarity, RawEdge
from chatbot.knowledge.storage.inmemory_repository import InMemoryGraphRepository
from chatbot.knowledge.storage.transaction_manager import KGTransactionManager
from chatbot.knowledge.validation.models import (
    ValidationDestination,
    ValidationResult,
)


class _FakeIntake:
    def process(self, edge, validation_result, resolved_entities):
        return DomainCandidate(
            candidate_id="candidate",
            raw_edge_id=edge.raw_edge_id,
            head_canonical_id="head",
            head_canonical_name="Head",
            tail_canonical_id="tail",
            tail_canonical_name="Tail",
            relation_type="Cause",
            polarity="+",
            semantic_tag="sem_confident",
            combined_conf=0.9,
            student_conf=0.9,
            fragment_text=edge.fragment_text,
        )


class _FakeStaticGuard:
    def check(self, candidate):
        return StaticGuardResult(
            candidate_id=candidate.candidate_id,
            static_pass=True,
            static_conflict=False,
            action=DomainAction.CREATE_NEW,
        )


class _FailIfUpdated:
    def update(self, candidate, tx=None):
        raise AssertionError("evaluate() must not mutate dynamic domain state")


def _validation(edge_id: str) -> ValidationResult:
    return ValidationResult(
        edge_id=edge_id,
        validation_passed=True,
        destination=ValidationDestination.DOMAIN_CANDIDATE,
        combined_conf=0.9,
        student_conf=0.9,
    )


def test_domain_evaluate_path_does_not_mutate_dynamic_store():
    pipeline = DomainPipeline.__new__(DomainPipeline)
    pipeline.intake = _FakeIntake()
    pipeline.static_guard = _FakeStaticGuard()
    pipeline.dynamic_update = _FailIfUpdated()
    pipeline.tx_manager = None
    pipeline._stats = {
        "total": 0,
        "domain_accepted": 0,
        "logged": 0,
        "static_matched": 0,
        "static_conflict": 0,
        "new_relations": 0,
        "updated_relations": 0,
    }

    edge = RawEdge(
        raw_edge_id="edge",
        head_entity_id="head-temp",
        tail_entity_id="tail-temp",
        relation_type="Cause",
        polarity_guess=Polarity.POSITIVE,
        student_conf=0.9,
        fragment_id="fragment",
        fragment_text="head causes tail",
    )

    result = pipeline.evaluate(
        edge,
        _validation("edge"),
        [],
    )

    assert result.final_destination == "domain"
    assert result.dynamic_result is None
    assert pipeline.get_stats()["domain_accepted"] == 1
    assert pipeline.get_stats()["new_relations"] == 0


def test_domain_relation_transaction_rollback_restores_previous_state():
    repository = InMemoryGraphRepository()
    tx_manager = KGTransactionManager(repository)
    adapter = DomainKGAdapter(
        repository=repository,
        tx_manager=tx_manager,
        read_only=False,
    )

    original = DynamicRelation(
        relation_id="EVD_original",
        head_id="head",
        head_name="Old head",
        tail_id="tail",
        tail_name="Old tail",
        relation_type="Cause",
        sign="+",
        domain_conf=0.5,
        evidence_count=1,
        conflict_count=0,
        origin="evidence_reconciled",
    )
    adapter.upsert_relation(original)

    updated = DynamicRelation(
        relation_id="EVD_original",
        head_id="head",
        head_name="New head",
        tail_id="tail",
        tail_name="New tail",
        relation_type="Cause",
        sign="-",
        domain_conf=0.8,
        evidence_count=4,
        conflict_count=2,
        origin="evidence_reconciled",
        drift_flag=True,
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with adapter.with_transaction() as tx:
            adapter.upsert_relation(updated, tx=tx)
            raise RuntimeError("synthetic failure")

    head = repository.get_entity("head")
    relation = repository.get_relation(
        "head",
        "domain:Cause",
        "tail",
    )

    assert head is not None
    assert head["props"]["name"] == "Old head"
    assert relation is not None
    assert relation["props"]["sign"] == "+"
    assert relation["props"]["domain_conf"] == 0.5
    assert relation["props"]["evidence_count"] == 1


def test_evidence_pipeline_can_use_non_mutating_domain_evaluation():
    from types import SimpleNamespace

    from chatbot.knowledge.evidence.pipeline import EvidenceProvenancePipeline

    class Extraction:
        def process(self, raw_text, doc_id, source_uri=None):
            return SimpleNamespace(
                fragments=[],
                raw_edges=[],
                resolved_entities=[],
            )

    class Validation:
        def validate_batch(self, edges, resolved_entities, fragment_texts=None):
            return []

    class Domain:
        def __init__(self):
            self.evaluated = False

        def evaluate_batch(self, edges, validation_results, resolved_entities):
            self.evaluated = True
            return []

        def process_batch(self, edges, validation_results, resolved_entities):
            raise AssertionError("mutating domain path must not run")

    class Projector:
        def project(self, extraction, validation_results=(), domain_results=()):
            return "projection"

    domain = Domain()
    pipeline = EvidenceProvenancePipeline(
        extraction_pipeline=Extraction(),
        validation_pipeline=Validation(),
        domain_pipeline=domain,
        projector=Projector(),
        apply_domain_updates=False,
    )

    result = pipeline.process(
        "text",
        "doc",
        source_uri="file://doc.txt",
    )

    assert domain.evaluated is True
    assert result.projection == "projection"


def test_inmemory_relation_delete_cleans_neighbor_indexes():
    repository = InMemoryGraphRepository()
    repository.upsert_entity("a", ["Entity"], {"name": "A"})
    repository.upsert_entity("b", ["Entity"], {"name": "B"})
    repository.upsert_relation("a", "domain:Cause", "b", {"value": 1})

    assert repository.get_neighbors("a", direction="out")
    assert repository.get_neighbors("b", direction="in")

    assert repository.delete_relation("a", "domain:Cause", "b") is True

    assert repository.get_neighbors("a", direction="out") == []
    assert repository.get_neighbors("b", direction="in") == []


def test_domain_adapter_round_trip_preserves_temporal_and_drift_metadata():
    from datetime import datetime

    repository = InMemoryGraphRepository()
    adapter = DomainKGAdapter(
        repository=repository,
        tx_manager=KGTransactionManager(repository),
        read_only=False,
    )
    created_at = datetime(2026, 1, 2, 3, 4, 5)
    last_update = datetime(2026, 2, 3, 4, 5, 6)

    relation = DynamicRelation(
        relation_id="EVD_roundtrip",
        head_id="head",
        head_name="Head",
        tail_id="tail",
        tail_name="Tail",
        relation_type="Cause",
        sign="+",
        domain_conf=0.7,
        evidence_count=3,
        conflict_count=1,
        created_at=created_at,
        last_update=last_update,
        origin="evidence_reconciled",
        semantic_tags=["sem_confident"],
        decay_applied=True,
        drift_flag=True,
    )
    adapter.upsert_relation(relation)

    restored = adapter.get_relation("head", "tail", "Cause")

    assert restored is not None
    assert restored.created_at == created_at
    assert restored.last_update == last_update
    assert restored.decay_applied is True
    assert restored.drift_flag is True
