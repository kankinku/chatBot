"""Selective ingestion and deterministic relation reconciliation contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import chatbot.knowledge.ingestion.manager as manager_module
from chatbot.knowledge.domain.kg_adapter import DomainKGAdapter
from chatbot.knowledge.domain.models import DomainProcessResult
from chatbot.knowledge.evidence import EvidenceProjector
from chatbot.knowledge.ingestion import (
    EvidenceRelationReconciler,
    IngestionAction,
    IngestionStateStore,
    SelectiveIngestionManager,
    SourceDocument,
)
from chatbot.knowledge.shared.models import (
    ExtractionResult,
    Fragment,
    Polarity,
    RawEdge,
    ResolvedEntity,
    ResolutionMode,
)
from chatbot.knowledge.storage.inmemory_repository import InMemoryGraphRepository
from chatbot.knowledge.storage.transaction_manager import KGTransactionManager
from chatbot.knowledge.validation.models import (
    SemanticTag,
    SemanticValidationResult,
    ValidationDestination,
    ValidationResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _projection(
    source_uri: str,
    text: str,
    *,
    doc_id: str = "doc",
):
    import hashlib

    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    fragment_id = "runtime-fragment"
    edge_id = "runtime-edge"
    head_id = "runtime-head"
    tail_id = "runtime-tail"

    extraction = ExtractionResult(
        doc_id=doc_id,
        source_uri=source_uri,
        source_hash=source_hash,
        fragments=[
            Fragment(
                fragment_id=fragment_id,
                text=text,
                doc_id=doc_id,
                source_start=0,
                source_end=len(text),
            )
        ],
        entity_candidates=[],
        resolved_entities=[
            ResolvedEntity(
                entity_id=head_id,
                canonical_id="event:deployment",
                canonical_name="Deployment",
                canonical_type="Event",
                resolution_mode=ResolutionMode.STATIC_DOMAIN,
                resolution_conf=1.0,
                surface_text="deployment",
                fragment_id=fragment_id,
            ),
            ResolvedEntity(
                entity_id=tail_id,
                canonical_id="indicator:error-rate",
                canonical_name="Error rate",
                canonical_type="Indicator",
                resolution_mode=ResolutionMode.STATIC_DOMAIN,
                resolution_conf=1.0,
                surface_text="error rate",
                fragment_id=fragment_id,
            ),
        ],
        raw_edges=[
            RawEdge(
                raw_edge_id=edge_id,
                head_entity_id=head_id,
                head_canonical_name="Deployment",
                tail_entity_id=tail_id,
                tail_canonical_name="Error rate",
                relation_type="Cause",
                polarity_guess=Polarity.POSITIVE,
                student_conf=0.9,
                fragment_id=fragment_id,
                fragment_text=text,
            )
        ],
    )
    validation = ValidationResult(
        edge_id=edge_id,
        validation_passed=True,
        destination=ValidationDestination.DOMAIN_CANDIDATE,
        combined_conf=0.9,
        student_conf=0.9,
        semantic_conf=0.9,
        semantic_result=SemanticValidationResult(
            edge_id=edge_id,
            semantic_tag=SemanticTag.SEM_CONFIDENT,
            semantic_confidence=0.9,
        ),
    )
    domain = DomainProcessResult(
        candidate_id="runtime-candidate",
        raw_edge_id=edge_id,
        final_destination="domain",
    )
    return EvidenceProjector().project(
        extraction,
        validation_results=[validation],
        domain_results=[domain],
    )


class _FakeEvidencePipeline:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def process(
        self,
        raw_text: str,
        doc_id: str,
        *,
        source_uri: str | None = None,
    ):
        assert source_uri is not None
        self.calls.append((source_uri, raw_text))
        if raw_text == "FAIL":
            raise RuntimeError("synthetic processing failure")
        return SimpleNamespace(
            projection=_projection(
                source_uri,
                raw_text,
                doc_id=doc_id,
            )
        )


def _manager(tmp_path: Path):
    repo = InMemoryGraphRepository()
    tx_manager = KGTransactionManager(repo)
    adapter = DomainKGAdapter(
        repository=repo,
        tx_manager=tx_manager,
        read_only=False,
    )
    reconciler = EvidenceRelationReconciler(adapter)
    pipeline = _FakeEvidencePipeline()
    store = IngestionStateStore(
        tmp_path / "knowledge-workspace/ingestion-state.json"
    )
    manager = SelectiveIngestionManager(
        project_root=PROJECT_ROOT,
        pipeline=pipeline,
        reconciler=reconciler,
        state_store=store,
    )
    return manager, pipeline, store, adapter, repo


def _doc(source_uri: str, text: str) -> SourceDocument:
    return SourceDocument(
        doc_id=Path(source_uri).stem or "doc",
        source_uri=source_uri,
        text=text,
    )


def test_selective_ingestion_skips_unchanged_and_recomputes_without_double_count(
    tmp_path: Path,
):
    manager, pipeline, store, adapter, _ = _manager(tmp_path)
    one = _doc("file://one.txt", "deployment increased error rate")
    two = _doc("file://two.txt", "deployment also increased error rate")

    first = manager.sync([one, two])
    assert first.processed == 2
    assert len(pipeline.calls) == 2

    relation = adapter.get_relation(
        "event:deployment",
        "indicator:error-rate",
        "Cause",
    )
    assert relation is not None
    assert relation.origin == "evidence_reconciled"
    assert relation.evidence_count == 2
    first_conf = relation.domain_conf

    second = manager.sync([one, two])
    assert second.processed == 0
    assert second.skipped == 2
    assert len(pipeline.calls) == 2

    changed_one = _doc(
        "file://one.txt",
        "deployment sharply increased error rate",
    )
    third = manager.sync([changed_one, two])
    assert third.processed == 1
    assert third.skipped == 1
    assert len(pipeline.calls) == 3

    relation = adapter.get_relation(
        "event:deployment",
        "indicator:error-rate",
        "Cause",
    )
    assert relation is not None
    assert relation.evidence_count == 2
    assert relation.domain_conf == first_conf

    state = store.load()
    assert state.records["file://one.txt"].source_hash != (
        first.source_results[0].source_hash
    )


def test_source_removal_reduces_support_then_deletes_last_managed_relation(
    tmp_path: Path,
):
    manager, _, _, adapter, _ = _manager(tmp_path)
    one = _doc("file://one.txt", "deployment increased error rate")
    two = _doc("file://two.txt", "deployment also increased error rate")

    manager.sync([one, two])
    report = manager.sync([two])

    assert report.removed == 1
    relation = adapter.get_relation(
        "event:deployment",
        "indicator:error-rate",
        "Cause",
    )
    assert relation is not None
    assert relation.evidence_count == 1

    report = manager.sync([])
    assert report.removed == 1
    assert adapter.get_relation(
        "event:deployment",
        "indicator:error-rate",
        "Cause",
    ) is None


def test_processor_stamp_change_reprocesses_unchanged_source(
    tmp_path: Path,
    monkeypatch,
):
    manager, pipeline, _, adapter, _ = _manager(tmp_path)
    document = _doc(
        "file://one.txt",
        "deployment increased error rate",
    )

    monkeypatch.setattr(
        manager_module,
        "processor_stamp",
        lambda root: "a" * 64,
    )
    first = manager.sync([document])
    assert first.source_results[0].action == IngestionAction.ADDED

    monkeypatch.setattr(
        manager_module,
        "processor_stamp",
        lambda root: "b" * 64,
    )
    second = manager.sync([document])

    assert second.source_results[0].action == IngestionAction.REPROCESSED
    assert len(pipeline.calls) == 2
    relation = adapter.get_relation(
        "event:deployment",
        "indicator:error-rate",
        "Cause",
    )
    assert relation is not None
    assert relation.evidence_count == 1


def test_failed_source_update_preserves_previous_projection(
    tmp_path: Path,
):
    manager, pipeline, store, adapter, _ = _manager(tmp_path)
    good = _doc("file://one.txt", "deployment increased error rate")

    manager.sync([good])
    before = store.load()
    before_hash = before.records["file://one.txt"].source_hash

    failed = _doc("file://one.txt", "FAIL")
    report = manager.sync([failed])

    assert report.failed == 1
    assert report.source_results[0].action == IngestionAction.FAILED
    after = store.load()
    assert after.records["file://one.txt"].source_hash == before_hash
    assert len(pipeline.calls) == 2

    relation = adapter.get_relation(
        "event:deployment",
        "indicator:error-rate",
        "Cause",
    )
    assert relation is not None
    assert relation.evidence_count == 1


def test_duplicate_source_uri_is_rejected_before_processing(tmp_path: Path):
    manager, pipeline, _, _, _ = _manager(tmp_path)
    documents = [
        _doc("file://same.txt", "first"),
        _doc("file://same.txt", "second"),
    ]

    with pytest.raises(ValueError, match="duplicate source_uri"):
        manager.sync(documents)

    assert pipeline.calls == []


def test_ingestion_state_detects_record_ledger_divergence(tmp_path: Path):
    manager, _, store, _, _ = _manager(tmp_path)
    manager.sync(
        [_doc("file://one.txt", "deployment increased error rate")]
    )
    payload = store.load().to_dict()
    payload["records"] = []

    from chatbot.knowledge.ingestion.models import IngestionState

    with pytest.raises(
        ValueError,
        match="records and evidence ledger sources differ",
    ):
        IngestionState.from_dict(payload)


class _FailingSaveStore(IngestionStateStore):
    def save(self, state):
        raise OSError("synthetic state persistence failure")


def test_state_save_failure_compensates_relation_state(tmp_path: Path):
    repo = InMemoryGraphRepository()
    tx_manager = KGTransactionManager(repo)
    adapter = DomainKGAdapter(
        repository=repo,
        tx_manager=tx_manager,
        read_only=False,
    )
    manager = SelectiveIngestionManager(
        project_root=PROJECT_ROOT,
        pipeline=_FakeEvidencePipeline(),
        reconciler=EvidenceRelationReconciler(adapter),
        state_store=_FailingSaveStore(
            tmp_path / "knowledge-workspace/ingestion-state.json"
        ),
    )

    with pytest.raises(OSError, match="synthetic state persistence failure"):
        manager.sync(
            [_doc("file://one.txt", "deployment increased error rate")]
        )

    assert adapter.get_relation(
        "event:deployment",
        "indicator:error-rate",
        "Cause",
    ) is None


def test_external_relation_is_not_overwritten_or_deleted(tmp_path: Path):
    manager, _, _, adapter, _ = _manager(tmp_path)
    from chatbot.knowledge.domain.models import DynamicRelation

    external = DynamicRelation(
        relation_id="manual-relation",
        head_id="event:deployment",
        head_name="Deployment",
        tail_id="indicator:error-rate",
        tail_name="Error rate",
        relation_type="Cause",
        sign="-",
        domain_conf=0.88,
        evidence_count=9,
        conflict_count=0,
        origin="manual",
    )
    adapter.upsert_relation(external)

    report = manager.sync(
        [_doc("file://one.txt", "deployment increased error rate")]
    )
    assert report.relation_results[0].action == "preserved_external"

    relation = adapter.get_relation(
        "event:deployment",
        "indicator:error-rate",
        "Cause",
    )
    assert relation is not None
    assert relation.origin == "manual"
    assert relation.domain_conf == 0.88
    assert relation.evidence_count == 9

    manager.sync([])
    relation = adapter.get_relation(
        "event:deployment",
        "indicator:error-rate",
        "Cause",
    )
    assert relation is not None
    assert relation.origin == "manual"


def test_processor_stamp_includes_top_level_knowledge_modules(tmp_path: Path):
    from chatbot.knowledge.ingestion.fingerprint import (
        PROCESSOR_PACKAGES,
        processor_stamp,
    )

    knowledge_root = tmp_path / "src/chatbot/knowledge"
    knowledge_root.mkdir(parents=True)
    (knowledge_root / "settings.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    for package in PROCESSOR_PACKAGES:
        package_root = knowledge_root / package
        package_root.mkdir(parents=True)
        (package_root / "module.py").write_text(
            f"PACKAGE = {package!r}\n",
            encoding="utf-8",
        )

    ontology = tmp_path / "config/ontology"
    ontology.mkdir(parents=True)
    (ontology / "policy.yaml").write_text(
        "policy: one\n",
        encoding="utf-8",
    )

    first = processor_stamp(tmp_path)
    (knowledge_root / "settings.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )
    second = processor_stamp(tmp_path)

    assert first != second
    assert len(first) == 64
    assert len(second) == 64
