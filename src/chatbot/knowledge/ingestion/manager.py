"""Selective source ingestion over the evidence provenance substrate."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from chatbot.knowledge.evidence import EvidenceProvenancePipeline

from .fingerprint import default_project_root, processor_stamp
from .models import (
    IngestionAction,
    IngestionRecord,
    IngestionState,
    SelectiveIngestionReport,
    SourceDocument,
    SourceSyncResult,
)
from .reconciler import (
    EvidenceRelationReconciler,
    relation_specs_from_graph,
)
from .state import IngestionStateStore


class SelectiveIngestionManager:
    """Process only new/stale sources and reconcile affected relations."""

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        pipeline: EvidenceProvenancePipeline | None = None,
        reconciler: EvidenceRelationReconciler | None = None,
        state_store: IngestionStateStore | None = None,
        use_llm: bool = False,
    ):
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else default_project_root()
        )
        self.pipeline = pipeline or EvidenceProvenancePipeline(
            use_llm=use_llm,
            apply_domain_updates=False,
        )
        self.reconciler = reconciler or EvidenceRelationReconciler()
        self.state_store = state_store or IngestionStateStore(
            self.project_root
            / "knowledge-workspace"
            / "ingestion-state.json"
        )

    def sync(
        self,
        documents: Iterable[SourceDocument],
        *,
        prune_missing: bool = True,
        prune_scope: set[str] | None = None,
    ) -> SelectiveIngestionReport:
        docs = sorted(documents, key=lambda item: item.source_uri)
        by_uri: dict[str, SourceDocument] = {}
        for document in docs:
            if document.source_uri in by_uri:
                raise ValueError(
                    f"duplicate source_uri in ingestion batch: "
                    f"{document.source_uri}"
                )
            by_uri[document.source_uri] = document

        state = self.state_store.load()
        backup = IngestionState.from_dict(state.to_dict())
        stamp = processor_stamp(self.project_root)
        report = SelectiveIngestionReport()
        affected_specs = {}

        for source_uri in sorted(by_uri):
            document = by_uri[source_uri]
            source_hash = hashlib.sha256(
                document.text.encode("utf-8")
            ).hexdigest()
            record = state.records.get(source_uri)
            previous = state.ledger.get(source_uri)

            if (
                record is not None
                and previous is not None
                and record.source_hash == source_hash
                and previous.source_hash == source_hash
                and record.projection_hash == previous.projection_hash
                and record.processor_stamp == stamp
            ):
                report.source_results.append(
                    SourceSyncResult(
                        source_uri=source_uri,
                        action=IngestionAction.SKIPPED,
                        source_hash=source_hash,
                        reason="source_and_processor_unchanged",
                    )
                )
                continue

            if previous is not None:
                affected_specs.update(
                    relation_specs_from_graph(previous.graph)
                )

            try:
                processed = self.pipeline.process(
                    raw_text=document.text,
                    doc_id=document.doc_id,
                    source_uri=source_uri,
                )
            except Exception as exc:
                report.source_results.append(
                    SourceSyncResult(
                        source_uri=source_uri,
                        action=IngestionAction.FAILED,
                        source_hash=source_hash,
                        reason=str(exc),
                    )
                )
                continue

            projection = processed.projection
            if projection.source_hash != source_hash:
                raise ValueError(
                    f"pipeline source hash mismatch: {source_uri}"
                )
            if projection.source_uri != source_uri:
                raise ValueError(
                    f"pipeline source uri mismatch: {source_uri}"
                )

            plan = state.ledger.upsert(projection)
            affected_specs.update(
                relation_specs_from_graph(projection.graph)
            )
            state.records[source_uri] = IngestionRecord(
                doc_id=document.doc_id,
                source_uri=source_uri,
                source_hash=projection.source_hash,
                projection_hash=projection.projection_hash,
                processor_stamp=stamp,
            )

            if record is None:
                action = IngestionAction.ADDED
            elif record.source_hash != source_hash:
                action = IngestionAction.UPDATED
            else:
                action = IngestionAction.REPROCESSED

            report.source_results.append(
                SourceSyncResult(
                    source_uri=source_uri,
                    action=action,
                    source_hash=source_hash,
                    reason=plan.reason,
                )
            )

        if prune_missing:
            incoming = set(by_uri)
            stored = set(state.records)
            if prune_scope is None:
                missing = stored - incoming
            else:
                missing = (stored & set(prune_scope)) - incoming
            for source_uri in sorted(missing):
                previous = state.ledger.get(source_uri)
                if previous is not None:
                    affected_specs.update(
                        relation_specs_from_graph(previous.graph)
                    )
                plan = state.ledger.remove(source_uri)
                state.records.pop(source_uri, None)
                report.source_results.append(
                    SourceSyncResult(
                        source_uri=source_uri,
                        action=IngestionAction.REMOVED,
                        source_hash=plan.previous_source_hash,
                        reason=plan.reason,
                    )
                )

        state.validate()

        try:
            report.relation_results = self.reconciler.reconcile(
                state.ledger,
                affected_specs,
            )
            self.state_store.save(state)
        except Exception as primary_error:
            # Relation reconciliation is transactional. If persistence fails
            # after a successful relation commit, replay the previous state
            # through the same deterministic reconciler as compensation.
            try:
                self.reconciler.reconcile(
                    backup.ledger,
                    affected_specs,
                )
            except Exception as compensation_error:
                raise RuntimeError(
                    "selective ingestion failed and compensating "
                    f"reconciliation also failed: {primary_error}"
                ) from compensation_error
            raise

        return report
