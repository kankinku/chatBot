"""Selective ingestion state and report models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chatbot.knowledge.evidence import EvidenceLedger


STATE_SCHEMA_VERSION = 1


class IngestionAction(str, Enum):
    ADDED = "added"
    UPDATED = "updated"
    REPROCESSED = "reprocessed"
    SKIPPED = "skipped"
    REMOVED = "removed"
    FAILED = "failed"


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    source_uri: str
    text: str

    def __post_init__(self) -> None:
        if not self.doc_id.strip():
            raise ValueError("doc_id must not be empty")
        if not self.source_uri.strip():
            raise ValueError("source_uri must not be empty")


@dataclass(frozen=True)
class IngestionRecord:
    doc_id: str
    source_uri: str
    source_hash: str
    projection_hash: str
    processor_stamp: str

    def to_dict(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "source_uri": self.source_uri,
            "source_hash": self.source_hash,
            "projection_hash": self.projection_hash,
            "processor_stamp": self.processor_stamp,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IngestionRecord":
        return cls(
            doc_id=value["doc_id"],
            source_uri=value["source_uri"],
            source_hash=value["source_hash"],
            projection_hash=value["projection_hash"],
            processor_stamp=value["processor_stamp"],
        )


@dataclass
class IngestionState:
    records: dict[str, IngestionRecord] = field(default_factory=dict)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "records": [
                self.records[source_uri].to_dict()
                for source_uri in sorted(self.records)
            ],
            "ledger": self.ledger.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IngestionState":
        if value.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported ingestion state schema")
        records: dict[str, IngestionRecord] = {}
        for item in value.get("records", []):
            record = IngestionRecord.from_dict(item)
            if record.source_uri in records:
                raise ValueError(
                    f"duplicate ingestion source: {record.source_uri}"
                )
            records[record.source_uri] = record

        ledger = EvidenceLedger.from_dict(value.get("ledger", {}))
        state = cls(records=records, ledger=ledger)
        state.validate()
        return state

    def validate(self) -> None:
        record_sources = set(self.records)
        ledger_sources = set(self.ledger.sources())
        if record_sources != ledger_sources:
            raise ValueError(
                "ingestion records and evidence ledger sources differ"
            )

        for source_uri, record in self.records.items():
            projection = self.ledger.get(source_uri)
            if projection is None:
                raise ValueError(
                    f"missing evidence projection: {source_uri}"
                )
            if projection.source_hash != record.source_hash:
                raise ValueError(
                    f"source hash mismatch in ingestion state: {source_uri}"
                )
            if projection.projection_hash != record.projection_hash:
                raise ValueError(
                    f"projection hash mismatch in ingestion state: {source_uri}"
                )
            if len(record.processor_stamp) != 64:
                raise ValueError(
                    f"invalid processor stamp: {source_uri}"
                )


@dataclass(frozen=True)
class RelationSpec:
    node_id: str
    head_id: str
    tail_id: str
    relation_type: str


@dataclass(frozen=True)
class SourceSyncResult:
    source_uri: str
    action: IngestionAction
    source_hash: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RelationReconcileResult:
    relation_node_id: str
    action: str
    evidence_count: int = 0
    conflict_count: int = 0
    domain_conf: float | None = None
    support_score: float = 0.0
    conflict_score: float = 0.0
    support_source_count: int = 0
    conflict_source_count: int = 0
    evidence_score_version: str = "legacy"


@dataclass
class SelectiveIngestionReport:
    source_results: list[SourceSyncResult] = field(default_factory=list)
    relation_results: list[RelationReconcileResult] = field(default_factory=list)

    @property
    def processed(self) -> int:
        return sum(
            result.action
            in {
                IngestionAction.ADDED,
                IngestionAction.UPDATED,
                IngestionAction.REPROCESSED,
            }
            for result in self.source_results
        )

    @property
    def skipped(self) -> int:
        return sum(
            result.action == IngestionAction.SKIPPED
            for result in self.source_results
        )

    @property
    def removed(self) -> int:
        return sum(
            result.action == IngestionAction.REMOVED
            for result in self.source_results
        )

    @property
    def failed(self) -> int:
        return sum(
            result.action == IngestionAction.FAILED
            for result in self.source_results
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "removed": self.removed,
            "failed": self.failed,
            "sources": [
                {
                    "source_uri": result.source_uri,
                    "action": result.action.value,
                    "source_hash": result.source_hash,
                    "reason": result.reason,
                }
                for result in self.source_results
            ],
            "relations": [
                {
                    "relation_node_id": result.relation_node_id,
                    "action": result.action,
                    "evidence_count": result.evidence_count,
                    "conflict_count": result.conflict_count,
                    "domain_conf": result.domain_conf,
                    "support_score": result.support_score,
                    "conflict_score": result.conflict_score,
                    "support_source_count": result.support_source_count,
                    "conflict_source_count": result.conflict_source_count,
                    "evidence_score_version": result.evidence_score_version,
                }
                for result in self.relation_results
            ],
        }
