"""Immutable Knowledge Core snapshot models for as-of replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from chatbot.knowledge.ingestion.models import IngestionState
from chatbot.knowledge.workspace.hashing import hash_value


REPLAY_SCHEMA_VERSION = 1
SNAPSHOT_PREFIX = "ksnap_"
SNAPSHOT_ORIGINS = {"bootstrap", "ingestion_commit"}


def normalize_utc(value: datetime) -> datetime:
    """Require a timezone-aware datetime and normalize it to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("replay datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def datetime_to_text(value: datetime) -> str:
    return normalize_utc(value).isoformat().replace("+00:00", "Z")


def datetime_from_text(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid replay datetime: {value}") from exc
    return normalize_utc(parsed)


def state_digest(state: IngestionState) -> str:
    state.validate()
    return hash_value(state.to_dict())


def snapshot_id_for(
    parent_snapshot_id: str | None,
    digest: str,
) -> str:
    if len(digest) != 64:
        raise ValueError("state digest must be a SHA-256 hex digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError("state digest must be a SHA-256 hex digest") from exc
    return SNAPSHOT_PREFIX + hash_value(
        {
            "parent_snapshot_id": parent_snapshot_id,
            "state_digest": digest,
        }
    )


@dataclass(frozen=True)
class SnapshotChangeSummary:
    action_counts: dict[str, int] = field(default_factory=dict)
    source_uris: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_counts: dict[str, int] = {}
        for key, count in sorted(self.action_counts.items()):
            name = str(key).strip()
            if not name:
                raise ValueError("snapshot action name must not be empty")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("snapshot action counts must be non-negative integers")
            if count:
                normalized_counts[name] = count
        normalized_sources = tuple(sorted({str(item) for item in self.source_uris}))
        object.__setattr__(self, "action_counts", normalized_counts)
        object.__setattr__(self, "source_uris", normalized_sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_counts": dict(self.action_counts),
            "source_uris": list(self.source_uris),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "SnapshotChangeSummary":
        value = value or {}
        return cls(
            action_counts={
                str(key): int(count)
                for key, count in value.get("action_counts", {}).items()
            },
            source_uris=tuple(str(item) for item in value.get("source_uris", [])),
        )


@dataclass(frozen=True)
class SnapshotIndexEntry:
    sequence: int
    snapshot_id: str
    parent_snapshot_id: str | None
    committed_at: datetime
    state_digest: str
    origin: str
    change_summary: SnapshotChangeSummary = field(
        default_factory=SnapshotChangeSummary
    )

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("snapshot sequence must be >= 1")
        if not self.snapshot_id.startswith(SNAPSHOT_PREFIX):
            raise ValueError("invalid snapshot id")
        if self.origin not in SNAPSHOT_ORIGINS:
            raise ValueError(f"unsupported snapshot origin: {self.origin}")
        object.__setattr__(self, "committed_at", normalize_utc(self.committed_at))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "snapshot_id": self.snapshot_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "committed_at": datetime_to_text(self.committed_at),
            "state_digest": self.state_digest,
            "origin": self.origin,
            "change_summary": self.change_summary.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SnapshotIndexEntry":
        return cls(
            sequence=int(value["sequence"]),
            snapshot_id=str(value["snapshot_id"]),
            parent_snapshot_id=(
                str(value["parent_snapshot_id"])
                if value.get("parent_snapshot_id") is not None
                else None
            ),
            committed_at=datetime_from_text(str(value["committed_at"])),
            state_digest=str(value["state_digest"]),
            origin=str(value["origin"]),
            change_summary=SnapshotChangeSummary.from_dict(
                value.get("change_summary")
            ),
        )


@dataclass(frozen=True)
class KnowledgeSnapshot:
    snapshot_id: str
    sequence: int
    committed_at: datetime
    parent_snapshot_id: str | None
    state_digest: str
    state: IngestionState
    origin: str
    change_summary: SnapshotChangeSummary = field(
        default_factory=SnapshotChangeSummary
    )
    schema_version: int = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported replay snapshot schema")
        object.__setattr__(self, "committed_at", normalize_utc(self.committed_at))
        entry = self.index_entry()
        expected_digest = state_digest(self.state)
        if self.state_digest != expected_digest:
            raise ValueError("replay snapshot state digest mismatch")
        expected_id = snapshot_id_for(
            self.parent_snapshot_id,
            self.state_digest,
        )
        if self.snapshot_id != expected_id:
            raise ValueError("replay snapshot id mismatch")
        if entry.sequence != self.sequence:
            raise ValueError("replay snapshot sequence mismatch")

    def index_entry(self) -> SnapshotIndexEntry:
        return SnapshotIndexEntry(
            sequence=self.sequence,
            snapshot_id=self.snapshot_id,
            parent_snapshot_id=self.parent_snapshot_id,
            committed_at=self.committed_at,
            state_digest=self.state_digest,
            origin=self.origin,
            change_summary=self.change_summary,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            **self.index_entry().to_dict(),
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeSnapshot":
        if value.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported replay snapshot schema")
        return cls(
            schema_version=REPLAY_SCHEMA_VERSION,
            snapshot_id=str(value["snapshot_id"]),
            sequence=int(value["sequence"]),
            committed_at=datetime_from_text(str(value["committed_at"])),
            parent_snapshot_id=(
                str(value["parent_snapshot_id"])
                if value.get("parent_snapshot_id") is not None
                else None
            ),
            state_digest=str(value["state_digest"]),
            state=IngestionState.from_dict(value["state"]),
            origin=str(value["origin"]),
            change_summary=SnapshotChangeSummary.from_dict(
                value.get("change_summary")
            ),
        )


@dataclass(frozen=True)
class ReplayVerificationReport:
    snapshots: int
    latest_snapshot_id: str | None
    latest_sequence: int
    state_digests: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshots": self.snapshots,
            "latest_snapshot_id": self.latest_snapshot_id,
            "latest_sequence": self.latest_sequence,
            "state_digests": list(self.state_digests),
        }
