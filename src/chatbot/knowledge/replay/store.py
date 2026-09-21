"""Immutable file-backed Knowledge Core snapshot store."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from chatbot.knowledge.ingestion.models import IngestionState
from .models import (
    REPLAY_SCHEMA_VERSION,
    KnowledgeSnapshot,
    ReplayVerificationReport,
    SnapshotChangeSummary,
    SnapshotIndexEntry,
    normalize_utc,
    snapshot_id_for,
    state_digest,
)


class KnowledgeReplayStore:
    """Store immutable canonical ingestion snapshots in a verified chain."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self.snapshots_dir = self.root / "snapshots"

    def record(
        self,
        state: IngestionState,
        *,
        committed_at: datetime,
        change_summary: SnapshotChangeSummary | None = None,
        origin: str = "ingestion_commit",
    ) -> KnowledgeSnapshot:
        committed_at = normalize_utc(committed_at)
        state_copy = IngestionState.from_dict(state.to_dict())
        digest = state_digest(state_copy)
        entries = self._load_verified_entries()
        latest = self._load_snapshot(entries[-1]) if entries else None
        expected_origin = "bootstrap" if latest is None else "ingestion_commit"
        if origin != expected_origin:
            raise ValueError(
                "first replay snapshot must use origin=bootstrap"
                if latest is None
                else "non-first replay snapshots must use origin=ingestion_commit"
            )

        if latest is not None and latest.state_digest == digest:
            return latest

        if latest is not None and committed_at < latest.committed_at:
            raise ValueError(
                "snapshot committed_at must not precede the latest snapshot"
            )

        parent = latest.snapshot_id if latest is not None else None
        sequence = latest.sequence + 1 if latest is not None else 1
        snapshot_id = snapshot_id_for(parent, digest)
        summary = change_summary or SnapshotChangeSummary()
        requested = KnowledgeSnapshot(
            snapshot_id=snapshot_id,
            sequence=sequence,
            committed_at=committed_at,
            parent_snapshot_id=parent,
            state_digest=digest,
            state=state_copy,
            origin=origin,
            change_summary=summary,
        )

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        blob_path = self._snapshot_path(snapshot_id)
        snapshot = requested

        created_blob = False
        if blob_path.exists():
            # Crash recovery: a previous attempt may have atomically written
            # the immutable blob before index replacement. Reuse that blob
            # only when it represents the exact same chain/state payload.
            existing = self._read_snapshot_file(blob_path)
            if (
                existing.snapshot_id != requested.snapshot_id
                or existing.sequence != requested.sequence
                or existing.parent_snapshot_id != requested.parent_snapshot_id
                or existing.state_digest != requested.state_digest
                or existing.state.to_dict() != requested.state.to_dict()
            ):
                raise ValueError(
                    f"conflicting immutable replay snapshot: {snapshot_id}"
                )
            snapshot = existing
        else:
            self._atomic_write_json(blob_path, requested.to_dict())
            created_blob = True

        entry = snapshot.index_entry()
        self._validate_new_entry(entries, entry)
        try:
            self._save_index([*entries, entry])
        except Exception:
            # Ordinary handled failures must not leave a blob representing a
            # transaction that the caller can still compensate. A hard
            # process interruption can still leave an orphan between the
            # atomic blob write and index replace; that path is recovered
            # above only when chain/state identity matches.
            if created_blob:
                blob_path.unlink(missing_ok=True)
            raise
        return snapshot

    def latest(self) -> KnowledgeSnapshot | None:
        entries = self._load_verified_entries()
        return self._load_snapshot(entries[-1]) if entries else None

    def get(self, snapshot_id: str) -> KnowledgeSnapshot:
        entries = self._load_verified_entries()
        by_id = {entry.snapshot_id: entry for entry in entries}
        entry = by_id.get(snapshot_id)
        if entry is None:
            raise KeyError(f"unknown replay snapshot: {snapshot_id}")
        return self._load_snapshot(entry)

    def as_of(self, at: datetime) -> KnowledgeSnapshot | None:
        at = normalize_utc(at)
        entries = self._load_verified_entries()
        candidates = [entry for entry in entries if entry.committed_at <= at]
        if not candidates:
            return None
        selected = max(
            candidates,
            key=lambda entry: (entry.committed_at, entry.sequence),
        )
        return self._load_snapshot(selected)

    def list(self) -> list[SnapshotIndexEntry]:
        return list(self._load_verified_entries())

    def verify(self) -> ReplayVerificationReport:
        entries = self._load_verified_entries()
        return ReplayVerificationReport(
            snapshots=len(entries),
            latest_snapshot_id=entries[-1].snapshot_id if entries else None,
            latest_sequence=entries[-1].sequence if entries else 0,
            state_digests=tuple(entry.state_digest for entry in entries),
        )

    def _load_verified_entries(self) -> list[SnapshotIndexEntry]:
        if not self.index_path.exists():
            return []
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid replay index JSON") from exc
        if value.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported replay index schema")
        raw_entries = value.get("snapshots", [])
        if not isinstance(raw_entries, list):
            raise ValueError("replay index snapshots must be a list")

        entries = [SnapshotIndexEntry.from_dict(item) for item in raw_entries]
        seen_ids: set[str] = set()
        previous: SnapshotIndexEntry | None = None

        for position, entry in enumerate(entries, 1):
            if entry.sequence != position:
                raise ValueError("replay snapshot sequence is not contiguous")
            if entry.snapshot_id in seen_ids:
                raise ValueError("duplicate replay snapshot id")
            seen_ids.add(entry.snapshot_id)

            expected_parent = previous.snapshot_id if previous is not None else None
            if entry.parent_snapshot_id != expected_parent:
                raise ValueError("broken replay snapshot parent chain")
            expected_origin = "bootstrap" if previous is None else "ingestion_commit"
            if entry.origin != expected_origin:
                raise ValueError("invalid replay snapshot origin position")
            if previous is not None and entry.committed_at < previous.committed_at:
                raise ValueError("replay committed_at order regressed")

            snapshot = self._load_snapshot(entry)
            if snapshot.index_entry().to_dict() != entry.to_dict():
                raise ValueError(
                    f"replay snapshot/index mismatch: {entry.snapshot_id}"
                )
            previous = entry

        return entries

    def _load_snapshot(self, entry: SnapshotIndexEntry) -> KnowledgeSnapshot:
        path = self._snapshot_path(entry.snapshot_id)
        if not path.is_file():
            raise ValueError(
                f"missing replay snapshot blob: {entry.snapshot_id}"
            )
        snapshot = self._read_snapshot_file(path)
        if snapshot.snapshot_id != entry.snapshot_id:
            raise ValueError("replay snapshot id does not match index")
        return snapshot

    @staticmethod
    def _validate_new_entry(
        entries: list[SnapshotIndexEntry],
        entry: SnapshotIndexEntry,
    ) -> None:
        if not entries:
            if (
                entry.sequence != 1
                or entry.parent_snapshot_id is not None
                or entry.origin != "bootstrap"
            ):
                raise ValueError("invalid first replay snapshot")
            return

        latest = entries[-1]
        if entry.sequence != latest.sequence + 1:
            raise ValueError("invalid replay snapshot sequence")
        if entry.parent_snapshot_id != latest.snapshot_id:
            raise ValueError("invalid replay snapshot parent")
        if entry.origin != "ingestion_commit":
            raise ValueError("invalid non-first replay snapshot origin")
        if entry.committed_at < latest.committed_at:
            raise ValueError("snapshot committed_at order regressed")

    def _read_snapshot_file(self, path: Path) -> KnowledgeSnapshot:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid replay snapshot JSON: {path.name}") from exc
        return KnowledgeSnapshot.from_dict(value)

    def _snapshot_path(self, snapshot_id: str) -> Path:
        if (
            not snapshot_id.startswith("ksnap_")
            or "/" in snapshot_id
            or "\\" in snapshot_id
            or ".." in snapshot_id
        ):
            raise ValueError("invalid replay snapshot id")
        return self.snapshots_dir / f"{snapshot_id}.json"

    def _save_index(self, entries: list[SnapshotIndexEntry]) -> None:
        payload = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "snapshots": [entry.to_dict() for entry in entries],
        }
        self._atomic_write_json(self.index_path, payload)

    @staticmethod
    def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temp.replace(path)
