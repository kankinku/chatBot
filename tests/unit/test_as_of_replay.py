"""As-of Knowledge Core snapshot and replay contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from chatbot.knowledge.evidence import EvidenceLedger, EvidenceProjection
from chatbot.knowledge.ingestion.models import IngestionRecord, IngestionState
from chatbot.knowledge.replay import (
    KnowledgeReplayService,
    KnowledgeReplayStore,
    SnapshotChangeSummary,
    snapshot_id_for,
    state_digest,
)
from chatbot.knowledge.workspace.hashing import hash_value
from chatbot.knowledge.workspace.models import (
    Confidence,
    MetaRelation,
    NodeKind,
    SourceRef,
    WorkspaceEdge,
    WorkspaceGraph,
    WorkspaceNode,
)


UTC = timezone.utc


def _state(
    source_uri: str | None = "file:one.txt",
    text: str = "alpha",
    *,
    processor_stamp: str = "a" * 64,
) -> IngestionState:
    if source_uri is None:
        return IngestionState()

    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source_ref = SourceRef(path=source_uri, hash=source_hash)
    document_id = "document:" + hash_value({"source_uri": source_uri})[:20]
    fragment_id = "fragment:" + hash_value(
        {"source_uri": source_uri, "source_hash": source_hash, "text": text}
    )[:24]
    fragment_props = {"text": text, "source_start": 0, "source_end": len(text)}
    graph = WorkspaceGraph(
        nodes=[
            WorkspaceNode(
                id=document_id,
                label=source_uri,
                kind=NodeKind.DOCUMENT,
                content_hash=source_hash,
                sources=[source_ref],
                properties={"doc_id": source_uri, "source_uri": source_uri},
            ),
            WorkspaceNode(
                id=fragment_id,
                label=text,
                kind=NodeKind.FRAGMENT,
                content_hash=hash_value(fragment_props),
                sources=[source_ref],
                properties=fragment_props,
            ),
        ],
        edges=[
            WorkspaceEdge(
                source=document_id,
                target=fragment_id,
                relation=MetaRelation.PRODUCES,
                confidence=Confidence.EXTRACTED,
                content_hash=hash_value(
                    {
                        "source": document_id,
                        "target": fragment_id,
                        "relation": MetaRelation.PRODUCES.value,
                    }
                ),
                sources=[source_ref],
            )
        ],
    )
    projection = EvidenceProjection(
        source_uri=source_uri,
        source_hash=source_hash,
        document_node_id=document_id,
        projection_hash=hash_value(graph.to_dict()),
        graph=graph,
    )
    ledger = EvidenceLedger()
    ledger.upsert(projection)
    state = IngestionState(
        records={
            source_uri: IngestionRecord(
                doc_id=source_uri,
                source_uri=source_uri,
                source_hash=source_hash,
                projection_hash=projection.projection_hash,
                processor_stamp=processor_stamp,
            )
        },
        ledger=ledger,
    )
    state.validate()
    return state


def _record(
    store: KnowledgeReplayStore,
    state: IngestionState,
    at: datetime,
    *,
    origin: str = "ingestion_commit",
):
    return store.record(
        state,
        committed_at=at,
        change_summary=SnapshotChangeSummary(
            action_counts={"updated": 1},
            source_uris=tuple(state.records),
        ),
        origin=origin,
    )


def test_bootstrap_snapshot_and_deterministic_chain_identity(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    state = _state()
    at = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)

    snapshot = _record(store, state, at, origin="bootstrap")

    digest = state_digest(state)
    assert snapshot.state_digest == digest
    assert snapshot.snapshot_id == snapshot_id_for(None, digest)
    assert snapshot.sequence == 1
    assert snapshot.parent_snapshot_id is None
    assert snapshot.origin == "bootstrap"
    assert store.verify().snapshots == 1


def test_unchanged_state_does_not_append_duplicate_snapshot(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    state = _state()
    first = _record(
        store,
        state,
        datetime(2026, 9, 21, 10, 0, tzinfo=UTC),
        origin="bootstrap",
    )
    second = _record(
        store,
        IngestionState.from_dict(state.to_dict()),
        datetime(2026, 9, 21, 11, 0, tzinfo=UTC),
    )

    assert second.snapshot_id == first.snapshot_id
    assert len(store.list()) == 1


def test_add_update_remove_history_and_as_of_resolution(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    t1 = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)
    t2 = t1 + timedelta(hours=1)
    t3 = t2 + timedelta(hours=1)

    first = _record(store, _state(text="alpha"), t1, origin="bootstrap")
    second = _record(store, _state(text="beta"), t2)
    third = _record(store, _state(source_uri=None), t3)

    assert first.snapshot_id != second.snapshot_id != third.snapshot_id
    assert second.parent_snapshot_id == first.snapshot_id
    assert third.parent_snapshot_id == second.snapshot_id

    before = store.as_of(t1 - timedelta(seconds=1))
    between = store.as_of(t2 - timedelta(seconds=1))
    after_update = store.as_of(t2)
    after_remove = store.as_of(t3)

    assert before is None
    assert between is not None
    assert between.snapshot_id == first.snapshot_id
    assert after_update is not None
    assert after_update.snapshot_id == second.snapshot_id
    assert after_remove is not None
    assert after_remove.state.records == {}


def test_same_commit_time_uses_sequence_as_tie_break(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    at = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)

    first = _record(store, _state(text="alpha"), at, origin="bootstrap")
    second = _record(store, _state(text="beta"), at)

    selected = store.as_of(at)
    assert selected is not None
    assert first.sequence == 1
    assert second.sequence == 2
    assert selected.snapshot_id == second.snapshot_id


def test_naive_datetime_is_rejected(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")

    with pytest.raises(ValueError, match="timezone-aware"):
        _record(store, _state(), datetime(2026, 9, 21, 10, 0))

    with pytest.raises(ValueError, match="timezone-aware"):
        store.as_of(datetime(2026, 9, 21, 10, 0))


def test_corrupt_snapshot_state_digest_is_rejected(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    snapshot = _record(
        store,
        _state(),
        datetime(2026, 9, 21, 10, 0, tzinfo=UTC),
        origin="bootstrap",
    )
    path = store.snapshots_dir / f"{snapshot.snapshot_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["state_digest"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="state digest mismatch"):
        store.verify()


def test_corrupt_snapshot_id_is_rejected(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    snapshot = _record(
        store,
        _state(),
        datetime(2026, 9, 21, 10, 0, tzinfo=UTC),
        origin="bootstrap",
    )
    path = store.snapshots_dir / f"{snapshot.snapshot_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["snapshot_id"] = "ksnap_" + "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="snapshot id mismatch"):
        store.verify()


def test_broken_parent_chain_and_missing_blob_are_rejected(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    t1 = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)
    first = _record(store, _state(text="alpha"), t1, origin="bootstrap")
    second = _record(store, _state(text="beta"), t1 + timedelta(hours=1))

    index = json.loads(store.index_path.read_text(encoding="utf-8"))
    index["snapshots"][1]["parent_snapshot_id"] = None
    store.index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="parent chain"):
        store.list()

    # Restore the valid index, then prove missing immutable blobs fail closed.
    index["snapshots"][1]["parent_snapshot_id"] = first.snapshot_id
    store.index_path.write_text(json.dumps(index), encoding="utf-8")
    (store.snapshots_dir / f"{second.snapshot_id}.json").unlink()
    with pytest.raises(ValueError, match="missing replay snapshot blob"):
        store.verify()


def test_conflicting_orphan_blob_is_not_overwritten(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    state = _state()
    digest = state_digest(state)
    snapshot_id = snapshot_id_for(None, digest)
    store.snapshots_dir.mkdir(parents=True)
    conflict = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "sequence": 99,
        "parent_snapshot_id": None,
        "committed_at": "2026-09-21T10:00:00Z",
        "state_digest": digest,
        "origin": "bootstrap",
        "change_summary": {"action_counts": {}, "source_uris": []},
        "state": state.to_dict(),
    }
    (store.snapshots_dir / f"{snapshot_id}.json").write_text(
        json.dumps(conflict),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting immutable"):
        _record(
            store,
            state,
            datetime(2026, 9, 21, 11, 0, tzinfo=UTC),
            origin="bootstrap",
        )


def test_replay_preserves_old_fragment_text_and_returns_defensive_state(
    tmp_path: Path,
):
    store = KnowledgeReplayStore(tmp_path / "replay")
    service = KnowledgeReplayService(store)
    t1 = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)
    first = _record(store, _state(text="old text"), t1, origin="bootstrap")
    _record(store, _state(text="new text"), t1 + timedelta(hours=1))

    replay = service.state_by_snapshot(first.snapshot_id)
    fragment_texts = [
        node.properties["text"]
        for node in replay.merged_graph().nodes
        if node.kind == NodeKind.FRAGMENT
    ]
    assert fragment_texts == ["old text"]

    copy_one = replay.state
    copy_one.records.clear()
    copy_two = replay.state
    assert list(copy_two.records) == ["file:one.txt"]


def test_processor_stamp_change_changes_state_digest(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    at = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)
    first = _record(
        store,
        _state(processor_stamp="a" * 64),
        at,
        origin="bootstrap",
    )
    second = _record(
        store,
        _state(processor_stamp="b" * 64),
        at + timedelta(hours=1),
    )

    assert first.state_digest != second.state_digest
    assert second.sequence == 2


def test_commit_time_cannot_move_backwards(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    first_at = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)
    _record(store, _state(text="alpha"), first_at, origin="bootstrap")

    with pytest.raises(ValueError, match="must not precede"):
        _record(
            store,
            _state(text="beta"),
            first_at - timedelta(seconds=1),
        )


def test_corrupt_index_sequence_and_schema_are_rejected(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    _record(
        store,
        _state(),
        datetime(2026, 9, 21, 10, 0, tzinfo=UTC),
        origin="bootstrap",
    )
    valid = json.loads(store.index_path.read_text(encoding="utf-8"))

    invalid_sequence = json.loads(json.dumps(valid))
    invalid_sequence["snapshots"][0]["sequence"] = 2
    store.index_path.write_text(
        json.dumps(invalid_sequence),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sequence is not contiguous"):
        store.verify()

    invalid_schema = json.loads(json.dumps(valid))
    invalid_schema["schema_version"] = 999
    store.index_path.write_text(json.dumps(invalid_schema), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported replay index schema"):
        store.verify()


def test_valid_orphan_blob_is_reused_after_index_write_interruption(
    tmp_path: Path,
):
    store = KnowledgeReplayStore(tmp_path / "replay")
    state = _state()
    at = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)
    digest = state_digest(state)
    snapshot_id = snapshot_id_for(None, digest)
    store.snapshots_dir.mkdir(parents=True)
    orphan = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "sequence": 1,
        "parent_snapshot_id": None,
        "committed_at": "2026-09-21T10:00:00Z",
        "state_digest": digest,
        "origin": "bootstrap",
        "change_summary": {"action_counts": {}, "source_uris": []},
        "state": state.to_dict(),
    }
    blob_path = store.snapshots_dir / f"{snapshot_id}.json"
    blob_path.write_text(json.dumps(orphan), encoding="utf-8")

    recovered = _record(
        store,
        state,
        at + timedelta(minutes=5),
        origin="bootstrap",
    )

    assert recovered.snapshot_id == snapshot_id
    assert recovered.committed_at == at
    assert len(store.list()) == 1


def test_snapshot_origin_position_is_enforced(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    at = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="first replay snapshot must use origin=bootstrap"):
        _record(store, _state(text="alpha"), at, origin="ingestion_commit")

    first = _record(store, _state(text="alpha"), at, origin="bootstrap")
    assert first.origin == "bootstrap"

    with pytest.raises(
        ValueError,
        match="non-first replay snapshots must use origin=ingestion_commit",
    ):
        _record(
            store,
            _state(text="beta"),
            at + timedelta(hours=1),
            origin="bootstrap",
        )


def test_corrupt_snapshot_origin_position_is_rejected(tmp_path: Path):
    store = KnowledgeReplayStore(tmp_path / "replay")
    at = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)
    _record(store, _state(text="alpha"), at, origin="bootstrap")
    _record(store, _state(text="beta"), at + timedelta(hours=1))

    index = json.loads(store.index_path.read_text(encoding="utf-8"))
    index["snapshots"][1]["origin"] = "bootstrap"
    store.index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid replay snapshot origin position"):
        store.verify()


def test_explicit_bootstrap_requires_existing_state_file(tmp_path: Path):
    from scripts.knowledge_replay import _load_bootstrap_state

    missing = tmp_path / "missing-ingestion-state.json"
    with pytest.raises(FileNotFoundError, match="bootstrap state file not found"):
        _load_bootstrap_state(str(missing))
