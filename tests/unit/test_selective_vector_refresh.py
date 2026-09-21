"""Selective vector-index refresh contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chatbot.core.types import Chunk
from chatbot.embedding.base_embedder import BaseEmbedder
from chatbot.retrieval.vector_index_sync import (
    SelectiveVectorIndexSynchronizer,
    VectorIndexManifestStore,
    chunk_vector_id,
)


class _FakeEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "fake-v1", dim: int = 3):
        self._model_name = model_name
        self._dim = dim
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            checksum = sum(ord(char) for char in text)
            base = [
                float(len(text)),
                float(checksum % 97),
                float(checksum % 193),
            ]
            rows.append(
                [base[index % len(base)] for index in range(self._dim)]
            )
        return np.asarray(rows, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name


class _FakeCollection:
    def __init__(self):
        self.records: dict[str, dict] = {}
        self.upsert_calls: list[list[str]] = []
        self.update_calls: list[list[str]] = []
        self.delete_calls: list[list[str]] = []
        self.fail_after_partial_upsert = False

    def get(self):
        return {"ids": sorted(self.records)}

    def upsert(self, *, ids, embeddings, documents, metadatas):
        self.upsert_calls.append(list(ids))
        for index, chunk_id in enumerate(ids):
            self.records[chunk_id] = {
                "embedding": embeddings[index],
                "document": documents[index],
                "metadata": dict(metadatas[index]),
            }
            if self.fail_after_partial_upsert:
                self.fail_after_partial_upsert = False
                raise RuntimeError("synthetic partial upsert failure")

    def update(self, *, ids, metadatas):
        self.update_calls.append(list(ids))
        for index, chunk_id in enumerate(ids):
            self.records[chunk_id]["metadata"] = dict(metadatas[index])

    def delete(self, *, ids):
        self.delete_calls.append(list(ids))
        for chunk_id in ids:
            self.records.pop(chunk_id, None)


def _chunk(
    doc_id: str,
    text: str,
    *,
    start: int = 0,
    filename: str | None = None,
) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        filename=filename or f"{doc_id}.txt",
        page=None,
        start_offset=start,
        length=len(text),
        text=text,
        extra={"source_uri": f"file:{doc_id}.txt"},
    )


def _syncer(
    tmp_path: Path,
    collection: _FakeCollection,
    embedder: _FakeEmbedder,
    *,
    reset_collection=None,
):
    return SelectiveVectorIndexSynchronizer(
        collection=collection,
        collection_name="chunks",
        embedder=embedder,
        manifest_store=VectorIndexManifestStore(
            tmp_path / "chunks.manifest.json"
        ),
        reset_collection=reset_collection,
    )


def test_chunk_vector_id_is_stable_across_text_change_and_list_order():
    first = _chunk("one", "alpha", start=10)
    changed = _chunk("one", "beta", start=10)

    assert chunk_vector_id(first) == chunk_vector_id(changed)


def test_initial_sync_then_unchanged_sync_performs_no_vector_writes(
    tmp_path: Path,
):
    collection = _FakeCollection()
    embedder = _FakeEmbedder()
    syncer = _syncer(tmp_path, collection, embedder)
    chunks = [_chunk("one", "alpha"), _chunk("two", "beta")]

    first = syncer.sync(chunks)
    second = syncer.sync(chunks)

    assert first.embedded == 2
    assert second.embedded == 0
    assert second.metadata_updated == 0
    assert second.removed == 0
    assert second.unchanged == 2
    assert embedder.calls == [["alpha", "beta"]]
    assert len(collection.upsert_calls) == 1
    assert collection.update_calls == []
    assert collection.delete_calls == []


def test_one_changed_chunk_reembeds_only_that_chunk(tmp_path: Path):
    collection = _FakeCollection()
    embedder = _FakeEmbedder()
    syncer = _syncer(tmp_path, collection, embedder)

    syncer.sync([_chunk("one", "alpha"), _chunk("two", "beta")])
    report = syncer.sync(
        [_chunk("one", "alpha changed"), _chunk("two", "beta")]
    )

    assert report.embedded == 1
    assert report.unchanged == 1
    assert embedder.calls[-1] == ["alpha changed"]
    assert len(collection.upsert_calls[-1]) == 1


def test_reorder_updates_metadata_without_reembedding(tmp_path: Path):
    collection = _FakeCollection()
    embedder = _FakeEmbedder()
    syncer = _syncer(tmp_path, collection, embedder)
    one = _chunk("one", "alpha")
    two = _chunk("two", "beta")

    syncer.sync([one, two])
    report = syncer.sync([two, one])

    assert report.embedded == 0
    assert report.metadata_updated == 2
    assert len(embedder.calls) == 1
    assert len(collection.update_calls[-1]) == 2
    assert collection.records[chunk_vector_id(two)]["metadata"]["chunk_index"] == "0"
    assert collection.records[chunk_vector_id(one)]["metadata"]["chunk_index"] == "1"


def test_removed_chunk_only_removes_stale_vector(tmp_path: Path):
    collection = _FakeCollection()
    embedder = _FakeEmbedder()
    syncer = _syncer(tmp_path, collection, embedder)
    one = _chunk("one", "alpha")
    two = _chunk("two", "beta")

    syncer.sync([one, two])
    report = syncer.sync([two])

    assert report.embedded == 0
    assert report.removed == 1
    assert collection.delete_calls[-1] == [chunk_vector_id(one)]
    assert set(collection.records) == {chunk_vector_id(two)}


def test_embedder_change_refreshes_all_current_embeddings(tmp_path: Path):
    collection = _FakeCollection()
    first_embedder = _FakeEmbedder("fake-v1")
    chunks = [_chunk("one", "alpha"), _chunk("two", "beta")]
    _syncer(tmp_path, collection, first_embedder).sync(chunks)

    second_embedder = _FakeEmbedder("fake-v2")
    report = _syncer(tmp_path, collection, second_embedder).sync(chunks)

    assert report.rebuilt is True
    assert report.embedded == 2
    assert second_embedder.calls == [["alpha", "beta"]]
    assert set(collection.records) == {
        chunk_vector_id(chunks[0]),
        chunk_vector_id(chunks[1]),
    }


def test_embedding_dimension_change_resets_collection(tmp_path: Path):
    original_collection = _FakeCollection()
    chunk = _chunk("one", "alpha")
    _syncer(
        tmp_path,
        original_collection,
        _FakeEmbedder("fake-v1", dim=3),
    ).sync([chunk])

    replacement_collection = _FakeCollection()
    reset_calls: list[bool] = []

    def reset_collection():
        reset_calls.append(True)
        return replacement_collection

    second_embedder = _FakeEmbedder("fake-v2", dim=4)
    report = _syncer(
        tmp_path,
        original_collection,
        second_embedder,
        reset_collection=reset_collection,
    ).sync([chunk])

    assert report.rebuilt is True
    assert report.embedded == 1
    assert reset_calls == [True]
    assert second_embedder.calls == [["alpha"]]
    assert set(replacement_collection.records) == {chunk_vector_id(chunk)}
    assert len(
        replacement_collection.records[chunk_vector_id(chunk)]["embedding"]
    ) == 4


def test_missing_manifest_with_legacy_ids_recovers_by_rebuilding(
    tmp_path: Path,
):
    collection = _FakeCollection()
    collection.records["legacy-random-id"] = {
        "embedding": [0.0, 0.0, 0.0],
        "document": "legacy",
        "metadata": {},
    }
    embedder = _FakeEmbedder()
    chunk = _chunk("one", "alpha")
    replacement = _FakeCollection()
    reset_calls: list[bool] = []

    def reset_collection():
        reset_calls.append(True)
        return replacement

    report = _syncer(
        tmp_path,
        collection,
        embedder,
        reset_collection=reset_collection,
    ).sync([chunk])

    assert report.rebuilt is True
    assert report.embedded == 1
    assert report.removed == 0
    assert reset_calls == [True]
    assert set(replacement.records) == {chunk_vector_id(chunk)}


def test_partial_mutation_failure_does_not_advance_manifest(tmp_path: Path):
    collection = _FakeCollection()
    embedder = _FakeEmbedder()
    syncer = _syncer(tmp_path, collection, embedder)
    original = _chunk("one", "alpha")
    syncer.sync([original])

    manifest_path = tmp_path / "chunks.manifest.json"
    before = manifest_path.read_text(encoding="utf-8")

    collection.fail_after_partial_upsert = True
    changed = _chunk("one", "alpha changed")
    with pytest.raises(RuntimeError, match="partial upsert"):
        syncer.sync([changed])

    assert manifest_path.read_text(encoding="utf-8") == before

    report = syncer.sync([changed])
    assert report.embedded == 1
    assert collection.records[chunk_vector_id(changed)]["document"] == (
        "alpha changed"
    )


def test_duplicate_deterministic_chunk_identity_is_rejected(tmp_path: Path):
    collection = _FakeCollection()
    embedder = _FakeEmbedder()
    syncer = _syncer(tmp_path, collection, embedder)
    first = _chunk("one", "alpha")
    duplicate = _chunk("one", "different text")

    with pytest.raises(ValueError, match="duplicate deterministic"):
        syncer.sync([first, duplicate])

    assert embedder.calls == []
    assert collection.records == {}
