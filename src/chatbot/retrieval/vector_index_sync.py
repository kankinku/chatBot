"""Selective synchronization for the derived Chroma vector index."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from chatbot.core.types import Chunk
from chatbot.embedding.base_embedder import BaseEmbedder


VECTOR_INDEX_SCHEMA_VERSION = 1
VECTOR_INDEX_BATCH_SIZE = 100


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def chunk_vector_id(chunk: Chunk) -> str:
    """Return a stable vector ID independent of list ordering and text."""
    source_identity = chunk.extra.get("source_uri") or chunk.doc_id
    identity = {
        "source": str(source_identity),
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page": chunk.page,
        "start_offset": chunk.start_offset,
    }
    return f"chunk_{_sha256_json(identity)}"


def chunk_content_hash(chunk: Chunk) -> str:
    return hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


def chunk_metadata(chunk: Chunk, chunk_index: int) -> dict[str, str]:
    metadata = {
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page": str(chunk.page) if chunk.page is not None else "None",
        "start_offset": str(chunk.start_offset),
        "length": str(chunk.length),
        "chunk_index": str(chunk_index),
    }
    for key, value in sorted(chunk.extra.items()):
        metadata[f"extra_{key}"] = str(value)
    return metadata


def embedder_stamp(embedder: BaseEmbedder) -> str:
    """Fingerprint embedding semantics, not batching/runtime placement."""
    payload: dict[str, Any] = {
        "class": (
            f"{embedder.__class__.__module__}."
            f"{embedder.__class__.__qualname__}"
        ),
        "model_name": str(embedder.model_name),
        "dim": int(embedder.dim),
    }
    config = getattr(embedder, "config", None)
    if config is not None and hasattr(config, "normalize_embeddings"):
        payload["normalize_embeddings"] = bool(
            getattr(config, "normalize_embeddings")
        )
    return _sha256_json(payload)


@dataclass(frozen=True)
class VectorIndexRecord:
    chunk_id: str
    content_hash: str
    metadata_hash: str

    def to_dict(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "content_hash": self.content_hash,
            "metadata_hash": self.metadata_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VectorIndexRecord":
        return cls(
            chunk_id=str(value["chunk_id"]),
            content_hash=str(value["content_hash"]),
            metadata_hash=str(value["metadata_hash"]),
        )


@dataclass
class VectorIndexManifest:
    collection_name: str
    embedder_stamp: str
    embedding_dim: int
    records: dict[str, VectorIndexRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
            "collection_name": self.collection_name,
            "embedder_stamp": self.embedder_stamp,
            "embedding_dim": self.embedding_dim,
            "records": [
                self.records[chunk_id].to_dict()
                for chunk_id in sorted(self.records)
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VectorIndexManifest":
        if value.get("schema_version") != VECTOR_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported vector-index manifest schema")
        records: dict[str, VectorIndexRecord] = {}
        for raw in value.get("records", []):
            record = VectorIndexRecord.from_dict(raw)
            if record.chunk_id in records:
                raise ValueError(
                    f"duplicate vector-index chunk id: {record.chunk_id}"
                )
            records[record.chunk_id] = record
        return cls(
            collection_name=str(value["collection_name"]),
            embedder_stamp=str(value["embedder_stamp"]),
            embedding_dim=int(value["embedding_dim"]),
            records=records,
        )


class VectorIndexManifestStore:
    """Atomic JSON persistence for regenerable vector-index state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> VectorIndexManifest | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid vector-index manifest payload")
        return VectorIndexManifest.from_dict(payload)

    def save(self, manifest: VectorIndexManifest) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.tmp")
        temp_path.write_text(
            json.dumps(
                manifest.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.path)


@dataclass(frozen=True)
class VectorIndexSyncReport:
    embedded: int
    metadata_updated: int
    removed: int
    unchanged: int
    rebuilt: bool

    @property
    def changed(self) -> int:
        return self.embedded + self.metadata_updated + self.removed


class SelectiveVectorIndexSynchronizer:
    """Synchronize only stale chunks into a Chroma-compatible collection."""

    def __init__(
        self,
        *,
        collection: Any,
        collection_name: str,
        embedder: BaseEmbedder,
        manifest_store: VectorIndexManifestStore,
        reset_collection: Callable[[], Any] | None = None,
        batch_size: int = VECTOR_INDEX_BATCH_SIZE,
    ):
        if batch_size <= 0:
            raise ValueError("vector index batch_size must be positive")
        self.collection = collection
        self.collection_name = collection_name
        self.embedder = embedder
        self.manifest_store = manifest_store
        self.reset_collection = reset_collection
        self.batch_size = batch_size

    def sync(self, chunks: Iterable[Chunk]) -> VectorIndexSyncReport:
        chunk_list = list(chunks)
        current_stamp = embedder_stamp(self.embedder)
        current_dim = int(self.embedder.dim)
        current: dict[str, tuple[Chunk, dict[str, str], VectorIndexRecord]] = {}

        for index, chunk in enumerate(chunk_list):
            chunk_id = chunk_vector_id(chunk)
            if chunk_id in current:
                raise ValueError(
                    "duplicate deterministic vector chunk id: "
                    f"{chunk_id}"
                )
            metadata = chunk_metadata(chunk, index)
            current[chunk_id] = (
                chunk,
                metadata,
                VectorIndexRecord(
                    chunk_id=chunk_id,
                    content_hash=chunk_content_hash(chunk),
                    metadata_hash=_sha256_json(metadata),
                ),
            )

        rebuilt = False
        try:
            manifest = self.manifest_store.load()
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            manifest = None
            rebuilt = True

        actual_ids = self._actual_ids()
        hard_rebuild = False
        if manifest is None:
            previous_records: dict[str, VectorIndexRecord] = {}
            if actual_ids:
                rebuilt = True
                hard_rebuild = True
        elif (
            manifest.collection_name != self.collection_name
            or manifest.embedding_dim != current_dim
        ):
            previous_records = {}
            rebuilt = True
            hard_rebuild = True
        elif manifest.embedder_stamp != current_stamp:
            previous_records = {}
            rebuilt = True
        else:
            previous_records = manifest.records

        target_ids = set(current)
        stale_ids = (set(previous_records) | actual_ids) - target_ids

        if hard_rebuild:
            if self.reset_collection is None:
                raise RuntimeError(
                    "vector collection reset is required for hard rebuild"
                )
            self.collection = self.reset_collection()
            actual_ids = set()
            stale_ids = set()
            previous_records = {}

        if rebuilt:
            # Target IDs can be overwritten safely by upsert. Only records
            # outside the target set must be removed from the collection.
            stale_ids = actual_ids - target_ids
            previous_records = {}

        embed_ids: list[str] = []
        metadata_ids: list[str] = []
        unchanged = 0

        for chunk_id in sorted(current):
            _, _, record = current[chunk_id]
            previous = previous_records.get(chunk_id)
            if previous is None or chunk_id not in actual_ids:
                embed_ids.append(chunk_id)
            elif previous.content_hash != record.content_hash:
                embed_ids.append(chunk_id)
            elif previous.metadata_hash != record.metadata_hash:
                metadata_ids.append(chunk_id)
            else:
                unchanged += 1

        # Persist the manifest only after every collection mutation succeeds.
        # A partial mutation therefore remains recoverable on the next run:
        # the previous manifest will cause the affected operations to replay.
        for start in range(0, len(embed_ids), self.batch_size):
            batch_ids = embed_ids[start:start + self.batch_size]
            texts = [current[chunk_id][0].text for chunk_id in batch_ids]
            embedded = self.embedder.embed_texts(texts)
            embedded_array = np.asarray(embedded)
            if embedded_array.ndim != 2:
                raise ValueError("embedder must return a 2D embedding array")
            if embedded_array.shape[0] != len(batch_ids):
                raise ValueError("embedding count does not match changed chunks")
            if embedded_array.shape[1] != int(self.embedder.dim):
                raise ValueError("embedding dimension mismatch")
            embeddings = [
                [float(value) for value in row]
                for row in embedded_array
            ]
            self.collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=[current[chunk_id][0].text for chunk_id in batch_ids],
                metadatas=[current[chunk_id][1] for chunk_id in batch_ids],
            )

        if metadata_ids:
            self.collection.update(
                ids=metadata_ids,
                metadatas=[current[chunk_id][1] for chunk_id in metadata_ids],
            )

        if stale_ids:
            self.collection.delete(ids=sorted(stale_ids))

        next_manifest = VectorIndexManifest(
            collection_name=self.collection_name,
            embedder_stamp=current_stamp,
            embedding_dim=current_dim,
            records={
                chunk_id: current[chunk_id][2]
                for chunk_id in sorted(current)
            },
        )
        self.manifest_store.save(next_manifest)

        return VectorIndexSyncReport(
            embedded=len(embed_ids),
            metadata_updated=len(metadata_ids),
            removed=len(stale_ids),
            unchanged=unchanged,
            rebuilt=rebuilt,
        )

    def _actual_ids(self) -> set[str]:
        data = self.collection.get()
        if not data:
            return set()
        return {str(value) for value in data.get("ids", [])}
