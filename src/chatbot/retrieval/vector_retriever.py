"""
Vector Retriever - Chroma DB based semantic retrieval.

The Chroma collection is treated as derived state. Phase 10 keeps it in sync
with the current chunk set using deterministic IDs and selective refreshes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from chatbot.core.exceptions import EmbeddingError
from chatbot.core.logger import get_logger
from chatbot.core.types import Chunk
from chatbot.embedding.base_embedder import BaseEmbedder

from .vector_index_sync import (
    SelectiveVectorIndexSynchronizer,
    VectorIndexManifestStore,
    VectorIndexSyncReport,
    chunk_vector_id,
)


logger = get_logger(__name__)


class VectorRetriever:
    """Chroma-backed vector retrieval with selective derived-index refresh."""

    def __init__(
        self,
        chunks: List[Chunk],
        embedder: BaseEmbedder,
        index_dir: Optional[str] = None,
        collection_name: str = "chatbot_chunks",
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.index_dir = index_dir or "vector_store"
        self.collection_name = collection_name
        self.chunk_index_to_id: List[str] = [
            chunk_vector_id(chunk) for chunk in chunks
        ]
        self.id_to_chunk_index = {
            chunk_id: index
            for index, chunk_id in enumerate(self.chunk_index_to_id)
        }
        if len(self.id_to_chunk_index) != len(self.chunk_index_to_id):
            raise EmbeddingError(
                "Duplicate deterministic chunk identities detected"
            )

        logger.info(
            "VectorRetriever initializing",
            num_chunks=len(chunks),
            collection_name=collection_name,
            embedding_dim=embedder.dim,
        )

        self.sync_report = self._init_chroma()
        logger.info(
            "VectorRetriever initialized",
            embedded=self.sync_report.embedded,
            metadata_updated=self.sync_report.metadata_updated,
            removed=self.sync_report.removed,
            unchanged=self.sync_report.unchanged,
            rebuilt=self.sync_report.rebuilt,
        )

    def _init_chroma(self) -> VectorIndexSyncReport:
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=self.index_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            embedding_function = self._get_embedding_function()
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function,
                )
            except Exception:
                self.collection = self._create_collection(
                    embedding_function,
                )

            manifest_path = (
                Path(self.index_dir)
                / f"{self.collection_name}.manifest.json"
            )
            synchronizer = SelectiveVectorIndexSynchronizer(
                collection=self.collection,
                collection_name=self.collection_name,
                embedder=self.embedder,
                manifest_store=VectorIndexManifestStore(manifest_path),
                reset_collection=lambda: self._reset_collection(
                    embedding_function
                ),
            )
            return synchronizer.sync(self.chunks)

        except ImportError as exc:
            raise EmbeddingError(
                "Chroma DB not available. Please install: pip install chromadb",
                cause=exc,
            ) from exc
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                "Failed to initialize or synchronize Chroma DB",
                cause=exc,
            ) from exc

    def _create_collection(self, embedding_function):
        return self.client.create_collection(
            name=self.collection_name,
            embedding_function=embedding_function,
            metadata={
                "description": "Chatbot chunks collection",
                "derived_index_schema": "selective-v1",
                "embedding_dim": int(self.embedder.dim),
            },
        )

    def _reset_collection(self, embedding_function):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # A missing collection is equivalent to an already-reset cache.
            pass
        self.collection = self._create_collection(embedding_function)
        return self.collection

    def _get_embedding_function(self):
        try:
            from chromadb import EmbeddingFunction

            class CustomEmbeddingFunction(EmbeddingFunction):
                def __init__(self, embedder):
                    super().__init__()
                    self.embedder = embedder

                def __call__(self, input):
                    if not input:
                        return []
                    if isinstance(input, str):
                        input = [input]
                    embeddings = self.embedder.embed_texts(input)
                    return [
                        [float(value) for value in embedding]
                        for embedding in embeddings
                    ]

            return CustomEmbeddingFunction(self.embedder)

        except ImportError:
            def embedding_function(input):
                if not input:
                    return []
                if isinstance(input, str):
                    input = [input]
                embeddings = self.embedder.embed_texts(input)
                return [
                    [float(value) for value in embedding]
                    for embedding in embeddings
                ]

            return embedding_function

    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        if not self.chunks or top_k <= 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self.chunks)),
                include=["metadatas", "distances"],
            )

            output: list[tuple[int, float]] = []
            ids = results.get("ids", [[]]) if results else [[]]
            metadatas = results.get("metadatas", [[]]) if results else [[]]
            distances = results.get("distances", [[]]) if results else [[]]

            result_ids = ids[0] if ids else []
            result_metadatas = metadatas[0] if metadatas else []
            result_distances = distances[0] if distances else []

            for chunk_id, metadata, distance in zip(
                result_ids,
                result_metadatas,
                result_distances,
            ):
                chunk_index = self._find_chunk_index(
                    str(chunk_id),
                    metadata,
                )
                if chunk_index is None:
                    continue
                similarity = max(0.0, 1.0 - (float(distance) / 2.0))
                output.append((chunk_index, similarity))

            logger.debug(
                "Chroma vector search completed",
                results=len(output),
                top_score=output[0][1] if output else 0.0,
            )
            return output

        except Exception as exc:
            logger.error(
                f"Chroma search failed: {exc}",
                exc_info=True,
            )
            return []

    def _find_chunk_index(
        self,
        chroma_id: str,
        metadata: Optional[dict] = None,
    ) -> Optional[int]:
        # Current metadata is refreshed even when only chunk ordering changes.
        if metadata:
            chunk_idx_str = metadata.get("chunk_index")
            if chunk_idx_str is not None:
                try:
                    chunk_index = int(chunk_idx_str)
                    if (
                        0 <= chunk_index < len(self.chunks)
                        and self.chunk_index_to_id[chunk_index] == chroma_id
                    ):
                        return chunk_index
                except (TypeError, ValueError, IndexError):
                    pass

        return self.id_to_chunk_index.get(chroma_id)
