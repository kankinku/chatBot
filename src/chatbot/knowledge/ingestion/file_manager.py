"""Selective raw-file ingestion using byte hashes before text extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .file_inventory import (
    DEFAULT_FILE_PATTERNS,
    ExtractedTextCache,
    FileInventoryRecord,
    FileInventoryStateStore,
    FileTextExtractor,
    doc_id_for,
    extractor_stamp,
    hash_source_file,
    scan_source_files,
    source_uri_for,
)
from .manager import SelectiveIngestionManager
from .models import SelectiveIngestionReport, SourceDocument


@dataclass(frozen=True)
class FileSyncResult:
    source_uri: str
    relative_path: str
    action: str
    byte_hash: str | None = None
    reason: str | None = None


@dataclass
class SelectiveFileIngestionReport:
    file_results: list[FileSyncResult] = field(default_factory=list)
    ingestion: SelectiveIngestionReport | None = None

    @property
    def extracted(self) -> int:
        return sum(item.action == "extracted" for item in self.file_results)

    @property
    def reused(self) -> int:
        return sum(item.action == "reused" for item in self.file_results)

    @property
    def removed(self) -> int:
        return sum(item.action == "removed" for item in self.file_results)

    @property
    def failed(self) -> int:
        file_failures = sum(
            item.action == "failed"
            for item in self.file_results
        )
        ingestion_failures = self.ingestion.failed if self.ingestion else 0
        return file_failures + ingestion_failures

    def to_dict(self) -> dict:
        return {
            "files": {
                "extracted": self.extracted,
                "reused": self.reused,
                "removed": self.removed,
                "failed": sum(
                    item.action == "failed"
                    for item in self.file_results
                ),
                "results": [
                    {
                        "source_uri": item.source_uri,
                        "relative_path": item.relative_path,
                        "action": item.action,
                        "byte_hash": item.byte_hash,
                        "reason": item.reason,
                    }
                    for item in self.file_results
                ],
            },
            "ingestion": (
                self.ingestion.to_dict()
                if self.ingestion is not None
                else None
            ),
        }


class SelectiveFileIngestionManager:
    """Hash raw files first and only extract changed/stale files."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        ingestion_manager: SelectiveIngestionManager,
        inventory_store: FileInventoryStateStore | None = None,
        text_cache: ExtractedTextCache | None = None,
        extractor: FileTextExtractor | None = None,
        stamp_provider: Callable[[str | Path], str] = extractor_stamp,
    ):
        self.project_root = Path(project_root).resolve()
        self.ingestion_manager = ingestion_manager
        workspace = self.project_root / "knowledge-workspace"
        self.inventory_store = inventory_store or FileInventoryStateStore(
            workspace / "file-inventory.json"
        )
        self.text_cache = text_cache or ExtractedTextCache(
            workspace / "extracted-text"
        )
        self.extractor = extractor or FileTextExtractor()
        self.stamp_provider = stamp_provider

    def sync(
        self,
        source_root: str | Path,
        *,
        patterns: Iterable[str] = DEFAULT_FILE_PATTERNS,
        prune_missing: bool = True,
    ) -> SelectiveFileIngestionReport:
        root = Path(source_root).resolve()
        state = self.inventory_store.load(root)
        previous_records = dict(state.records)
        stamp = self.stamp_provider(self.project_root)
        files = scan_source_files(root, patterns)
        report = SelectiveFileIngestionReport()
        documents: list[SourceDocument] = []
        current_uris: set[str] = set()
        unsafe_to_prune = False

        for relative_path, path in files:
            source_uri = source_uri_for(relative_path)
            current_uris.add(source_uri)
            byte_hash = hash_source_file(path)
            previous = state.records.get(source_uri)
            text = None
            action = "extracted"
            reason = "new_source"

            cache_expected = (
                previous is not None
                and previous.byte_hash == byte_hash
                and previous.extractor_stamp == stamp
            )
            if cache_expected:
                text = self.text_cache.read(previous)
                if text is not None:
                    action = "reused"
                    reason = "source_bytes_and_extractor_unchanged"

            if text is None:
                try:
                    text = self.extractor.extract(path)
                except Exception as exc:
                    fallback = (
                        self.text_cache.read(previous)
                        if previous is not None
                        else None
                    )
                    if fallback is not None:
                        documents.append(
                            SourceDocument(
                                doc_id=doc_id_for(previous.relative_path),
                                source_uri=previous.source_uri,
                                text=fallback,
                            )
                        )
                    elif previous is not None:
                        unsafe_to_prune = True
                    report.file_results.append(
                        FileSyncResult(
                            source_uri=source_uri,
                            relative_path=relative_path,
                            action="failed",
                            byte_hash=byte_hash,
                            reason=str(exc),
                        )
                    )
                    continue

                reason = (
                    "cache_missing_or_corrupt"
                    if cache_expected
                    else "source_bytes_changed"
                    if previous is not None
                    and previous.byte_hash != byte_hash
                    else "extractor_changed"
                    if previous is not None
                    else "new_source"
                )
                cache_key = self.text_cache.cache_key(
                    byte_hash,
                    stamp,
                )
                text_hash = self.text_cache.write(cache_key, text)
                state.records[source_uri] = FileInventoryRecord(
                    relative_path=relative_path,
                    source_uri=source_uri,
                    byte_hash=byte_hash,
                    extractor_stamp=stamp,
                    text_hash=text_hash,
                    cache_key=cache_key,
                )
            else:
                state.records[source_uri] = previous

            documents.append(
                SourceDocument(
                    doc_id=doc_id_for(relative_path),
                    source_uri=source_uri,
                    text=text,
                )
            )
            report.file_results.append(
                FileSyncResult(
                    source_uri=source_uri,
                    relative_path=relative_path,
                    action=action,
                    byte_hash=byte_hash,
                    reason=reason,
                )
            )

        if prune_missing:
            for source_uri in sorted(set(state.records) - current_uris):
                record = state.records.pop(source_uri)
                report.file_results.append(
                    FileSyncResult(
                        source_uri=source_uri,
                        relative_path=record.relative_path,
                        action="removed",
                        byte_hash=record.byte_hash,
                        reason="source_file_missing",
                    )
                )

        effective_prune = prune_missing and not unsafe_to_prune
        try:
            report.ingestion = self.ingestion_manager.sync(
                documents,
                prune_missing=effective_prune,
                prune_scope=(set(previous_records) | current_uris),
            )
        except Exception:
            state.records = previous_records
            raise

        self.inventory_store.save(state)
        self.text_cache.prune(
            record.cache_key
            for record in state.records.values()
        )
        return report
