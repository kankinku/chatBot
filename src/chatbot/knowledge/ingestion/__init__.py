"""Selective, source-hash-based Knowledge Core ingestion."""

from .file_inventory import (
    DEFAULT_FILE_PATTERNS,
    ExtractedTextCache,
    FileInventoryRecord,
    FileInventoryState,
    FileInventoryStateStore,
    FileTextExtractor,
    extractor_stamp,
    hash_source_file,
    scan_source_files,
)
from .file_manager import (
    FileSyncResult,
    SelectiveFileIngestionManager,
    SelectiveFileIngestionReport,
)
from .fingerprint import processor_stamp
from .manager import SelectiveIngestionManager
from .models import (
    IngestionAction,
    IngestionRecord,
    IngestionState,
    RelationReconcileResult,
    RelationSpec,
    SelectiveIngestionReport,
    SourceDocument,
    SourceSyncResult,
)
from .reconciler import (
    EvidenceRelationReconciler,
    relation_specs_from_graph,
)
from .state import IngestionStateStore

__all__ = [
    "DEFAULT_FILE_PATTERNS",
    "EvidenceRelationReconciler",
    "ExtractedTextCache",
    "FileInventoryRecord",
    "FileInventoryState",
    "FileInventoryStateStore",
    "FileSyncResult",
    "FileTextExtractor",
    "IngestionAction",
    "IngestionRecord",
    "IngestionState",
    "IngestionStateStore",
    "RelationReconcileResult",
    "RelationSpec",
    "SelectiveFileIngestionManager",
    "SelectiveFileIngestionReport",
    "SelectiveIngestionManager",
    "SelectiveIngestionReport",
    "SourceDocument",
    "SourceSyncResult",
    "extractor_stamp",
    "hash_source_file",
    "processor_stamp",
    "relation_specs_from_graph",
    "scan_source_files",
]
