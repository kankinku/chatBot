"""Selective, source-hash-based Knowledge Core ingestion."""

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
    "EvidenceRelationReconciler",
    "IngestionAction",
    "IngestionRecord",
    "IngestionState",
    "IngestionStateStore",
    "RelationReconcileResult",
    "RelationSpec",
    "SelectiveIngestionManager",
    "SelectiveIngestionReport",
    "SourceDocument",
    "SourceSyncResult",
    "processor_stamp",
    "relation_specs_from_graph",
]
