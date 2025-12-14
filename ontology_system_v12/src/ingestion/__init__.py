"""
Ingestion 패키지
Delta Ingestion + Append-only 저장소 관리
"""
from .source_registry import SourceRegistry
from .fetch_state_store import FetchStateStore
from .delta_fetcher import DeltaFetcher
from .normalizer import Normalizer
from .idempotency_guard import IdempotencyGuard

__all__ = [
    "SourceRegistry",
    "FetchStateStore",
    "DeltaFetcher",
    "Normalizer",
    "IdempotencyGuard",
]
