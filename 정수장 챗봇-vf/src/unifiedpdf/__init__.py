"""
UnifiedPDFPipeline package.

This package provides a unified, Windows-friendly RAG pipeline for PDF corpora
with hybrid retrieval (BM25 + vector), RRF merging, deduplication, calibrated
filtering, dynamic context building, guardrails, and a simple LLM adapter.
"""

__all__ = [
    "config",
    "types",
    "analyzer",
    "retriever",
    "merger",
    "filtering",
    "context",
    "guardrail",
    "llm",
    "facade",
    "metrics",
    "timeouts",
    "vector_store",
    "reranker",
    "measurements",
]
