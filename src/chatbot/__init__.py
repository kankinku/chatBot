"""Top-level chatbot package with lazy compatibility exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    # Core
    "ChatbotException": ("chatbot.core", "ChatbotException"),
    "ConfigurationError": ("chatbot.core", "ConfigurationError"),
    "get_logger": ("chatbot.core", "get_logger"),
    "Chunk": ("chatbot.core", "Chunk"),
    "RetrievedSpan": ("chatbot.core", "RetrievedSpan"),
    "Answer": ("chatbot.core", "Answer"),
    # Preprocessing
    "TextCleaner": ("chatbot.preprocessing", "TextCleaner"),
    "OCRCorrector": ("chatbot.preprocessing", "OCRCorrector"),
    "Normalizer": ("chatbot.preprocessing", "Normalizer"),
    "PDFExtractor": ("chatbot.preprocessing", "PDFExtractor"),
    # Chunking
    "BaseChunker": ("chatbot.chunking", "BaseChunker"),
    "SlidingWindowChunker": ("chatbot.chunking", "SlidingWindowChunker"),
    "NumericChunker": ("chatbot.chunking", "NumericChunker"),
    # Embedding
    "BaseEmbedder": ("chatbot.embedding", "BaseEmbedder"),
    "SBERTEmbedder": ("chatbot.embedding", "SBERTEmbedder"),
    "create_embedder": ("chatbot.embedding", "create_embedder"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
