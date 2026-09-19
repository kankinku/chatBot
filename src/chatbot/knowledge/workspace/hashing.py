"""Stable hashing helpers for the generated knowledge workspace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import SourceRef


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_value(value: Any) -> str:
    return hash_bytes(canonical_json(value).encode("utf-8"))


def hash_file(path: str | Path) -> str:
    return hash_bytes(Path(path).read_bytes())


def digest_sources(sources: list[SourceRef]) -> str:
    lines = [
        f"{source.path}:{source.hash}"
        for source in sorted(sources)
    ]
    return hash_bytes("\n".join(lines).encode("utf-8"))
