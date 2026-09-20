"""Persistence for selective ingestion derived state."""

from __future__ import annotations

import json
from pathlib import Path

from .models import IngestionState


class IngestionStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> IngestionState:
        if not self.path.exists():
            return IngestionState()
        value = json.loads(self.path.read_text(encoding="utf-8"))
        return IngestionState.from_dict(value)

    def save(self, state: IngestionState) -> None:
        state.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.tmp")
        temp.write_text(
            json.dumps(
                state.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temp.replace(self.path)
