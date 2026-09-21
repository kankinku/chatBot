"""Read-only access to historical Knowledge Core states."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from chatbot.knowledge.ingestion.models import IngestionState
from chatbot.knowledge.workspace.models import WorkspaceGraph

from .models import KnowledgeSnapshot
from .store import KnowledgeReplayStore


@dataclass(frozen=True)
class ReplayState:
    snapshot_id: str
    sequence: int
    committed_at: datetime
    parent_snapshot_id: str | None
    state_digest: str
    _state: IngestionState

    @classmethod
    def from_snapshot(cls, snapshot: KnowledgeSnapshot) -> "ReplayState":
        return cls(
            snapshot_id=snapshot.snapshot_id,
            sequence=snapshot.sequence,
            committed_at=snapshot.committed_at,
            parent_snapshot_id=snapshot.parent_snapshot_id,
            state_digest=snapshot.state_digest,
            _state=IngestionState.from_dict(snapshot.state.to_dict()),
        )

    @property
    def state(self) -> IngestionState:
        """Return a defensive copy so replay callers cannot mutate history."""
        return IngestionState.from_dict(self._state.to_dict())

    def merged_graph(self) -> WorkspaceGraph:
        return self._state.ledger.merged_graph()

    def sources(self) -> list[str]:
        return sorted(self._state.records)

    def source_digests(self) -> dict[str, str]:
        return {
            source_uri: self._state.records[source_uri].source_hash
            for source_uri in sorted(self._state.records)
        }


class KnowledgeReplayService:
    """Resolve immutable historical states without touching live stores."""

    def __init__(self, store: KnowledgeReplayStore):
        self.store = store

    def state_as_of(self, at: datetime) -> ReplayState | None:
        snapshot = self.store.as_of(at)
        return ReplayState.from_snapshot(snapshot) if snapshot is not None else None

    def state_by_snapshot(self, snapshot_id: str) -> ReplayState:
        return ReplayState.from_snapshot(self.store.get(snapshot_id))
