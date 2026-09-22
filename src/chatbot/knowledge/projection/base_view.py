"""Resolve current or historical canonical Knowledge Core state for projection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from chatbot.knowledge.ingestion.models import IngestionState
from chatbot.knowledge.ingestion.state import IngestionStateStore
from chatbot.knowledge.replay import KnowledgeReplayService
from chatbot.knowledge.replay.models import state_digest
from chatbot.knowledge.workspace.models import WorkspaceGraph

from .models import ProjectionBase


@dataclass(frozen=True)
class ProjectionBaseState:
    metadata: ProjectionBase
    _state: IngestionState

    def __post_init__(self) -> None:
        # Freeze a defensive canonical copy so the state projected by the
        # engine cannot drift away from the identity recorded in metadata.
        state_copy = IngestionState.from_dict(self._state.to_dict())
        actual_digest = state_digest(state_copy)
        if self.metadata.state_digest != actual_digest:
            raise ValueError("projection base state digest mismatch")
        object.__setattr__(self, "_state", state_copy)

    @property
    def state(self) -> IngestionState:
        return IngestionState.from_dict(self._state.to_dict())

    def merged_graph(self) -> WorkspaceGraph:
        return self._state.ledger.merged_graph()


class ProjectionBaseResolver:
    def __init__(
        self,
        *,
        replay_service: KnowledgeReplayService,
        current_state_store: IngestionStateStore,
    ):
        self.replay_service = replay_service
        self.current_state_store = current_state_store

    def snapshot(self, snapshot_id: str) -> ProjectionBaseState:
        replay = self.replay_service.state_by_snapshot(snapshot_id)
        return ProjectionBaseState(
            metadata=ProjectionBase(
                state_digest=replay.state_digest,
                snapshot_id=replay.snapshot_id,
                committed_at=replay.committed_at,
                origin="replay_snapshot",
            ),
            _state=replay.state,
        )

    def as_of(self, at: datetime) -> ProjectionBaseState:
        replay = self.replay_service.state_as_of(at)
        if replay is None:
            raise LookupError("no replay snapshot available at requested as_of")
        return ProjectionBaseState(
            metadata=ProjectionBase(
                state_digest=replay.state_digest,
                snapshot_id=replay.snapshot_id,
                committed_at=replay.committed_at,
                origin="replay_snapshot",
            ),
            _state=replay.state,
        )

    def current(self) -> ProjectionBaseState:
        current = self.current_state_store.load()
        digest = state_digest(current)
        latest = self.replay_service.store.latest()
        if latest is not None and latest.state_digest == digest:
            return ProjectionBaseState(
                metadata=ProjectionBase(
                    state_digest=digest,
                    snapshot_id=latest.snapshot_id,
                    committed_at=latest.committed_at,
                    origin="replay_snapshot",
                ),
                _state=current,
            )
        return ProjectionBaseState(
            metadata=ProjectionBase(
                state_digest=digest,
                origin="current_state",
            ),
            _state=current,
        )
