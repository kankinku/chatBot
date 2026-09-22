"""High-level current/historical scenario projection service."""

from __future__ import annotations

from datetime import datetime

from .base_view import ProjectionBaseResolver
from .engine import ScenarioProjectionEngine
from .models import RegimeSpec, ScenarioProjection, ScenarioSpec
from .store import ProjectionStore


class ProjectionService:
    def __init__(
        self,
        *,
        base_resolver: ProjectionBaseResolver,
        engine: ScenarioProjectionEngine | None = None,
        store: ProjectionStore | None = None,
    ):
        self.base_resolver = base_resolver
        self.engine = engine or ScenarioProjectionEngine()
        self.store = store

    def project_snapshot(
        self,
        snapshot_id: str,
        scenario: ScenarioSpec,
        regime: RegimeSpec | None = None,
        *,
        persist: bool = False,
    ) -> ScenarioProjection:
        projection = self.engine.project(
            self.base_resolver.snapshot(snapshot_id),
            scenario,
            regime,
        )
        self._maybe_persist(projection, persist)
        return projection

    def project_as_of(
        self,
        at: datetime,
        scenario: ScenarioSpec,
        regime: RegimeSpec | None = None,
        *,
        persist: bool = False,
    ) -> ScenarioProjection:
        projection = self.engine.project(
            self.base_resolver.as_of(at),
            scenario,
            regime,
        )
        self._maybe_persist(projection, persist)
        return projection

    def project_current(
        self,
        scenario: ScenarioSpec,
        regime: RegimeSpec | None = None,
        *,
        persist: bool = False,
    ) -> ScenarioProjection:
        projection = self.engine.project(
            self.base_resolver.current(),
            scenario,
            regime,
        )
        self._maybe_persist(projection, persist)
        return projection

    def load(self, projection_id: str):
        if self.store is None:
            raise RuntimeError("projection store is not configured")
        return self.store.load(projection_id)

    def verify(self, projection_id: str):
        if self.store is None:
            raise RuntimeError("projection store is not configured")
        return self.store.verify(projection_id)

    def _maybe_persist(
        self,
        projection: ScenarioProjection,
        persist: bool,
    ) -> None:
        if not persist:
            return
        if self.store is None:
            raise RuntimeError("projection store is not configured")
        self.store.save(projection)
