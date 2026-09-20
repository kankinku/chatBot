"""Deterministic, source-diversity-aware evidence score aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from chatbot.knowledge.workspace.models import WorkspaceNode


SCORE_VERSION = "evidence-score-v1"


@dataclass(frozen=True)
class EvidenceScorePolicy:
    baseline: float = 0.5
    support_gain: float = 0.4
    conflict_penalty: float = 0.45
    minimum: float = 0.1
    maximum: float = 0.95
    fallback_quality: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.baseline <= 1.0:
            raise ValueError("baseline must be within 0..1")
        if self.support_gain < 0.0:
            raise ValueError("support_gain must be non-negative")
        if self.conflict_penalty < 0.0:
            raise ValueError("conflict_penalty must be non-negative")
        if not 0.0 <= self.minimum <= self.maximum <= 1.0:
            raise ValueError("score bounds must satisfy 0 <= minimum <= maximum <= 1")
        if not 0.0 <= self.fallback_quality <= 1.0:
            raise ValueError("fallback_quality must be within 0..1")


@dataclass(frozen=True)
class EvidenceScoreSummary:
    score: float
    support_score: float
    conflict_score: float
    support_strength: float
    conflict_strength: float
    support_source_count: int
    conflict_source_count: int
    support_assertion_count: int
    conflict_assertion_count: int
    version: str = SCORE_VERSION


class EvidenceScoreAggregator:
    """Aggregate assertion quality without treating repetition as independence."""

    def __init__(self, policy: EvidenceScorePolicy | None = None):
        self.policy = policy or EvidenceScorePolicy()

    def score(
        self,
        support_assertions: list[WorkspaceNode],
        conflict_assertions: list[WorkspaceNode],
    ) -> EvidenceScoreSummary:
        support_by_source = self._quality_by_source(support_assertions)
        conflict_by_source = self._quality_by_source(conflict_assertions)

        support_strength = sum(support_by_source.values())
        conflict_strength = sum(conflict_by_source.values())

        support_score = self._saturate(support_strength)
        conflict_score = self._saturate(conflict_strength)

        raw = (
            self.policy.baseline
            + self.policy.support_gain * support_score
            - self.policy.conflict_penalty * conflict_score
        )
        score = max(
            self.policy.minimum,
            min(self.policy.maximum, raw),
        )

        return EvidenceScoreSummary(
            score=round(score, 12),
            support_score=round(support_score, 12),
            conflict_score=round(conflict_score, 12),
            support_strength=round(support_strength, 12),
            conflict_strength=round(conflict_strength, 12),
            support_source_count=len(support_by_source),
            conflict_source_count=len(conflict_by_source),
            support_assertion_count=len(support_assertions),
            conflict_assertion_count=len(conflict_assertions),
        )

    def assertion_quality(self, assertion: WorkspaceNode) -> float:
        props = assertion.properties
        combined = props.get("combined_conf")
        if isinstance(combined, Real):
            return self._clamp(float(combined))

        components = []
        for key in ("semantic_conf", "sign_score", "student_conf"):
            value = props.get(key)
            if isinstance(value, Real):
                components.append(self._clamp(float(value)))

        if components:
            return sum(components) / len(components)
        return self.policy.fallback_quality

    def _quality_by_source(
        self,
        assertions: list[WorkspaceNode],
    ) -> dict[str, float]:
        by_source: dict[str, float] = {}
        for assertion in assertions:
            quality = self.assertion_quality(assertion)
            source_paths = {
                source.path
                for source in assertion.sources
                if source.path
            }
            if not source_paths:
                source_paths = {f"assertion:{assertion.id}"}

            for source_path in source_paths:
                current = by_source.get(source_path)
                if current is None or quality > current:
                    by_source[source_path] = quality
        return by_source

    @staticmethod
    def _saturate(strength: float) -> float:
        if strength <= 0.0:
            return 0.0
        return 1.0 - math.exp(-strength)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
