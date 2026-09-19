"""Dependency-aware blast-radius traversal for the generated workspace."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .models import MetaRelation, WorkspaceGraph


_FORWARD_IMPACT = {
    MetaRelation.PRODUCES,
    MetaRelation.CONFIGURES,
    MetaRelation.VALIDATES,
}
_REVERSE_IMPACT = {
    MetaRelation.DEPENDS_ON,
    MetaRelation.DERIVED_FROM,
    MetaRelation.SUPPORTED_BY,
    MetaRelation.CONTRADICTED_BY,
}


@dataclass(frozen=True)
class ImpactHit:
    node_id: str
    from_id: str
    relation: str
    depth: int


def blast_radius(graph: WorkspaceGraph, seeds: list[str]) -> list[ImpactHit]:
    """Return deterministic downstream impact from changed source/dependency nodes.

    Structural part_of and semantic domain_relation edges are deliberately
    excluded because they express topology or meaning rather than recomputation.
    """

    outgoing: dict[str, list] = {}
    incoming: dict[str, list] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.source, []).append(edge)
        incoming.setdefault(edge.target, []).append(edge)

    queue = deque((seed, 0) for seed in sorted(set(seeds)))
    seen = set(seeds)
    hits: list[ImpactHit] = []

    while queue:
        current, depth = queue.popleft()
        candidates: list[tuple[str, str, str]] = []

        for edge in outgoing.get(current, []):
            if edge.relation in _FORWARD_IMPACT:
                candidates.append((edge.target, current, edge.relation.value))

        for edge in incoming.get(current, []):
            if edge.relation in _REVERSE_IMPACT:
                candidates.append((edge.source, current, edge.relation.value))

        for target, from_id, relation in sorted(candidates):
            if target in seen:
                continue
            seen.add(target)
            hits.append(
                ImpactHit(
                    node_id=target,
                    from_id=from_id,
                    relation=relation,
                    depth=depth + 1,
                )
            )
            queue.append((target, depth + 1))

    return hits
