"""Structural invariants for relationship/provenance workspace graphs."""

from __future__ import annotations

from .models import MetaRelation, WorkspaceGraph


class WorkspaceInvariantError(ValueError):
    pass


def validate_graph(graph: WorkspaceGraph) -> None:
    node_ids = [node.id for node in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise WorkspaceInvariantError("workspace graph contains duplicate node ids")

    known = set(node_ids)
    identities = set()

    for node in graph.nodes:
        if not node.id or not node.content_hash:
            raise WorkspaceInvariantError("workspace node identity/hash must be non-empty")

    for edge in graph.edges:
        if edge.identity in identities:
            raise WorkspaceInvariantError(
                f"workspace graph contains duplicate edge: {edge.identity}"
            )
        identities.add(edge.identity)

        if edge.source not in known:
            raise WorkspaceInvariantError(
                f"workspace edge has missing source: {edge.source}"
            )
        if edge.target not in known:
            raise WorkspaceInvariantError(
                f"workspace edge has missing target: {edge.target}"
            )
        if not edge.content_hash:
            raise WorkspaceInvariantError("workspace edge hash must be non-empty")
        if edge.relation == MetaRelation.DOMAIN_RELATION and not edge.semantic_type:
            raise WorkspaceInvariantError(
                "domain_relation edges require semantic_type"
            )
