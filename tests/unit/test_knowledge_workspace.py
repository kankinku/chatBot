"""Relationship/provenance workspace contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chatbot.knowledge.workspace import (
    Confidence,
    MetaRelation,
    NodeKind,
    WorkspaceEdge,
    WorkspaceGraph,
    WorkspaceInvariantError,
    WorkspaceNode,
    blast_radius,
    build_workspace,
    check_workspace,
    load_workspace_graph,
    validate_graph,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def workspace_repo(tmp_path: Path) -> Path:
    _write(
        tmp_path / "config/knowledge_workspace.yaml",
        """workspace:
  source_roots:
    - config/ontology
    - data/ontology
  output_dir: knowledge-workspace
""",
    )
    _write(
        tmp_path / "config/ontology/entity_types.yaml",
        """entity_types:
  Event:
    description: event
  Indicator:
    description: indicator
""",
    )
    _write(
        tmp_path / "config/ontology/relation_types.yaml",
        """relation_types:
  Cause:
    description: causes
    has_polarity: true
""",
    )
    _write(
        tmp_path / "config/ontology/validation_schema.yaml",
        """validation_rules:
  allowed_combinations:
    - head_type: Event
      tail_type: Indicator
      relations: [Cause]
""",
    )
    _write(
        tmp_path / "data/ontology/domain/entities.json",
        json.dumps(
            [
                {
                    "entity_id": "event:deployment",
                    "name": "Deployment",
                    "type": "Event",
                },
                {
                    "entity_id": "indicator:error-rate",
                    "name": "Error rate",
                    "type": "Indicator",
                },
            ],
            ensure_ascii=False,
        ),
    )
    _write(
        tmp_path / "data/ontology/domain/relations.json",
        json.dumps(
            [
                {
                    "src_id": "event:deployment",
                    "rel_type": "Cause",
                    "dst_id": "indicator:error-rate",
                    "sign": "+",
                    "domain_conf": 0.8,
                }
            ],
            ensure_ascii=False,
        ),
    )
    _write(
        tmp_path / "data/ontology/samples/sample_documents.json",
        json.dumps(
            [
                {
                    "doc_id": "doc-1",
                    "title": "Deployment incident",
                    "text": "A deployment increased the error rate.",
                }
            ],
            ensure_ascii=False,
        ),
    )
    return tmp_path


def test_workspace_build_is_deterministic_and_incremental(workspace_repo: Path):
    first = build_workspace(workspace_repo, reuse=False)
    graph_path = workspace_repo / "knowledge-workspace/graph.json"
    index_path = workspace_repo / "knowledge-workspace/INDEX.md"
    cold_graph = graph_path.read_bytes()
    cold_index = index_path.read_bytes()

    assert first.parsed == 7
    assert first.reused == 0
    assert first.files == 7
    assert check_workspace(workspace_repo).clean

    second = build_workspace(workspace_repo)
    assert second.parsed == 0
    assert second.reused == 7
    assert graph_path.read_bytes() == cold_graph
    assert index_path.read_bytes() == cold_index

    entity_types = workspace_repo / "config/ontology/entity_types.yaml"
    stat = entity_types.stat()
    os.utime(
        entity_types,
        (stat.st_atime + 10, stat.st_mtime + 10),
    )
    touched = build_workspace(workspace_repo)
    assert touched.parsed == 0
    assert touched.reused == 7
    assert graph_path.read_bytes() == cold_graph

    entity_types.write_text(
        entity_types.read_text(encoding="utf-8")
        + "  Concept:\n    description: concept\n",
        encoding="utf-8",
    )
    drift = check_workspace(workspace_repo)
    assert drift.changed == ["config/ontology/entity_types.yaml"]
    assert not drift.clean

    incremental = build_workspace(workspace_repo)
    warm_graph = graph_path.read_bytes()
    warm_index = index_path.read_bytes()

    assert incremental.parsed == 1
    assert incremental.reused == 6
    assert check_workspace(workspace_repo).clean

    cold_again = build_workspace(workspace_repo, reuse=False)
    assert cold_again.parsed == 7
    assert cold_again.reused == 0
    assert graph_path.read_bytes() == warm_graph
    assert index_path.read_bytes() == warm_index


def test_workspace_preserves_assertion_provenance_and_typed_relation(
    workspace_repo: Path,
):
    build_workspace(workspace_repo, reuse=False)
    graph = load_workspace_graph(workspace_repo)

    nodes = {node.id: node for node in graph.nodes}
    relation_edges = [
        edge
        for edge in graph.edges
        if edge.relation == MetaRelation.DOMAIN_RELATION
    ]
    assertions = [
        node
        for node in graph.nodes
        if node.kind == NodeKind.ASSERTION
    ]

    assert nodes["entity:event:deployment"].kind == NodeKind.ENTITY
    assert nodes["entity:indicator:error-rate"].kind == NodeKind.ENTITY
    assert len(assertions) == 1
    assert assertions[0].properties["domain_conf"] == 0.8
    assert assertions[0].sources[0].path == "data/ontology/domain/relations.json"

    assert len(relation_edges) == 1
    relation = relation_edges[0]
    assert relation.source == "entity:event:deployment"
    assert relation.target == "entity:indicator:error-rate"
    assert relation.semantic_type == "Cause"
    assert relation.sources[0].path == "data/ontology/domain/relations.json"


def test_workspace_blast_radius_uses_dependency_edges_only(
    workspace_repo: Path,
):
    build_workspace(workspace_repo, reuse=False)
    graph = load_workspace_graph(workspace_repo)

    hits = blast_radius(
        graph,
        ["source:config/ontology/entity_types.yaml"],
    )
    ids = [hit.node_id for hit in hits]

    assert "entity_type:Event" in ids
    assert "entity_type:Indicator" in ids
    assert "system:knowledge_core" in ids
    assert all(hit.relation != MetaRelation.DOMAIN_RELATION.value for hit in hits)


def test_workspace_check_detects_add_remove_and_generator_state(
    workspace_repo: Path,
):
    build_workspace(workspace_repo, reuse=False)

    added = workspace_repo / "data/ontology/extra.json"
    _write(added, json.dumps({"value": 1}))
    report = check_workspace(workspace_repo)
    assert report.added == ["data/ontology/extra.json"]
    assert not report.generator_changed

    added.unlink()
    removed = workspace_repo / "data/ontology/samples/sample_documents.json"
    removed.unlink()
    report = check_workspace(workspace_repo)
    assert report.removed == ["data/ontology/samples/sample_documents.json"]


def test_workspace_graph_invariants_reject_dangling_edges():
    graph = WorkspaceGraph(
        nodes=[
            WorkspaceNode(
                id="source:a",
                label="a",
                kind=NodeKind.SOURCE,
                content_hash="hash",
            )
        ],
        edges=[
            WorkspaceEdge(
                source="source:a",
                target="missing:b",
                relation=MetaRelation.PRODUCES,
                confidence=Confidence.DECLARED,
                content_hash="edge-hash",
            )
        ],
    )

    with pytest.raises(WorkspaceInvariantError):
        validate_graph(graph)


def test_workspace_rejects_output_overlapping_source_root(tmp_path: Path):
    _write(
        tmp_path / "config/knowledge_workspace.yaml",
        """workspace:
  source_roots:
    - data/ontology
  output_dir: data/ontology/generated
""",
    )
    (tmp_path / "data/ontology").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="must not overlap"):
        build_workspace(tmp_path, reuse=False)
