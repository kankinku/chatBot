"""Parse canonical ontology/config sources into deterministic workspace records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .hashing import hash_value
from .models import (
    Confidence,
    MetaRelation,
    NodeKind,
    SourceRef,
    WorkspaceEdge,
    WorkspaceNode,
)


def _source_node(relative: str, source: SourceRef) -> WorkspaceNode:
    return WorkspaceNode(
        id=f"source:{relative}",
        label=relative,
        kind=NodeKind.SOURCE,
        content_hash=source.hash,
        sources=[source],
        properties={"path": relative},
    )


def _edge(
    source_id: str,
    target_id: str,
    relation: MetaRelation,
    source: SourceRef,
    *,
    semantic_type: str | None = None,
    confidence: Confidence = Confidence.DECLARED,
    properties: dict[str, Any] | None = None,
) -> WorkspaceEdge:
    props = properties or {}
    payload = {
        "source": source_id,
        "target": target_id,
        "relation": relation.value,
        "semantic_type": semantic_type,
        "properties": props,
    }
    return WorkspaceEdge(
        source=source_id,
        target=target_id,
        relation=relation,
        confidence=confidence,
        content_hash=hash_value(payload),
        sources=[source],
        semantic_type=semantic_type,
        properties=props,
    )


def _config_node(
    relative: str,
    key: str,
    value: Any,
    source: SourceRef,
) -> WorkspaceNode:
    return WorkspaceNode(
        id=f"config:{relative}#{key}",
        label=key,
        kind=NodeKind.CONFIG,
        content_hash=hash_value(value),
        sources=[source],
        properties={
            "path": relative,
            "key": key,
            "value": value,
        },
    )


def _pick_id(item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _entity_ref(entity_id: str, source: SourceRef) -> WorkspaceNode:
    return WorkspaceNode(
        id=f"entity:{entity_id}",
        label=entity_id,
        kind=NodeKind.ENTITY_REF,
        content_hash=hash_value({"entity_ref": entity_id}),
        sources=[source],
        properties={
            "entity_id": entity_id,
            "placeholder": True,
        },
    )


def _parse_yaml(
    relative: str,
    data: dict[str, Any],
    source: SourceRef,
) -> tuple[list[WorkspaceNode], list[WorkspaceEdge]]:
    source_id = f"source:{relative}"
    nodes: list[WorkspaceNode] = []
    edges: list[WorkspaceEdge] = []

    if relative.endswith("entity_types.yaml"):
        for name, value in sorted((data.get("entity_types") or {}).items()):
            node_id = f"entity_type:{name}"
            nodes.append(
                WorkspaceNode(
                    id=node_id,
                    label=name,
                    kind=NodeKind.ENTITY_TYPE,
                    content_hash=hash_value(value),
                    sources=[source],
                    properties=value or {},
                )
            )
            edges.append(
                _edge(source_id, node_id, MetaRelation.PRODUCES, source)
            )
            edges.append(
                _edge(
                    node_id,
                    "system:knowledge_core",
                    MetaRelation.CONFIGURES,
                    source,
                )
            )
        return nodes, edges

    if relative.endswith("relation_types.yaml"):
        for name, value in sorted((data.get("relation_types") or {}).items()):
            node_id = f"relation_type:{name}"
            nodes.append(
                WorkspaceNode(
                    id=node_id,
                    label=name,
                    kind=NodeKind.RELATION_TYPE,
                    content_hash=hash_value(value),
                    sources=[source],
                    properties=value or {},
                )
            )
            edges.append(
                _edge(source_id, node_id, MetaRelation.PRODUCES, source)
            )
            edges.append(
                _edge(
                    node_id,
                    "system:knowledge_core",
                    MetaRelation.CONFIGURES,
                    source,
                )
            )
        return nodes, edges

    for key, value in sorted(data.items()):
        node = _config_node(relative, key, value, source)
        nodes.append(node)
        edges.append(
            _edge(source_id, node.id, MetaRelation.PRODUCES, source)
        )
        relation = (
            MetaRelation.VALIDATES
            if relative.endswith("validation_schema.yaml")
            else MetaRelation.CONFIGURES
        )
        edges.append(
            _edge(
                node.id,
                "system:knowledge_core",
                relation,
                source,
            )
        )

    return nodes, edges


def _parse_entities(
    relative: str,
    items: list[Any],
    source: SourceRef,
) -> tuple[list[WorkspaceNode], list[WorkspaceEdge]]:
    source_id = f"source:{relative}"
    nodes: list[WorkspaceNode] = []
    edges: list[WorkspaceEdge] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        entity_id = _pick_id(
            item,
            ("entity_id", "canonical_id", "id", "name", "canonical_name"),
        ) or hash_value(item)[:16]
        label = str(
            item.get("canonical_name")
            or item.get("name")
            or item.get("label")
            or entity_id
        )
        node = WorkspaceNode(
            id=f"entity:{entity_id}",
            label=label,
            kind=NodeKind.ENTITY,
            content_hash=hash_value(item),
            sources=[source],
            properties=dict(item),
        )
        nodes.append(node)
        edges.append(
            _edge(source_id, node.id, MetaRelation.PRODUCES, source)
        )

    return nodes, edges


def _parse_relations(
    relative: str,
    items: list[Any],
    source: SourceRef,
) -> tuple[list[WorkspaceNode], list[WorkspaceEdge]]:
    source_id = f"source:{relative}"
    nodes: list[WorkspaceNode] = []
    edges: list[WorkspaceEdge] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        head = _pick_id(item, ("src_id", "head_id", "source_id", "head"))
        tail = _pick_id(item, ("dst_id", "tail_id", "target_id", "tail"))
        semantic = _pick_id(
            item,
            ("rel_type", "relation_type", "type", "relation"),
        )
        if not head or not tail or not semantic:
            continue

        assertion_hash = hash_value(item)
        assertion_id = f"assertion:{assertion_hash[:20]}"
        nodes.append(
            WorkspaceNode(
                id=assertion_id,
                label=f"{head} {semantic} {tail}",
                kind=NodeKind.ASSERTION,
                content_hash=assertion_hash,
                sources=[source],
                properties=dict(item),
            )
        )
        nodes.extend(
            [
                _entity_ref(head, source),
                _entity_ref(tail, source),
            ]
        )

        edges.extend(
            [
                _edge(
                    source_id,
                    assertion_id,
                    MetaRelation.PRODUCES,
                    source,
                ),
                _edge(
                    assertion_id,
                    f"entity:{head}",
                    MetaRelation.DEPENDS_ON,
                    source,
                ),
                _edge(
                    assertion_id,
                    f"entity:{tail}",
                    MetaRelation.DEPENDS_ON,
                    source,
                ),
                _edge(
                    f"entity:{head}",
                    f"entity:{tail}",
                    MetaRelation.DOMAIN_RELATION,
                    source,
                    semantic_type=semantic,
                    confidence=Confidence.VALIDATED,
                ),
            ]
        )

    return nodes, edges


def _parse_documents(
    relative: str,
    items: list[Any],
    source: SourceRef,
) -> tuple[list[WorkspaceNode], list[WorkspaceEdge]]:
    source_id = f"source:{relative}"
    nodes: list[WorkspaceNode] = []
    edges: list[WorkspaceEdge] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        doc_id = _pick_id(
            item,
            ("doc_id", "document_id", "id", "title"),
        ) or hash_value(item)[:16]
        label = str(item.get("title") or item.get("name") or doc_id)
        node = WorkspaceNode(
            id=f"document:{doc_id}",
            label=label,
            kind=NodeKind.DOCUMENT,
            content_hash=hash_value(item),
            sources=[source],
            properties=dict(item),
        )
        nodes.append(node)
        edges.append(
            _edge(source_id, node.id, MetaRelation.PRODUCES, source)
        )

    return nodes, edges


def _parse_json(
    relative: str,
    data: Any,
    source: SourceRef,
) -> tuple[list[WorkspaceNode], list[WorkspaceEdge]]:
    items = data if isinstance(data, list) else [data]

    if relative.endswith("/domain/entities.json"):
        return _parse_entities(relative, items, source)
    if relative.endswith("/domain/relations.json"):
        return _parse_relations(relative, items, source)
    if relative.endswith("/samples/sample_documents.json"):
        return _parse_documents(relative, items, source)

    source_id = f"source:{relative}"
    nodes: list[WorkspaceNode] = []
    edges: list[WorkspaceEdge] = []
    for ordinal, item in enumerate(items):
        node = WorkspaceNode(
            id=f"config_data:{relative}#{ordinal}",
            label=f"{relative}#{ordinal}",
            kind=NodeKind.CONFIG,
            content_hash=hash_value(item),
            sources=[source],
            properties={"value": item},
        )
        nodes.append(node)
        edges.append(
            _edge(source_id, node.id, MetaRelation.PRODUCES, source)
        )
    return nodes, edges


def parse_source(
    root: str | Path,
    path: str | Path,
    source_hash: str,
) -> tuple[list[WorkspaceNode], list[WorkspaceEdge]]:
    root_path = Path(root).resolve()
    source_path = Path(path).resolve()
    relative = source_path.relative_to(root_path).as_posix()
    source = SourceRef(path=relative, hash=source_hash)

    nodes = [_source_node(relative, source)]
    edges: list[WorkspaceEdge] = []

    text = source_path.read_text(encoding="utf-8-sig")
    if source_path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text) or {}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        child_nodes, child_edges = _parse_yaml(relative, payload, source)
    else:
        payload = json.loads(text or "null")
        child_nodes, child_edges = _parse_json(relative, payload, source)

    nodes.extend(child_nodes)
    edges.extend(child_edges)
    return nodes, edges
