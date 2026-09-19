"""Build and check the local deterministic knowledge workspace."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .hashing import digest_sources, hash_file, hash_value
from .invariants import validate_graph
from .models import (
    DriftReport,
    NodeKind,
    SCHEMA_VERSION,
    SourceRef,
    WorkspaceEdge,
    WorkspaceGraph,
    WorkspaceNode,
)
from .parsers import parse_source


GENERATOR_VERSION = 1
DEFAULT_OUTPUT_DIR = "knowledge-workspace"
DEFAULT_SOURCE_ROOTS = ("config/ontology", "data/ontology")
WORKSPACE_CONFIG = "config/knowledge_workspace.yaml"
SUPPORTED_SUFFIXES = {".json", ".yaml", ".yml"}


@dataclass(frozen=True)
class BuildResult:
    parsed: int
    reused: int
    files: int
    nodes: int
    edges: int
    repo_digest: str
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "reused": self.reused,
            "files": self.files,
            "nodes": self.nodes,
            "edges": self.edges,
            "repo_digest": self.repo_digest,
            "output_dir": self.output_dir,
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _load_workspace_config(root: Path) -> dict[str, Any]:
    path = root / WORKSPACE_CONFIG
    if not path.exists():
        return {
            "source_roots": list(DEFAULT_SOURCE_ROOTS),
            "output_dir": DEFAULT_OUTPUT_DIR,
        }

    data = yaml.safe_load(_read_text(path)) or {}
    if not isinstance(data, dict):
        raise ValueError("knowledge workspace config must be a mapping")
    workspace = data.get("workspace", data)
    if not isinstance(workspace, dict):
        raise ValueError("workspace section must be a mapping")

    source_roots = workspace.get("source_roots", list(DEFAULT_SOURCE_ROOTS))
    output_dir = workspace.get("output_dir", DEFAULT_OUTPUT_DIR)

    if not isinstance(source_roots, list) or not all(
        isinstance(item, str) and item.strip() for item in source_roots
    ):
        raise ValueError("workspace.source_roots must be a list of paths")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("workspace.output_dir must be a path")

    return {
        "source_roots": source_roots,
        "output_dir": output_dir,
    }


def _inside_root(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository: {path}") from exc
    return resolved


def _resolve_output(
    root: Path,
    config: dict[str, Any],
    output_dir: str | Path | None,
) -> Path:
    output = _inside_root(
        root,
        root / (output_dir or config["output_dir"]),
        "workspace output directory",
    )
    for relative in config["source_roots"]:
        source_root = _inside_root(
            root,
            root / relative,
            "workspace source root",
        )
        if (
            output == source_root
            or source_root in output.parents
            or output in source_root.parents
        ):
            raise ValueError(
                "workspace output directory must not overlap a source root"
            )
    return output


def _discover_sources(root: Path) -> list[Path]:
    config = _load_workspace_config(root)
    paths: set[Path] = set()

    config_path = root / WORKSPACE_CONFIG
    if config_path.is_file():
        paths.add(
            _inside_root(root, config_path, "workspace config")
        )

    for relative in config["source_roots"]:
        source_root = _inside_root(
            root,
            root / relative,
            "workspace source root",
        )

        if not source_root.exists():
            continue
        if source_root.is_file():
            if source_root.suffix.lower() in SUPPORTED_SUFFIXES:
                paths.add(source_root)
            continue

        for path in source_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                paths.add(
                    _inside_root(root, path, "workspace source file")
                )

    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _generator_stamp() -> str:
    base = Path(__file__).resolve().parent
    names = (
        "builder.py",
        "hashing.py",
        "impact.py",
        "invariants.py",
        "models.py",
        "parsers.py",
    )
    lines = [f"{name}:{hash_file(base / name)}" for name in names]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _merge_sources(*groups: list[SourceRef]) -> list[SourceRef]:
    merged = {
        (source.path, source.hash): source
        for group in groups
        for source in group
    }
    return [merged[key] for key in sorted(merged)]


def _merge_nodes(nodes: list[WorkspaceNode]) -> list[WorkspaceNode]:
    by_id: dict[str, WorkspaceNode] = {}

    for node in sorted(nodes, key=lambda item: (item.id, item.kind.value)):
        current = by_id.get(node.id)
        if current is None:
            by_id[node.id] = node
            continue

        if current.kind == NodeKind.ENTITY_REF and node.kind == NodeKind.ENTITY:
            node.sources = _merge_sources(current.sources, node.sources)
            by_id[node.id] = node
            continue
        if current.kind == NodeKind.ENTITY and node.kind == NodeKind.ENTITY_REF:
            current.sources = _merge_sources(current.sources, node.sources)
            continue

        if (
            current.kind != node.kind
            or current.content_hash != node.content_hash
            or current.properties != node.properties
        ):
            raise ValueError(f"conflicting workspace node id: {node.id}")

        current.sources = _merge_sources(current.sources, node.sources)

    return [by_id[key] for key in sorted(by_id)]


def _merge_edges(edges: list[WorkspaceEdge]) -> list[WorkspaceEdge]:
    by_id: dict[tuple[str, str, str, str], WorkspaceEdge] = {}

    for edge in sorted(edges, key=lambda item: item.identity):
        current = by_id.get(edge.identity)
        if current is None:
            by_id[edge.identity] = edge
            continue

        if (
            current.content_hash != edge.content_hash
            or current.properties != edge.properties
            or current.confidence != edge.confidence
        ):
            raise ValueError(f"conflicting workspace edge: {edge.identity}")

        current.sources = _merge_sources(current.sources, edge.sources)

    return [by_id[key] for key in sorted(by_id)]


def _system_node() -> WorkspaceNode:
    props = {
        "role": "canonical knowledge extraction/validation/reasoning core",
    }
    return WorkspaceNode(
        id="system:knowledge_core",
        label="Knowledge Core",
        kind=NodeKind.SYSTEM,
        content_hash=hash_value(props),
        properties=props,
    )


def _load_cache(path: Path, generator_stamp: str) -> dict[str, Any]:
    empty = {
        "schema_version": SCHEMA_VERSION,
        "generator_stamp": generator_stamp,
        "files": {},
    }
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty

    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("generator_stamp") != generator_stamp
    ):
        return empty
    if not isinstance(data.get("files"), dict):
        return empty
    return data


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _card_filename(node_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", node_id).strip("-").lower()
    if not slug:
        slug = "node"
    suffix = hashlib.sha1(node_id.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:80]}--{suffix}.md"


def _render_cards(output_dir: Path, graph: WorkspaceGraph) -> None:
    cards_dir = output_dir / "cards"
    if cards_dir.exists():
        shutil.rmtree(cards_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        node.id: _card_filename(node.id)
        for node in graph.nodes
    }
    outgoing: dict[str, list[WorkspaceEdge]] = {}
    incoming: dict[str, list[WorkspaceEdge]] = {}

    for edge in graph.edges:
        outgoing.setdefault(edge.source, []).append(edge)
        incoming.setdefault(edge.target, []).append(edge)

    for node in graph.nodes:
        lines = [
            f"# {node.label}",
            "",
            f"- ID: {node.id}",
            f"- Kind: {node.kind.value}",
            f"- Content hash: {node.content_hash}",
            "",
            "## Sources",
            "",
        ]

        if node.sources:
            for source in sorted(node.sources):
                lines.append(f"- {source.path} ({source.hash})")
        else:
            lines.append("- synthetic")

        links: list[tuple[str, WorkspaceEdge, str]] = []
        for edge in sorted(
            outgoing.get(node.id, []),
            key=lambda item: item.identity,
        ):
            links.append(("out", edge, edge.target))
        for edge in sorted(
            incoming.get(node.id, []),
            key=lambda item: item.identity,
        ):
            links.append(("in", edge, edge.source))

        lines.extend(["", "## Relations", ""])
        if links:
            for direction, edge, other in links:
                filename = filenames.get(other)
                target = f"[{other}]({filename})" if filename else other
                semantic = (
                    f" / {edge.semantic_type}"
                    if edge.semantic_type
                    else ""
                )
                arrow = "->" if direction == "out" else "<-"
                lines.append(
                    f"- {arrow} {edge.relation.value}{semantic}: {target}"
                )
        else:
            lines.append("- none")

        lines.extend(["", "## Properties", ""])
        pretty = json.dumps(
            node.properties,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        lines.extend(f"    {line}" for line in pretty.splitlines())
        lines.append("")
        _atomic_write(cards_dir / filenames[node.id], "\n".join(lines))

    index = [
        "# Knowledge Workspace",
        "",
        "Generated relationship/provenance cache.",
        "Regenerate with: python3 scripts/knowledge_workspace.py build",
        "",
        f"- Nodes: {len(graph.nodes)}",
        f"- Edges: {len(graph.edges)}",
        "",
        "## Nodes",
        "",
    ]
    for node in graph.nodes:
        index.append(
            f"- [{node.id}](cards/{filenames[node.id]}) - {node.kind.value}"
        )
    index.append("")
    _atomic_write(output_dir / "INDEX.md", "\n".join(index))


def build_workspace(
    root: str | Path,
    *,
    output_dir: str | Path | None = None,
    reuse: bool = True,
) -> BuildResult:
    root_path = Path(root).resolve()
    config = _load_workspace_config(root_path)
    output = _resolve_output(root_path, config, output_dir)

    cache_path = output / ".cache" / "source-index.json"
    generator_stamp = _generator_stamp()
    cache = (
        _load_cache(cache_path, generator_stamp)
        if reuse
        else {
            "schema_version": SCHEMA_VERSION,
            "generator_stamp": generator_stamp,
            "files": {},
        }
    )
    old_files = cache.get("files", {})

    parsed = 0
    reused = 0
    entries: dict[str, Any] = {}
    all_nodes: list[WorkspaceNode] = [_system_node()]
    all_edges: list[WorkspaceEdge] = []
    source_refs: list[SourceRef] = []

    for path in _discover_sources(root_path):
        relative = path.relative_to(root_path).as_posix()
        source_hash = hash_file(path)
        source_ref = SourceRef(path=relative, hash=source_hash)
        source_refs.append(source_ref)
        cached = old_files.get(relative)

        if reuse and cached and cached.get("hash") == source_hash:
            file_nodes = [
                WorkspaceNode.from_dict(item)
                for item in cached.get("nodes", [])
            ]
            file_edges = [
                WorkspaceEdge.from_dict(item)
                for item in cached.get("edges", [])
            ]
            reused += 1
        else:
            file_nodes, file_edges = parse_source(
                root_path,
                path,
                source_hash,
            )
            parsed += 1

        entries[relative] = {
            "hash": source_hash,
            "nodes": [item.to_dict() for item in file_nodes],
            "edges": [item.to_dict() for item in file_edges],
        }
        all_nodes.extend(file_nodes)
        all_edges.extend(file_edges)

    graph = WorkspaceGraph(
        nodes=_merge_nodes(all_nodes),
        edges=_merge_edges(all_edges),
    )
    validate_graph(graph)

    repo_digest = digest_sources(source_refs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_stamp": generator_stamp,
        "repo_digest": repo_digest,
        "sources": [
            source.to_dict()
            for source in sorted(source_refs)
        ],
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
    }

    _write_json(output / "manifest.json", manifest)
    _write_json(output / "graph.json", graph.to_dict())
    _render_cards(output, graph)
    _write_json(
        cache_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generator_stamp": generator_stamp,
            "files": {
                key: entries[key]
                for key in sorted(entries)
            },
        },
    )

    return BuildResult(
        parsed=parsed,
        reused=reused,
        files=len(source_refs),
        nodes=len(graph.nodes),
        edges=len(graph.edges),
        repo_digest=repo_digest,
        output_dir=output.relative_to(root_path).as_posix(),
    )


def check_workspace(
    root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> DriftReport:
    root_path = Path(root).resolve()
    config = _load_workspace_config(root_path)
    output = _resolve_output(root_path, config, output_dir)
    manifest_path = output / "manifest.json"

    current = {
        path.relative_to(root_path).as_posix(): hash_file(path)
        for path in _discover_sources(root_path)
    }

    if not manifest_path.exists():
        return DriftReport(
            added=sorted(current),
            generator_changed=True,
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DriftReport(
            added=sorted(current),
            generator_changed=True,
        )

    recorded = {
        item["path"]: item["hash"]
        for item in manifest.get("sources", [])
    }
    current_paths = set(current)
    recorded_paths = set(recorded)

    return DriftReport(
        added=sorted(current_paths - recorded_paths),
        changed=sorted(
            path
            for path in current_paths & recorded_paths
            if current[path] != recorded[path]
        ),
        removed=sorted(recorded_paths - current_paths),
        generator_changed=(
            manifest.get("generator_stamp") != _generator_stamp()
        ),
    )


def load_workspace_graph(
    root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> WorkspaceGraph:
    root_path = Path(root).resolve()
    config = _load_workspace_config(root_path)
    output = _resolve_output(root_path, config, output_dir)
    data = json.loads(
        (output / "graph.json").read_text(encoding="utf-8")
    )
    graph = WorkspaceGraph.from_dict(data)
    validate_graph(graph)
    return graph
