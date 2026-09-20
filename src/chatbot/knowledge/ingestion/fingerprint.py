"""Fingerprint selective-ingestion code and ontology policy."""

from __future__ import annotations

import hashlib
from pathlib import Path

from chatbot.knowledge.workspace.hashing import hash_file


PROCESSOR_PACKAGES = (
    "shared",
    "extraction",
    "validation",
    "domain",
    "evidence",
    "ingestion",
    "llm",
    "workspace",
)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def processor_stamp(project_root: str | Path | None = None) -> str:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else default_project_root()
    )
    knowledge_root = root / "src/chatbot/knowledge"

    paths: set[Path] = {
        path
        for path in knowledge_root.glob("*.py")
        if path.is_file()
    }
    for package in PROCESSOR_PACKAGES:
        package_root = knowledge_root / package
        if not package_root.is_dir():
            raise FileNotFoundError(
                f"processor package not found: {package_root}"
            )
        paths.update(
            path
            for path in package_root.rglob("*.py")
            if path.is_file()
        )

    shared_models = knowledge_root / "shared/models.py"
    paths.add(shared_models)

    ontology_root = root / "config/ontology"
    if not ontology_root.is_dir():
        raise FileNotFoundError(
            f"ontology config directory not found: {ontology_root}"
        )
    paths.update(
        path
        for path in ontology_root.glob("*.yaml")
        if path.is_file()
    )

    lines = []
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        lines.append(f"{relative}:{hash_file(path)}")

    return hashlib.sha256(
        "\n".join(lines).encode("utf-8")
    ).hexdigest()
