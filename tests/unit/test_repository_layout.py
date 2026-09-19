"""Canonical repository layout contracts."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_runtime_directories_exist():
    expected = (
        "apps/web",
        "services/gateway",
        "services/inference",
        "src/chatbot",
        "config",
        "data",
        "tests",
        "scripts",
        "deploy/docker",
        "deploy/compose",
    )
    for relative in expected:
        assert (REPO_ROOT / relative).is_dir(), relative


def test_active_chatbot_version_wrapper_is_removed():
    assert not (REPO_ROOT / "Chatbot_v6").exists()


def test_retired_snapshot_directories_are_removed_from_active_tree():
    retired = (
        "Chatbot_v1",
        "Chatbot_v2",
        "Chatbot_v3",
        "Chatbot_v4",
        "Chatbot_v5.final",
        "onTology_system_v9",
        "ontology_system_v11",
        "ontology_system_v12",
        "test_chatbot",
    )
    assert [name for name in retired if (REPO_ROOT / name).exists()] == []


def test_v13_remains_only_as_explicit_knowledge_core_migration_source():
    assert (REPO_ROOT / "ontology_system_v13").is_dir()
    lineage = (REPO_ROOT / "docs/history/version-lineage.md").read_text(encoding="utf-8")
    assert "임시 migration source" in lineage


def test_moon_release_commands_are_exact_workflow_commands():
    config = json.loads((REPO_ROOT / "moon.config.json").read_text(encoding="utf-8"))
    workflow = (REPO_ROOT / ".github/workflows/validate.yml").read_text(encoding="utf-8")

    release_commands = config["validation"]["profiles"]["release"]
    assert release_commands
    for command in release_commands:
        assert f"run: {command}" in workflow


def test_deployment_files_target_canonical_paths():
    compose = (REPO_ROOT / "deploy/compose/docker-compose.yml").read_text(encoding="utf-8")
    assert "deploy/docker/inference.Dockerfile" in compose
    assert "deploy/docker/gateway.Dockerfile" in compose
    assert "deploy/docker/web.Dockerfile" in compose
    assert "../../src:/app/src" in compose
    assert "../../services/inference:/app/services/inference" in compose
    assert "../../services/gateway:/app" in compose


def test_canonical_python_sources_do_not_import_legacy_modules_package():
    roots = (
        REPO_ROOT / "src",
        REPO_ROOT / "services/inference",
        REPO_ROOT / "config",
        REPO_ROOT / "scripts",
        REPO_ROOT / "tests",
    )
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            legacy_from = "from " + "modules."
            legacy_import = "import " + "modules."
            if legacy_from in source or legacy_import in source:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []
