"""Canonical Knowledge Core contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from chatbot.knowledge.bootstrap import build_graph_repository, load_config
from chatbot.knowledge.domain.kg_adapter import DomainKGAdapter
from chatbot.knowledge.extraction import ExtractionPipeline
from chatbot.knowledge.settings import Settings
from chatbot.knowledge.storage.inmemory_repository import InMemoryGraphRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_knowledge_settings_resolve_canonical_paths():
    settings = Settings(project_root=PROJECT_ROOT)

    expected_configs = {
        "entity_types": PROJECT_ROOT / "config/ontology/entity_types.yaml",
        "relation_types": PROJECT_ROOT / "config/ontology/relation_types.yaml",
        "alias_dictionary": PROJECT_ROOT / "config/ontology/alias_dictionary.yaml",
        "validation_schema": PROJECT_ROOT / "config/ontology/validation_schema.yaml",
        "static_domain": PROJECT_ROOT / "config/ontology/static_domain.yaml",
        "infrastructure": PROJECT_ROOT / "config/ontology/infrastructure.yaml",
    }
    for name, expected in expected_configs.items():
        assert settings.get_config_path(name) == expected
        assert expected.is_file()

    assert settings.resolve_path(settings.store.domain_data_path) == (
        PROJECT_ROOT / "data/ontology/domain"
    )


def test_infrastructure_config_substitutes_neo4j_password(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "test-secret")
    config = load_config(str(PROJECT_ROOT / "config/ontology/infrastructure.yaml"))

    assert config["storage"]["backend"] == "inmemory"
    assert config["storage"]["neo4j"]["password"] == "test-secret"
    assert "password" != config["storage"]["neo4j"]["password"]


def test_inmemory_repository_and_domain_seed_bootstrap():
    repository = build_graph_repository({"storage": {"backend": "inmemory"}})

    assert isinstance(repository, InMemoryGraphRepository)

    adapter = DomainKGAdapter(repository=repository, read_only=False)
    adapter.load_domain_data()

    assert repository.count_entities() == 0
    assert repository.count_relations() == 0


def test_rule_based_extraction_pipeline_runs_without_llm():
    pipeline = ExtractionPipeline(llm_client=None, use_llm=False)
    result = pipeline.process(
        raw_text="정수장의 탁도가 증가하면 응집제 주입률을 조정한다.",
        doc_id="knowledge-smoke",
    )

    assert result.doc_id == "knowledge-smoke"
    assert result.source_uri == "document:knowledge-smoke"
    assert result.source_hash is not None
    assert len(result.source_hash) == 64
    assert result.error_count >= 0
    assert isinstance(result.warning_messages, list)


def test_chatbot_root_import_does_not_eager_load_heavy_rag_dependencies():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    probe = (
        "import sys, chatbot; "
        "assert 'numpy' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr


class _FakeResult:
    def single(self):
        return None

    def __iter__(self):
        return iter(())


class _FakeSession:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))
        return _FakeResult()


class _FakeDriver:
    def __init__(self):
        self.calls = []

    def session(self, *, database):
        return _FakeSession(self.calls)

    def close(self):
        return None


class _FakeGraphDatabase:
    driver_instance = _FakeDriver()

    @classmethod
    def driver(cls, uri, auth):
        cls.driver_instance = _FakeDriver()
        return cls.driver_instance


def test_neo4j_repository_uses_bound_values_and_quoted_relation_types(monkeypatch):
    import chatbot.knowledge.storage.neo4j_repository as module

    monkeypatch.setattr(module, "GraphDatabase", _FakeGraphDatabase)
    repository = module.Neo4jGraphRepository(
        uri="bolt://example",
        user="neo4j",
        password="secret",
    )

    repository.upsert_entity("entity-1", ["DomainEntity"], {"name": "A"})
    repository.upsert_relation(
        "entity-1",
        "domain:Affect",
        "entity-2",
        {"weight": 0.9},
    )
    repository.delete_relation("entity-1", "domain:Affect", "entity-2")

    calls = repository.driver.calls
    assert len(calls) == 3

    entity_query, entity_params = calls[0]
    assert "{id: $id}" in entity_query
    assert "SET n += $props" in entity_query
    assert entity_params["id"] == "entity-1"

    relation_query, relation_params = calls[1]
    assert "{id: $src_id}" in relation_query
    assert "{id: $dst_id}" in relation_query
    assert "`domain:Affect`" in relation_query
    assert "SET r += $props" in relation_query
    assert relation_params["src_id"] == "entity-1"

    delete_query, delete_params = calls[2]
    assert "`domain:Affect`" in delete_query
    assert "$src_id" in delete_query
    assert "$dst_id" in delete_query
    assert delete_params["dst_id"] == "entity-2"


@pytest.mark.parametrize("invalid", ["", None])
def test_neo4j_identifier_rejects_empty_values(invalid):
    import chatbot.knowledge.storage.neo4j_repository as module

    with pytest.raises(ValueError):
        module._quote_identifier(invalid)


def test_ontology_infrastructure_has_no_plaintext_default_password():
    text = (PROJECT_ROOT / "config/ontology/infrastructure.yaml").read_text(encoding="utf-8")

    assert 'password: "password"' not in text
    assert 'password: "${NEO4J_PASSWORD}"' in text
