"""Knowledge core configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def _default_project_root() -> Path:
    override = os.environ.get("CHATBOT_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


class OllamaSettings(BaseModel):
    """Ollama LLM settings."""

    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2:latest")
    timeout: int = Field(default=120)
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=4096)


class ExtractionSettings(BaseModel):
    """Knowledge extraction settings."""

    min_fragment_length: int = Field(default=10)
    max_fragment_length: int = Field(default=500)
    ner_confidence_threshold: float = Field(default=0.5)
    fuzzy_match_threshold: float = Field(default=0.8)
    relation_confidence_threshold: float = Field(default=0.5)


class StoreSettings(BaseModel):
    """Knowledge storage settings."""

    graph_db_path: str = Field(default="data/ontology/graph.db")
    document_db_path: str = Field(default="data/ontology/documents.db")
    vector_db_path: str = Field(default="data/ontology/vectors")
    domain_data_path: Path = Field(default=Path("data/ontology/domain"))


class LoggingSettings(BaseModel):
    """Logging settings."""

    log_dir: str = Field(default="logs")
    log_file: str = Field(default="knowledge-core.log")
    log_level: str = Field(default="INFO")
    max_bytes: int = Field(default=10 * 1024 * 1024)
    backup_count: int = Field(default=5)


class Settings(BaseModel):
    """Knowledge core settings and canonical repository paths."""

    project_root: Path = Field(default_factory=_default_project_root)

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    entity_types_path: str = Field(default="config/ontology/entity_types.yaml")
    relation_types_path: str = Field(default="config/ontology/relation_types.yaml")
    alias_dictionary_path: str = Field(default="config/ontology/alias_dictionary.yaml")
    validation_schema_path: str = Field(default="config/ontology/validation_schema.yaml")
    static_domain_path: str = Field(default="config/ontology/static_domain.yaml")
    infrastructure_path: str = Field(default="config/ontology/infrastructure.yaml")

    class Config:
        arbitrary_types_allowed = True

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.project_root / candidate

    def get_config_path(self, config_name: str) -> Path:
        config_map = {
            "entity_types": self.entity_types_path,
            "relation_types": self.relation_types_path,
            "alias_dictionary": self.alias_dictionary_path,
            "validation_schema": self.validation_schema_path,
            "static_domain": self.static_domain_path,
            "infrastructure": self.infrastructure_path,
        }
        return self.resolve_path(config_map.get(config_name, config_name))

    def load_yaml_config(self, config_name: str) -> dict:
        config_path = self.get_config_path(config_name)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
