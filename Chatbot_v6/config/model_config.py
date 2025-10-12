"""
Model Configuration

임베딩 및 LLM 모델 설정을 정의합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_P,
    DEFAULT_LLM_TOP_K,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_NUM_PREDICT,
    DEFAULT_LLM_KEEP_ALIVE_MINUTES,
)
from modules.core.exceptions import ConfigurationError


@dataclass
class EmbeddingModelConfig:
    """임베딩 모델 설정"""
    
    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str = DEFAULT_EMBEDDING_DEVICE  # "cuda" or "cpu"
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    normalize_embeddings: bool = True
    show_progress_bar: bool = False
    
    def validate(self) -> None:
        """설정 값 검증"""
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ConfigurationError(
                f"device must be 'cuda', 'cpu', or 'auto', got {self.device}",
                error_code="E003"
            )
        if self.batch_size < 1:
            raise ConfigurationError(
                "batch_size must be >= 1",
                error_code="E003"
            )


@dataclass
class LLMModelConfig:
    """LLM 모델 설정"""
    
    model_name: str = DEFAULT_LLM_MODEL
    temperature: float = DEFAULT_LLM_TEMPERATURE
    top_p: float = DEFAULT_LLM_TOP_P
    top_k: int = DEFAULT_LLM_TOP_K
    num_ctx: int = DEFAULT_LLM_NUM_CTX
    num_predict: int = DEFAULT_LLM_NUM_PREDICT
    repeat_penalty: float = 1.1
    keep_alive_minutes: int = DEFAULT_LLM_KEEP_ALIVE_MINUTES
    
    # Ollama connection settings
    host: str = "ollama"  # Docker service name or localhost
    port: int = 11434
    timeout: int = 60
    
    def validate(self) -> None:
        """설정 값 검증"""
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigurationError(
                "temperature must be between 0.0 and 2.0",
                error_code="E003"
            )
        if not (0.0 <= self.top_p <= 1.0):
            raise ConfigurationError(
                "top_p must be between 0.0 and 1.0",
                error_code="E003"
            )
        if self.top_k < 1:
            raise ConfigurationError(
                "top_k must be >= 1",
                error_code="E003"
            )
        if self.num_ctx < 1:
            raise ConfigurationError(
                "num_ctx must be >= 1",
                error_code="E003"
            )
        if self.port < 1 or self.port > 65535:
            raise ConfigurationError(
                "port must be between 1 and 65535",
                error_code="E003"
            )
    
    def get_ollama_url(self) -> str:
        """Ollama API URL 생성"""
        return f"http://{self.host}:{self.port}"
    
    def get_generation_options(self) -> Dict[str, Any]:
        """생성 옵션 딕셔너리 반환"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }


@dataclass
class ModelConfig:
    """전체 모델 설정"""
    
    embedding: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    llm: LLMModelConfig = field(default_factory=LLMModelConfig)
    
    def validate(self) -> None:
        """모든 설정 검증"""
        self.embedding.validate()
        self.llm.validate()

