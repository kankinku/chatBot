"""
Pipeline Configuration

파이프라인 동작을 제어하는 모든 설정을 정의합니다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Any

from .constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD_NUMERIC,
    DEFAULT_CONFIDENCE_THRESHOLD_LONG,
    DEFAULT_GUARD_OVERLAP_THRESHOLD,
    DEFAULT_CONTEXT_MIN_OVERLAP,
    DEFAULT_KEYWORD_FILTER_MIN,
    DEFAULT_RERANK_THRESHOLD,
    DEFAULT_RETRIEVAL_VECTOR_WEIGHT,
    DEFAULT_RETRIEVAL_BM25_WEIGHT,
    DEFAULT_RETRIEVAL_RRF_K,
    DEFAULT_CONTEXT_K,
    DEFAULT_CONTEXT_K_NUMERIC,
    DEFAULT_CONTEXT_K_MIN,
    DEFAULT_CONTEXT_K_MAX,
    DEFAULT_JACCARD_THRESHOLD,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_MIN_CHUNK_LENGTH,
    DEFAULT_CACHE_ENABLED,
    DEFAULT_CACHE_SIZE,
    DEFAULT_LLM_RETRIES,
    DEFAULT_LLM_RETRY_BACKOFF_MS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_VECTOR_STORE_DIR,
    DEFAULT_DOMAIN_DICT_PATH,
    ModeType,
)
from modules.core.exceptions import ConfigurationError


@dataclass
class ThresholdsConfig:
    """임계값 설정"""
    
    # Base thresholds
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    confidence_threshold_numeric: float = DEFAULT_CONFIDENCE_THRESHOLD_NUMERIC
    confidence_threshold_long: float = DEFAULT_CONFIDENCE_THRESHOLD_LONG
    guard_overlap_threshold: float = DEFAULT_GUARD_OVERLAP_THRESHOLD
    guard_key_tokens_min: int = 1
    base_no_answer_confidence: float = 0.60
    analyzer_threshold_delta: float = -0.02
    
    # Context quality thresholds
    context_min_overlap: float = DEFAULT_CONTEXT_MIN_OVERLAP
    keyword_filter_min: int = DEFAULT_KEYWORD_FILTER_MIN
    
    # Reranker threshold
    rerank_threshold: float = DEFAULT_RERANK_THRESHOLD
    
    # QA mismatch detection thresholds
    qa_overlap_min: float = 0.07
    qa_token_hit_min_ratio: float = 0.52
    answer_ctx_min_overlap: float = 0.07
    numeric_preservation_min: float = 0.62
    numeric_preservation_severe: float = 0.32
    mismatch_trigger_count: int = 2
    
    def validate(self) -> None:
        """설정 값 검증"""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError(
                "confidence_threshold must be between 0.0 and 1.0",
                error_code="E003"
            )
        if not (0.0 <= self.rerank_threshold <= 1.0):
            raise ConfigurationError(
                "rerank_threshold must be between 0.0 and 1.0",
                error_code="E003"
            )


@dataclass
class RRFConfig:
    """RRF (Reciprocal Rank Fusion) 설정"""
    
    vector_weight: float = DEFAULT_RETRIEVAL_VECTOR_WEIGHT
    bm25_weight: float = DEFAULT_RETRIEVAL_BM25_WEIGHT
    base_rrf_k: int = DEFAULT_RETRIEVAL_RRF_K
    
    def validate(self) -> None:
        """설정 값 검증"""
        total_weight = self.vector_weight + self.bm25_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ConfigurationError(
                f"vector_weight + bm25_weight must equal 1.0, got {total_weight}",
                error_code="E003"
            )
        if self.base_rrf_k < 1:
            raise ConfigurationError(
                "base_rrf_k must be >= 1",
                error_code="E003"
            )


@dataclass
class ContextConfig:
    """컨텍스트 설정"""
    
    k_default: int = DEFAULT_CONTEXT_K
    k_numeric: int = DEFAULT_CONTEXT_K_NUMERIC
    k_definition_min: int = 4
    k_definition_max: int = 6
    k_min: int = DEFAULT_CONTEXT_K_MIN
    k_max: int = DEFAULT_CONTEXT_K_MAX
    allow_neighbor_from_adjacent_page: bool = True
    
    def validate(self) -> None:
        """설정 값 검증"""
        if self.k_min > self.k_max:
            raise ConfigurationError(
                f"k_min ({self.k_min}) must be <= k_max ({self.k_max})",
                error_code="E003"
            )
        if self.k_default < self.k_min or self.k_default > self.k_max:
            raise ConfigurationError(
                f"k_default ({self.k_default}) must be between k_min and k_max",
                error_code="E003"
            )


@dataclass
class ModeConfig:
    """실행 모드 설정"""
    
    mode: str = "accuracy"  # "accuracy" or "speed"
    use_cross_reranker: bool = False
    use_gpu: bool = False
    store_backend: str = "auto"  # "faiss", "hnsw", or "auto"
    rerank_top_n: int = 50
    
    # Retrieval execution/control flags
    enable_parallel_search: bool = True
    enable_retrieval_cache: bool = DEFAULT_CACHE_ENABLED
    retrieval_cache_size: int = DEFAULT_CACHE_SIZE
    
    # GPU 강제 사용 플래그
    force_gpu_embedding: bool = False  # 임베딩 GPU 강제 사용
    force_gpu_faiss: bool = False      # FAISS GPU 강제 사용
    
    def validate(self) -> None:
        """설정 값 검증"""
        valid_modes = {"accuracy", "speed", "balanced"}
        if self.mode not in valid_modes:
            raise ConfigurationError(
                f"mode must be one of {valid_modes}, got {self.mode}",
                error_code="E003"
            )
        
        valid_backends = {"faiss", "hnsw", "auto"}
        if self.store_backend not in valid_backends:
            raise ConfigurationError(
                f"store_backend must be one of {valid_backends}, got {self.store_backend}",
                error_code="E003"
            )


@dataclass
class DeduplicationConfig:
    """중복 제거 설정"""
    
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD
    enable_semantic_dedup: bool = False
    min_chunk_length: int = DEFAULT_MIN_CHUNK_LENGTH
    
    def validate(self) -> None:
        """설정 값 검증"""
        if not (0.0 <= self.jaccard_threshold <= 1.0):
            raise ConfigurationError(
                "jaccard_threshold must be between 0.0 and 1.0",
                error_code="E003"
            )
        if self.enable_semantic_dedup and not (0.0 <= self.semantic_threshold <= 1.0):
            raise ConfigurationError(
                "semantic_threshold must be between 0.0 and 1.0",
                error_code="E003"
            )


@dataclass
class DomainConfig:
    """도메인 특화 설정"""
    
    domain_dict_path: Optional[str] = DEFAULT_DOMAIN_DICT_PATH
    
    def validate(self) -> None:
        """설정 값 검증"""
        if self.domain_dict_path:
            path = Path(self.domain_dict_path)
            if not path.exists():
                raise ConfigurationError(
                    f"Domain dictionary not found: {self.domain_dict_path}",
                    error_code="E001"
                )


@dataclass
class PipelineConfig:
    """전체 파이프라인 설정"""
    
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    rrf: RRFConfig = field(default_factory=RRFConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    flags: ModeConfig = field(default_factory=ModeConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    
    seed: int = 42
    model_name: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    vector_store_dir: str = DEFAULT_VECTOR_STORE_DIR
    llm_retries: int = DEFAULT_LLM_RETRIES
    llm_retry_backoff_ms: int = DEFAULT_LLM_RETRY_BACKOFF_MS
    
    def validate(self) -> None:
        """모든 설정 검증"""
        self.thresholds.validate()
        self.rrf.validate()
        self.context.validate()
        self.flags.validate()
        self.deduplication.validate()
        self.domain.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)
    
    def config_hash(self) -> str:
        """설정의 해시값 생성 (캐싱용)"""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        """딕셔너리에서 생성"""
        # Nested dataclasses 처리
        if "thresholds" in data and isinstance(data["thresholds"], dict):
            data["thresholds"] = ThresholdsConfig(**data["thresholds"])
        if "rrf" in data and isinstance(data["rrf"], dict):
            data["rrf"] = RRFConfig(**data["rrf"])
        if "context" in data and isinstance(data["context"], dict):
            data["context"] = ContextConfig(**data["context"])
        if "flags" in data and isinstance(data["flags"], dict):
            data["flags"] = ModeConfig(**data["flags"])
        if "deduplication" in data and isinstance(data["deduplication"], dict):
            data["deduplication"] = DeduplicationConfig(**data["deduplication"])
        if "domain" in data and isinstance(data["domain"], dict):
            data["domain"] = DomainConfig(**data["domain"])
        
        config = cls(**data)
        config.validate()
        return config
    
    @classmethod
    def from_file(cls, file_path: str | Path) -> PipelineConfig:
        """파일에서 설정 로드"""
        path = Path(file_path)
        if not path.exists():
            raise ConfigurationError(
                f"Config file not found: {file_path}",
                error_code="E001"
            )
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix == ".json":
                    data = json.load(f)
                elif path.suffix in {".yaml", ".yml"}:
                    import yaml
                    data = yaml.safe_load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {path.suffix}",
                        error_code="E002"
                    )
            
            return cls.from_dict(data)
        
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in config file: {e}",
                error_code="E002"
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load config from {file_path}: {e}",
                error_code="E002"
            ) from e
    
    def save_to_file(self, file_path: str | Path) -> None:
        """설정을 파일로 저장"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                if path.suffix == ".json":
                    json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
                elif path.suffix in {".yaml", ".yml"}:
                    import yaml
                    yaml.safe_dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {path.suffix}",
                        error_code="E002"
                    )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save config to {file_path}: {e}",
                error_code="E002"
            ) from e

