from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class Thresholds:
    # Base thresholds - 더 관대한 임계값으로 조정
    confidence_threshold: float = 0.20  # 0.25 -> 0.20
    confidence_threshold_numeric: float = 0.12  # 0.16 -> 0.12
    confidence_threshold_long: float = 0.13  # 0.17 -> 0.13
    guard_overlap_threshold: float = 0.10  # 0.12 -> 0.10
    guard_key_tokens_min: int = 1
    base_no_answer_confidence: float = 0.60  # 0.65 -> 0.60
    analyzer_threshold_delta: float = -0.02


@dataclass
class DeduplicationPolicy:
    # 중복 제거 관련 설정
    jaccard_threshold: float = 0.9  # Jaccard 유사도 임계값
    semantic_threshold: float = 0.0  # 의미적 유사도 임계값 (0.0이면 비활성화)
    enable_semantic_dedup: bool = False  # 의미적 중복 제거 활성화
    min_chunk_length: int = 50  # 중복 제거 대상 최소 청크 길이


@dataclass
class RRFPolicy:
    # Default weights; effective may be adjusted by analyzer/type
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    base_rrf_k: int = 60  # standard RRF constant


@dataclass
class ContextPolicy:
    k_default: int = 6
    k_numeric: int = 8
    k_definition_min: int = 4
    k_definition_max: int = 6
    k_min: int = 4
    k_max: int = 10
    allow_neighbor_from_adjacent_page: bool = True


@dataclass
class ModeFlags:
    mode: str = "accuracy"  # or "speed"
    use_cross_reranker: bool = False
    use_gpu: bool = False
    store_backend: str = "auto"  # "faiss", "hnsw", or "auto" (fallback: inmem)
    rerank_top_n: int = 50
    # Retrieval execution/control flags
    enable_parallel_search: bool = True
    enable_retrieval_cache: bool = True
    retrieval_cache_size: int = 256


@dataclass
class DomainConfig:
    # Optional path to a JSON file with domain terms
    # Format: {"units": [...], "keywords": [...], "procedural": [...], "comparative": [...], "definition": [...], "problem": [...]}
    domain_dict_path: Optional[str] = None


@dataclass
class PipelineConfig:
    thresholds: Thresholds = Thresholds()
    rrf: RRFPolicy = RRFPolicy()
    context: ContextPolicy = ContextPolicy()
    flags: ModeFlags = ModeFlags()
    domain: DomainConfig = DomainConfig()
    deduplication: DeduplicationPolicy = DeduplicationPolicy()
    seed: int = 42
    model_name: str = "qwen2:1.5b-instruct-q4_K_M"
    embedding_model: str = "jhgan/ko-sroberta-multitask"
    vector_store_dir: str = "vector_store"
    llm_retries: int = 2
    llm_retry_backoff_ms: int = 400

    def to_dict(self) -> Dict:
        return asdict(self)

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]
