"""
Constants - One Source of Truth

모든 상수를 단일 소스에서 관리합니다.
선택값을 설정으로 분리하여 유연성을 확보합니다.
"""

from enum import IntEnum, Enum
from typing import Final


# ============================================================================
# Status Codes (설정 가능한 상태 코드)
# ============================================================================

class StatusCode(IntEnum):
    """
    상태 코드 정의
    
    사용자가 원하는 방식으로 성공/실패를 표현할 수 있습니다:
    - 0/1 방식 (SUCCESS=0, FAILURE=1)
    - 1/0 방식 (SUCCESS=1, FAILURE=0)
    - HTTP 방식 (SUCCESS=200, FAILURE=500)
    
    기본값: 0=성공, 1=실패
    """
    SUCCESS = 0
    FAILURE = 1
    PARTIAL_SUCCESS = 2
    TIMEOUT = 3
    NOT_FOUND = 4


class ErrorCode(Enum):
    """
    에러 코드 정의
    
    각 에러 유형에 대한 고유 코드를 정의합니다.
    """
    # Configuration Errors (E001-E099)
    E001_CONFIG_FILE_NOT_FOUND = "E001"
    E002_INVALID_CONFIG_FORMAT = "E002"
    E003_MISSING_REQUIRED_CONFIG = "E003"
    
    # Embedding Errors (E100-E199)
    E101_EMBEDDING_MODEL_LOAD_FAILED = "E101"
    E102_EMBEDDING_GENERATION_FAILED = "E102"
    E103_EMBEDDING_DIMENSION_MISMATCH = "E103"
    
    # Retrieval Errors (E200-E299)
    E201_VECTOR_STORE_NOT_FOUND = "E201"
    E202_RETRIEVAL_TIMEOUT = "E202"
    E203_RETRIEVAL_FAILED = "E203"
    E204_BM25_INDEX_ERROR = "E204"
    
    # Generation Errors (E300-E399)
    E301_LLM_CONNECTION_FAILED = "E301"
    E302_LLM_TIMEOUT = "E302"
    E303_LLM_RESPONSE_EMPTY = "E303"
    E304_LLM_RESPONSE_INVALID = "E304"
    
    # Preprocessing Errors (E400-E499)
    E401_PDF_LOAD_FAILED = "E401"
    E402_TEXT_EXTRACTION_FAILED = "E402"
    E403_OCR_CORRECTION_FAILED = "E403"
    
    # Chunking Errors (E500-E599)
    E501_CHUNKING_FAILED = "E501"
    E502_INVALID_CHUNK_SIZE = "E502"
    
    # Pipeline Errors (E600-E699)
    E601_PIPELINE_INIT_FAILED = "E601"
    E602_PIPELINE_EXECUTION_FAILED = "E602"
    
    # System Errors (E900-E999)
    E901_OUT_OF_MEMORY = "E901"
    E902_DISK_SPACE_INSUFFICIENT = "E902"
    E903_PERMISSION_DENIED = "E903"


# ============================================================================
# Default Values
# ============================================================================

# Chunking
DEFAULT_CHUNK_SIZE: Final[int] = 802
DEFAULT_CHUNK_OVERLAP: Final[int] = 200
DEFAULT_WASTEWATER_CHUNK_SIZE: Final[int] = 900
DEFAULT_WASTEWATER_OVERLAP_RATIO: Final[float] = 0.25
DEFAULT_NUMERIC_CONTEXT_WINDOW: Final[int] = 3
DEFAULT_ENABLE_NUMERIC_CHUNKING: Final[bool] = True
DEFAULT_PRESERVE_TABLE_CONTEXT: Final[bool] = True
DEFAULT_USE_PAGE_BASED_CHUNKING: Final[bool] = True
DEFAULT_ENABLE_BOUNDARY_SNAP: Final[bool] = True
DEFAULT_BOUNDARY_SNAP_MARGIN_RATIO: Final[float] = 0.10

# Embedding
DEFAULT_EMBEDDING_MODEL: Final[str] = "jhgan/ko-sroberta-multitask"
DEFAULT_EMBEDDING_BATCH_SIZE: Final[int] = 64  # 🚀 최적화 5: 배치 크기 증가 (32→64)
DEFAULT_EMBEDDING_DEVICE: Final[str] = "cuda"  # or "cpu"

# LLM
DEFAULT_LLM_MODEL: Final[str] = "qwen2.5:3b-instruct-q4_K_M"
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.0
DEFAULT_LLM_TOP_P: Final[float] = 0.9
DEFAULT_LLM_TOP_K: Final[int] = 40
DEFAULT_LLM_NUM_CTX: Final[int] = 8192
DEFAULT_LLM_NUM_PREDICT: Final[int] = 512
DEFAULT_LLM_KEEP_ALIVE_MINUTES: Final[int] = 5

# Retrieval
DEFAULT_RETRIEVAL_TOP_K: Final[int] = 50
DEFAULT_RETRIEVAL_VECTOR_WEIGHT: Final[float] = 0.2  # v5와 동일: 키워드 검색에 집중
DEFAULT_RETRIEVAL_BM25_WEIGHT: Final[float] = 0.8   # v5와 동일: BM25 우선
DEFAULT_RETRIEVAL_RRF_K: Final[int] = 60

# Filtering
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.20
DEFAULT_CONFIDENCE_THRESHOLD_NUMERIC: Final[float] = 0.12
DEFAULT_CONFIDENCE_THRESHOLD_LONG: Final[float] = 0.13
DEFAULT_GUARD_OVERLAP_THRESHOLD: Final[float] = 0.10
DEFAULT_CONTEXT_MIN_OVERLAP: Final[float] = 0.07
DEFAULT_KEYWORD_FILTER_MIN: Final[int] = 1
DEFAULT_RERANK_THRESHOLD: Final[float] = 0.41

# Context
DEFAULT_CONTEXT_K: Final[int] = 6
DEFAULT_CONTEXT_K_NUMERIC: Final[int] = 8
DEFAULT_CONTEXT_K_MIN: Final[int] = 4
DEFAULT_CONTEXT_K_MAX: Final[int] = 10

# Timeouts (seconds)
DEFAULT_LLM_TIMEOUT: Final[int] = 60
DEFAULT_RERANK_TIMEOUT: Final[int] = 20
DEFAULT_SEARCH_TIMEOUT: Final[int] = 10
DEFAULT_EMBEDDING_TIMEOUT: Final[int] = 30

# Retries
DEFAULT_LLM_RETRIES: Final[int] = 3
DEFAULT_LLM_RETRY_BACKOFF_MS: Final[int] = 800

# Cache
DEFAULT_CACHE_ENABLED: Final[bool] = True
DEFAULT_CACHE_SIZE: Final[int] = 256

# Deduplication
DEFAULT_JACCARD_THRESHOLD: Final[float] = 0.9
DEFAULT_SEMANTIC_THRESHOLD: Final[float] = 0.0
DEFAULT_MIN_CHUNK_LENGTH: Final[int] = 50

# Logging
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_LOG_FORMAT: Final[str] = "json"  # or "text"
DEFAULT_LOG_DIR: Final[str] = "logs"
DEFAULT_LOG_MAX_SIZE: Final[str] = "10MB"
DEFAULT_LOG_MAX_FILES: Final[int] = 3

# Data Paths
DEFAULT_DATA_DIR: Final[str] = "data"
DEFAULT_VECTOR_STORE_DIR: Final[str] = "vector_store"
DEFAULT_DOMAIN_DICT_PATH: Final[str] = "data/domain_dictionary.json"

# Ollama
DEFAULT_OLLAMA_HOST: Final[str] = "ollama"  # Docker service name
DEFAULT_OLLAMA_PORT: Final[int] = 11434


# ============================================================================
# Question Types
# ============================================================================

class QuestionType(Enum):
    """질문 유형 정의"""
    NUMERIC = "numeric"
    DEFINITION = "definition"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    PROBLEM = "problem"
    SYSTEM_INFO = "system_info"
    TECHNICAL_SPEC = "technical_spec"
    OPERATIONAL = "operational"
    GENERAL = "general"


# ============================================================================
# Mode Types
# ============================================================================

class ModeType(Enum):
    """실행 모드 정의"""
    ACCURACY = "accuracy"  # 정확도 우선
    SPEED = "speed"        # 속도 우선
    BALANCED = "balanced"  # 균형


# ============================================================================
# Measurement Units (정수장 특화)
# ============================================================================

# 단위 목록
UNIT_SYNONYMS: Final[dict] = {
    "mg/l": {"ppm"},
    "ppm": {"mg/l"},
    "ug/l": {"ppb"},
    "ppb": {"ug/l"},
    "us/cm": {"µs/cm", "μs/cm"},
    "°c": {"℃"},
    "℃": {"°c"},
    "ntu": {"탁도", "turbidity"},
    "ph": {"산성도", "알칼리도"},
    "do": {"용존산소", "dissolved oxygen"},
    "bod": {"생물학적산소요구량", "biological oxygen demand"},
    "cod": {"화학적산소요구량", "chemical oxygen demand"},
    "toc": {"총유기탄소", "total organic carbon"},
    "cfu": {"대장균군", "coliform"},
    "m³/d": {"m3/d", "m3/day"},
    "m³/h": {"m3/h", "m3/hour"},
    "l/s": {"liter/s", "liter/sec"},
}

# 단위 변환 계수
UNIT_CONVERSIONS: Final[dict] = {
    ("l/s", "m3/d"): 86.4,
    ("m3/d", "l/s"): 1.0 / 86.4,
    ("mg/l", "ppm"): 1.0,
    ("ppm", "mg/l"): 1.0,
    ("ug/l", "ppb"): 1.0,
    ("ppb", "ug/l"): 1.0,
    ("m³/d", "l/s"): 1.0 / 86.4,
    ("l/s", "m³/d"): 86.4,
    ("m³/h", "l/s"): 1.0 / 3.6,
    ("l/s", "m³/h"): 3.6,
    ("m³/h", "m³/d"): 24.0,
    ("m³/d", "m³/h"): 1.0 / 24.0,
    ("kgf/cm²", "bar"): 0.980665,
    ("bar", "kgf/cm²"): 1.01972,
    ("mpa", "kgf/cm²"): 10.1972,
    ("kgf/cm²", "mpa"): 0.0980665,
}

