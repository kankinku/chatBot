# 개선된 PDF RAG 시스템 설계 문서

## 문서 정보
- **버전**: 2.0
- **작성일**: 2025년 10월 9일
- **대상**: 정수처리 챗봇 차세대 시스템
- **기반**: v5.final 결함 분석 및 개선안

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [핵심 설계 원칙](#2-핵심-설계-원칙)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [설정 관리 체계](#4-설정-관리-체계)
5. [청킹 시스템 재설계](#5-청킹-시스템-재설계)
6. [질문 분류 시스템](#6-질문-분류-시스템)
7. [검색 및 필터링 파이프라인](#7-검색-및-필터링-파이프라인)
8. [캐싱 시스템](#8-캐싱-시스템)
9. [에러 처리 전략](#9-에러-처리-전략)
10. [메타데이터 최적화](#10-메타데이터-최적화)
11. [성능 모니터링](#11-성능-모니터링)
12. [테스트 전략](#12-테스트-전략)

---

## 1. 시스템 개요

### 1.1 설계 목표
- **성능**: 기존 대비 50% 속도 향상
- **유지보수성**: 모듈 간 결합도 최소화
- **확장성**: 새 기능 추가 시 기존 코드 수정 불필요
- **안정성**: 모든 에러 케이스 명시적 처리

### 1.2 주요 개선사항
1. **청킹 계층 추상화**: 중복 제거 및 O(N log N) 최적화
2. **질문 분류 단순화**: 9종 → 6종 핵심 카테고리
3. **설정 외부화**: YAML 기반 One Source of Truth
4. **필터링 최적화**: Lazy evaluation + 캐싱
5. **신뢰도 계산 개선**: Robust Z-score
6. **중앙 캐시 관리**: CacheHub 도입
7. **메타데이터 경량화**: 참조 기반 구조
8. **에러 처리 강화**: 계층별 명시적 처리
9. **테스트 자동화**: pytest + 벤치마크

### 1.3 아키텍처 개요
```
[PDF 입력]
    ↓
[OCR + LLM 후처리] ─→ [Error Handler]
    ↓
[Chunking System] ─→ [Index Builder]
    ├─ BaseChunker (추상)
    ├─ StandardChunker
    ├─ WastewaterChunker
    └─ NumericChunker
    ↓
[Embedding + Vector Store]
    ↓
[Query Pipeline]
    ├─ Question Analyzer
    ├─ Hybrid Retrieval
    ├─ Reranker
    └─ Filter Pipeline
    ↓
[Response Generator]
    ↓
[Validation + Logging]
```

---

## 2. 핵심 설계 원칙

### 2.1 One Source of Truth (SSOT)
**규칙**: 모든 설정, 상수, 열거형은 단일 위치에서 관리

#### 적용 영역
```yaml
# config/constants.yaml
error_codes:
  SUCCESS: 0
  FAILURE: 1
  PARTIAL_SUCCESS: 2
  VALIDATION_ERROR: 3
  TIMEOUT_ERROR: 4

chunk_types:
  STANDARD: "standard"
  NUMERIC_EXPANDED: "numeric_expanded"
  WASTEWATER: "wastewater"

question_types:
  DEFINITION: "definition"
  NUMERIC: "numeric"
  PROCEDURE: "procedure"
  COMPARISON: "comparison"
  PROBLEM: "problem"
  GENERAL: "general"

filter_stages:
  PRE_FILTER: 0
  RERANK: 1
  CONFIDENCE: 2
  DIVERSITY: 3
```

#### 코드 사용 예시
```python
from config.constants import ErrorCode, ChunkType, QuestionType

# 잘못된 방식 (금지)
if status == 0:  # Magic number
    return "success"

# 올바른 방식
if status == ErrorCode.SUCCESS:
    return "success"
```

### 2.2 선택값 Config 관리
**규칙**: 모든 선택 가능한 값, 플래그, 옵션은 config 파일에 명시

```yaml
# config/options.yaml
features:
  enable_llm_correction: true
  enable_numeric_chunking: true
  enable_wastewater_mode: false
  enable_dynamic_k: true

processing_modes:
  ocr_correction: "selective"  # selective | full | disabled
  chunk_strategy: "hybrid"     # standard | hybrid | adaptive
  filter_pipeline: "full"      # minimal | standard | full

optimization:
  enable_parallel_search: true
  enable_cache: true
  cache_strategy: "lru"        # lru | fifo | lfu
```

### 2.3 에러 처리 설계
**규칙**: 모든 에러는 명시적으로 처리하고 로깅

#### 에러 계층
```
Level 1: Critical (시스템 중단)
  - 설정 파일 누락
  - 필수 모델 로드 실패
  
Level 2: Error (기능 실패)
  - LLM API 호출 실패
  - 벡터 인덱스 손상
  
Level 3: Warning (부분 실패)
  - OCR 일부 페이지 실패
  - 캐시 미스
  
Level 4: Info (정상 작동)
  - 처리 완료
  - 성능 메트릭
```

#### 에러 처리 패턴
```python
# core/errors.py
from typing import Optional, Any
from enum import Enum
from dataclasses import dataclass

class ErrorLevel(Enum):
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class SystemError:
    """시스템 에러 표준 구조"""
    code: int
    level: ErrorLevel
    message: str
    detail: Optional[str] = None
    context: Optional[dict] = None
    recoverable: bool = True
    
    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "level": self.level.value,
            "message": self.message,
            "detail": self.detail,
            "context": self.context,
            "recoverable": self.recoverable
        }

# 사용 예시
from core.errors import SystemError, ErrorLevel
from config.constants import ErrorCode

def process_chunk(text: str) -> Result[Chunk, SystemError]:
    try:
        chunk = chunker.create(text)
        return Ok(chunk)
    except ValueError as e:
        error = SystemError(
            code=ErrorCode.VALIDATION_ERROR,
            level=ErrorLevel.ERROR,
            message="Invalid chunk text",
            detail=str(e),
            context={"text_length": len(text)},
            recoverable=True
        )
        logger.error(error.to_dict())
        return Err(error)
```

### 2.4 단일 책임 원칙 (SRP)
**규칙**: 각 클래스/함수는 하나의 명확한 책임만 가짐

#### 적용 사례
```python
# 잘못된 설계 (여러 책임)
class PDFProcessor:
    def process(self, pdf_path):
        text = self.extract_text(pdf_path)  # 책임 1: 추출
        corrected = self.correct_ocr(text)  # 책임 2: 교정
        chunks = self.chunk_text(corrected)  # 책임 3: 청킹
        return chunks

# 올바른 설계 (단일 책임)
class PDFExtractor:
    """책임: PDF에서 텍스트 추출"""
    def extract(self, pdf_path: str) -> str: ...

class OCRCorrector:
    """책임: OCR 오류 교정"""
    def correct(self, text: str) -> str: ...

class TextChunker:
    """책임: 텍스트 청킹"""
    def chunk(self, text: str) -> List[Chunk]: ...

# 조합은 Facade 패턴으로
class PDFProcessingFacade:
    def __init__(self):
        self.extractor = PDFExtractor()
        self.corrector = OCRCorrector()
        self.chunker = TextChunker()
    
    def process(self, pdf_path: str) -> List[Chunk]:
        text = self.extractor.extract(pdf_path)
        corrected = self.corrector.correct(text)
        chunks = self.chunker.chunk(corrected)
        return chunks
```

### 2.5 기능별 폴더 정리
**규칙**: 기능 중심 구조, 계층은 최대 3단계

---

## 3. 프로젝트 구조

```
pdf-rag-system/
│
├─ config/                          # 설정 파일 (SSOT)
│  ├─ constants.yaml                # 상수 정의
│  ├─ options.yaml                  # 기능 옵션
│  ├─ thresholds.yaml               # 임계값
│  ├─ weights.yaml                  # 검색 가중치
│  ├─ domain_dictionary.json        # 도메인 사전
│  └─ loader.py                     # Config 로더
│
├─ core/                            # 핵심 추상화
│  ├─ __init__.py
│  ├─ types.py                      # 공통 타입 정의
│  ├─ errors.py                     # 에러 클래스
│  ├─ result.py                     # Result[T, E] 모나드
│  ├─ base.py                       # 추상 베이스 클래스
│  └─ validation.py                 # 검증 유틸
│
├─ chunking/                        # 청킹 시스템
│  ├─ __init__.py
│  ├─ base.py                       # BaseChunker 추상 클래스
│  ├─ standard.py                   # StandardChunker
│  ├─ wastewater.py                 # WastewaterChunker
│  ├─ numeric.py                    # NumericChunker
│  ├─ index.py                      # ChunkIndexBuilder
│  ├─ boundary.py                   # 경계 스냅 유틸
│  ├─ metadata.py                   # 메타데이터 관리
│  └─ tests/
│     ├─ test_standard.py
│     ├─ test_numeric.py
│     └─ test_boundary.py
│
├─ ocr/                             # OCR 처리
│  ├─ __init__.py
│  ├─ extractor.py                  # PDF 텍스트 추출
│  ├─ corrector.py                  # LLM 기반 교정
│  ├─ noise_scorer.py               # 노이즈 점수 계산
│  ├─ selector.py                   # 저품질 선별
│  └─ tests/
│     └─ test_corrector.py
│
├─ query/                           # 질문 처리
│  ├─ __init__.py
│  ├─ analyzer.py                   # 질문 분석
│  ├─ classifier.py                 # 질문 분류
│  ├─ tokenizer.py                  # 토큰화
│  └─ tests/
│     └─ test_analyzer.py
│
├─ retrieval/                       # 검색 시스템
│  ├─ __init__.py
│  ├─ hybrid.py                     # 하이브리드 검색
│  ├─ vector.py                     # 벡터 검색
│  ├─ bm25.py                       # BM25 검색
│  ├─ reranker.py                   # 재순위
│  └─ tests/
│     └─ test_hybrid.py
│
├─ filtering/                       # 필터링 시스템
│  ├─ __init__.py
│  ├─ pipeline.py                   # 필터 파이프라인
│  ├─ pre_filter.py                 # 사전 필터
│  ├─ confidence.py                 # 신뢰도 필터
│  ├─ diversity.py                  # 다양성 필터
│  ├─ scoring.py                    # 점수 계산 (Lazy)
│  └─ tests/
│     └─ test_pipeline.py
│
├─ context/                         # 컨텍스트 관리
│  ├─ __init__.py
│  ├─ manager.py                    # 동적 컨텍스트 관리
│  ├─ neighbor.py                   # 이웃 추가 로직
│  ├─ merger.py                     # 중복 제거
│  └─ tests/
│     └─ test_manager.py
│
├─ caching/                         # 캐싱 시스템
│  ├─ __init__.py
│  ├─ hub.py                        # CacheHub (중앙 관리)
│  ├─ strategies.py                 # LRU, FIFO, LFU
│  ├─ thread_safe.py                # 스레드 안전 래퍼
│  └─ tests/
│     └─ test_hub.py
│
├─ measurements/                    # 측정값 처리
│  ├─ __init__.py
│  ├─ extractor.py                  # 측정값 추출
│  ├─ converter.py                  # 단위 변환
│  ├─ validator.py                  # 수치 검증
│  └─ tests/
│     └─ test_extractor.py
│
├─ monitoring/                      # 모니터링
│  ├─ __init__.py
│  ├─ logger.py                     # 구조화된 로거
│  ├─ metrics.py                    # 성능 메트릭
│  ├─ profiler.py                   # 프로파일러
│  └─ dashboard.py                  # 실시간 대시보드
│
├─ utils/                           # 공통 유틸
│  ├─ __init__.py
│  ├─ text.py                       # 텍스트 처리
│  ├─ ngrams.py                     # N-gram 생성
│  ├─ serialization.py              # 직렬화
│  └─ timing.py                     # 타이밍 데코레이터
│
├─ api/                             # API 레이어
│  ├─ __init__.py
│  ├─ facade.py                     # 통합 Facade
│  ├─ handlers.py                   # 요청 핸들러
│  └─ schemas.py                    # 요청/응답 스키마
│
├─ tests/                           # 통합 테스트
│  ├─ integration/
│  ├─ benchmarks/
│  └─ fixtures/
│
├─ scripts/                         # 유틸리티 스크립트
│  ├─ build_index.py
│  ├─ validate_config.py
│  └─ benchmark.py
│
├─ docs/                            # 문서
│  ├─ architecture.md
│  ├─ api_reference.md
│  └─ development_guide.md
│
├─ pyproject.toml                   # 프로젝트 메타데이터
├─ requirements.txt
├─ pytest.ini
└─ README.md
```

### 3.1 폴더 구조 원칙

1. **기능별 분리**: 각 폴더는 하나의 기능 영역을 담당
2. **테스트 포함**: 각 기능 폴더 내부에 tests/ 서브폴더
3. **최대 3단계**: 폴더 깊이는 최대 3단계까지만
4. **명확한 명명**: 폴더명은 기능을 직접적으로 표현

---

## 4. 설정 관리 체계

### 4.1 설정 파일 구조

#### 4.1.1 constants.yaml
```yaml
# config/constants.yaml
# One Source of Truth for all constants

# Error codes
errors:
  SUCCESS: 0
  FAILURE: 1
  PARTIAL_SUCCESS: 2
  VALIDATION_ERROR: 3
  TIMEOUT_ERROR: 4
  CONFIG_ERROR: 5
  MODEL_ERROR: 6
  CACHE_ERROR: 7

# Chunk types
chunk_types:
  STANDARD: "standard"
  NUMERIC_EXPANDED: "numeric_expanded"
  WASTEWATER: "wastewater"
  HYBRID: "hybrid"

# Question types (simplified to 6 core types)
question_types:
  DEFINITION: "definition"
  NUMERIC: "numeric"
  PROCEDURE: "procedure"
  COMPARISON: "comparison"
  PROBLEM: "problem"
  GENERAL: "general"

# Filter stages
filter_stages:
  PRE_FILTER: 0
  RERANK: 1
  CONFIDENCE: 2
  DIVERSITY: 3

# Cache strategies
cache_strategies:
  LRU: "lru"
  FIFO: "fifo"
  LFU: "lfu"

# Error recovery actions
recovery_actions:
  RETRY: "retry"
  FALLBACK: "fallback"
  SKIP: "skip"
  ABORT: "abort"
```

#### 4.1.2 thresholds.yaml
```yaml
# config/thresholds.yaml
# All threshold values

chunking:
  chunk_size: 802
  chunk_overlap: 200
  snap_margin_ratio: 0.05  # ±5% window
  min_chunk_size: 50
  max_chunk_size: 2000
  
  wastewater:
    chunk_size: 900
    overlap_ratio: 0.25
  
  numeric:
    context_window: 3
    min_measurements: 1
    expansion_limit: 1500

ocr_correction:
  noise_threshold: 0.5
  max_chars_budget: 10000
  batch_size: 4
  timeout_seconds: 20

filtering:
  # Pre-filter
  min_overlap: 0.07
  keyword_filter_min: 1
  
  # Rerank
  rerank_threshold: 0.41
  
  # Confidence
  confidence_threshold: 0.20
  z_score_clip_min: -3.0
  z_score_clip_max: 3.0
  
  # Diversity
  max_same_page: 2
  max_same_section: 3

context:
  k_min: 4
  k_max: 10
  k_default: 6
  k_numeric: 8
  k_definition_base: 4
  
  neighbor:
    max_paragraph_neighbors: 1
    max_page_neighbors: 1
    allow_adjacent_page: true

performance:
  cache_size: 256
  max_parallel_workers: 4
  request_timeout: 30
```

#### 4.1.3 weights.yaml
```yaml
# config/weights.yaml
# Search weights for hybrid retrieval

# Default weights
default:
  vector: 0.58
  bm25: 0.42

# Question type specific weights
by_question_type:
  definition:
    vector: 0.58
    bm25: 0.42
  
  numeric:
    vector: 0.58
    bm25: 0.42
  
  procedure:
    vector: 0.70
    bm25: 0.30
  
  comparison:
    vector: 0.60
    bm25: 0.40
  
  problem:
    vector: 0.55
    bm25: 0.45
  
  general:
    vector: 0.58
    bm25: 0.42

# Scoring weights
scoring:
  vector_score: 0.4
  bm25_score: 0.3
  rerank_score: 0.3

# Normalization
normalization:
  method: "min_max"  # min_max | z_score | robust
  clip_outliers: true
```

#### 4.1.4 options.yaml
```yaml
# config/options.yaml
# Feature flags and options

features:
  enable_llm_correction: true
  enable_numeric_chunking: true
  enable_wastewater_mode: false
  enable_dynamic_k: true
  enable_neighbor_expansion: true
  enable_semantic_dedup: false
  enable_parallel_search: true

processing_modes:
  ocr_correction: "selective"  # selective | full | disabled
  chunk_strategy: "hybrid"     # standard | wastewater | hybrid | adaptive
  filter_pipeline: "full"      # minimal | standard | full
  cache_strategy: "lru"        # lru | fifo | lfu

optimization:
  use_cache: true
  use_index: true
  use_lazy_evaluation: true
  preload_models: true

logging:
  level: "INFO"  # DEBUG | INFO | WARNING | ERROR | CRITICAL
  format: "json"  # text | json
  include_trace: true
  log_performance: true

validation:
  strict_mode: false
  validate_numeric: true
  validate_ranges: true
  max_validation_errors: 10
```

### 4.2 Config 로더

```python
# config/loader.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from functools import lru_cache

from core.errors import SystemError, ErrorLevel
from config.constants import ErrorCode

@dataclass
class ThresholdConfig:
    """임계값 설정"""
    chunk_size: int
    chunk_overlap: int
    snap_margin_ratio: float
    min_overlap: float
    rerank_threshold: float
    confidence_threshold: float
    k_min: int
    k_max: int
    k_default: int
    k_numeric: int
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ThresholdConfig':
        """딕셔너리에서 생성"""
        return cls(
            chunk_size=data['chunking']['chunk_size'],
            chunk_overlap=data['chunking']['chunk_overlap'],
            snap_margin_ratio=data['chunking']['snap_margin_ratio'],
            min_overlap=data['filtering']['min_overlap'],
            rerank_threshold=data['filtering']['rerank_threshold'],
            confidence_threshold=data['filtering']['confidence_threshold'],
            k_min=data['context']['k_min'],
            k_max=data['context']['k_max'],
            k_default=data['context']['k_default'],
            k_numeric=data['context']['k_numeric'],
        )

@dataclass
class WeightConfig:
    """검색 가중치 설정"""
    default_vector: float
    default_bm25: float
    by_question_type: Dict[str, Dict[str, float]]
    
    def get_weights(self, question_type: str) -> tuple[float, float]:
        """질문 유형별 가중치 반환"""
        weights = self.by_question_type.get(
            question_type, 
            {"vector": self.default_vector, "bm25": self.default_bm25}
        )
        return weights['vector'], weights['bm25']

@dataclass
class OptionsConfig:
    """기능 옵션 설정"""
    features: Dict[str, bool]
    processing_modes: Dict[str, str]
    optimization: Dict[str, bool]
    logging: Dict[str, Any]
    validation: Dict[str, Any]

@dataclass
class SystemConfig:
    """전체 시스템 설정 통합"""
    constants: Dict[str, Any]
    thresholds: ThresholdConfig
    weights: WeightConfig
    options: OptionsConfig
    
    _instance: Optional['SystemConfig'] = None
    
    @classmethod
    def load(cls, config_dir: Path = Path("config")) -> 'SystemConfig':
        """설정 파일들을 로드하여 SystemConfig 생성"""
        try:
            # Constants
            constants_path = config_dir / "constants.yaml"
            with open(constants_path, 'r', encoding='utf-8') as f:
                constants = yaml.safe_load(f)
            
            # Thresholds
            thresholds_path = config_dir / "thresholds.yaml"
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                thresholds_data = yaml.safe_load(f)
            thresholds = ThresholdConfig.from_dict(thresholds_data)
            
            # Weights
            weights_path = config_dir / "weights.yaml"
            with open(weights_path, 'r', encoding='utf-8') as f:
                weights_data = yaml.safe_load(f)
            weights = WeightConfig(
                default_vector=weights_data['default']['vector'],
                default_bm25=weights_data['default']['bm25'],
                by_question_type=weights_data['by_question_type']
            )
            
            # Options
            options_path = config_dir / "options.yaml"
            with open(options_path, 'r', encoding='utf-8') as f:
                options_data = yaml.safe_load(f)
            options = OptionsConfig(**options_data)
            
            return cls(
                constants=constants,
                thresholds=thresholds,
                weights=weights,
                options=options
            )
            
        except FileNotFoundError as e:
            raise SystemError(
                code=ErrorCode.CONFIG_ERROR,
                level=ErrorLevel.CRITICAL,
                message=f"Config file not found: {e.filename}",
                recoverable=False
            )
        except yaml.YAMLError as e:
            raise SystemError(
                code=ErrorCode.CONFIG_ERROR,
                level=ErrorLevel.CRITICAL,
                message=f"Invalid YAML format: {str(e)}",
                recoverable=False
            )
    
    @classmethod
    def get_instance(cls) -> 'SystemConfig':
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance

# 전역 접근 함수
@lru_cache(maxsize=1)
def get_config() -> SystemConfig:
    """설정 인스턴스 가져오기 (캐싱)"""
    return SystemConfig.get_instance()
```

### 4.3 설정 검증

```python
# scripts/validate_config.py
from pathlib import Path
from typing import List, Tuple
import yaml

from config.loader import SystemConfig
from core.errors import SystemError

def validate_config(config_dir: Path = Path("config")) -> Tuple[bool, List[str]]:
    """설정 파일 검증"""
    errors = []
    
    # 1. 필수 파일 존재 확인
    required_files = [
        "constants.yaml",
        "thresholds.yaml",
        "weights.yaml",
        "options.yaml"
    ]
    
    for file in required_files:
        if not (config_dir / file).exists():
            errors.append(f"Missing required file: {file}")
    
    if errors:
        return False, errors
    
    # 2. YAML 형식 검증
    for file in required_files:
        try:
            with open(config_dir / file, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML in {file}: {str(e)}")
    
    # 3. 값 범위 검증
    try:
        config = SystemConfig.load(config_dir)
        
        # 임계값 범위 체크
        t = config.thresholds
        if not (0 < t.snap_margin_ratio < 0.2):
            errors.append("snap_margin_ratio must be in (0, 0.2)")
        
        if not (0 <= t.min_overlap <= 1):
            errors.append("min_overlap must be in [0, 1]")
        
        if not (t.k_min < t.k_max):
            errors.append("k_min must be less than k_max")
        
        # 가중치 합 체크
        for qtype, weights in config.weights.by_question_type.items():
            total = weights['vector'] + weights['bm25']
            if not (0.99 <= total <= 1.01):  # 부동소수점 오차 허용
                errors.append(
                    f"Weights for {qtype} must sum to 1.0 (got {total})"
                )
    
    except SystemError as e:
        errors.append(f"Config load error: {e.message}")
    
    return len(errors) == 0, errors

if __name__ == "__main__":
    success, errors = validate_config()
    if success:
        print("✓ All config files are valid")
    else:
        print("✗ Config validation failed:")
        for error in errors:
            print(f"  - {error}")
        exit(1)
```

---

## 5. 청킹 시스템 재설계

### 5.1 추상화 계층

```python
# chunking/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from core.types import Chunk
from core.result import Result
from core.errors import SystemError

@dataclass
class ChunkConfig:
    """청킹 설정 기본 클래스"""
    chunk_size: int
    chunk_overlap: int
    snap_margin_ratio: float = 0.05
    min_chunk_size: int = 50
    
    def validate(self) -> Result[None, SystemError]:
        """설정 검증"""
        if self.chunk_size <= 0:
            return Err(SystemError(
                code=ErrorCode.VALIDATION_ERROR,
                level=ErrorLevel.ERROR,
                message="chunk_size must be positive"
            ))
        if self.chunk_overlap >= self.chunk_size:
            return Err(SystemError(
                code=ErrorCode.VALIDATION_ERROR,
                level=ErrorLevel.ERROR,
                message="chunk_overlap must be less than chunk_size"
            ))
        return Ok(None)

class BaseChunker(ABC):
    """청킹 추상 베이스 클래스"""
    
    def __init__(self, config: ChunkConfig):
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """설정 검증"""
        result = self.config.validate()
        if result.is_err():
            raise result.unwrap_err()
    
    @abstractmethod
    def chunk(self, text: str, doc_id: str, filename: str) -> Result[List[Chunk], SystemError]:
        """텍스트를 청크로 분할 (구현 필수)"""
        pass
    
    def _calculate_step(self) -> int:
        """스텝 크기 계산 (공통)"""
        return max(1, self.config.chunk_size - self.config.chunk_overlap)
    
    def _snap_boundary(
        self, 
        text: str, 
        start: int, 
        end: int
    ) -> int:
        """경계 스냅 (공통 로직)"""
        from chunking.boundary import snap_to_word_boundary
        return snap_to_word_boundary(
            text, 
            start, 
            end, 
            self.config.snap_margin_ratio * self.config.chunk_size
        )
    
    def _create_chunk(
        self,
        text: str,
        doc_id: str,
        filename: str,
        start_offset: int,
        chunk_id: int,
        chunk_type: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """청크 객체 생성 (공통)"""
        return Chunk(
            doc_id=doc_id,
            filename=filename,
            chunk_id=chunk_id,
            start_offset=start_offset,
            length=len(text),
            text=text.strip(),
            chunk_type=chunk_type,
            extra=extra or {}
        )
```

### 5.2 경계 스냅 유틸

```python
# chunking/boundary.py
from typing import Set

# 경계 문자 (One Source of Truth)
BOUNDARY_CHARS: Set[str] = {" ", "\n", "\t", ".", "!", "?", "。", "！", "？"}

def snap_to_word_boundary(
    text: str,
    start: int,
    end: int,
    margin: float
) -> int:
    """
    경계를 단어 경계로 스냅
    
    Args:
        text: 전체 텍스트
        start: 시작 위치
        end: 종료 위치
        margin: 스냅 허용 범위 (문자 수)
    
    Returns:
        조정된 종료 위치
    """
    margin_int = int(margin)
    
    # 마진 범위 내에서 경계 문자 탐색
    search_start = max(start, end - margin_int)
    search_end = min(len(text), end + margin_int)
    
    # 뒤쪽 우선 탐색
    for i in range(end, search_end):
        if i < len(text) and text[i] in BOUNDARY_CHARS:
            return i
    
    # 앞쪽 탐색
    for i in range(end - 1, search_start - 1, -1):
        if i >= 0 and text[i] in BOUNDARY_CHARS:
            return i + 1
    
    # 경계를 찾지 못하면 원래 위치 반환
    return end

def find_sentence_boundaries(text: str) -> List[int]:
    """문장 경계 위치 찾기"""
    sentence_ends = {".", "!", "?", "。", "！", "？"}
    boundaries = [0]
    
    for i, char in enumerate(text):
        if char in sentence_ends:
            # 다음 문자가 공백이거나 끝이면 문장 경계
            if i + 1 >= len(text) or text[i + 1].isspace():
                boundaries.append(i + 1)
    
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    
    return boundaries

def estimate_snap_quality(text: str, pos: int, margin: int) -> float:
    """
    경계 스냅 품질 점수 계산
    
    Returns:
        0.0 (나쁨) ~ 1.0 (완벽)
    """
    if pos >= len(text):
        return 1.0
    
    # 경계 문자에 정확히 위치
    if text[pos] in BOUNDARY_CHARS:
        return 1.0
    
    # 마진 내 경계 문자까지 거리
    for i in range(1, margin + 1):
        if pos + i < len(text) and text[pos + i] in BOUNDARY_CHARS:
            return 1.0 - (i / margin) * 0.5
        if pos - i >= 0 and text[pos - i] in BOUNDARY_CHARS:
            return 1.0 - (i / margin) * 0.5
    
    return 0.0  # 마진 내 경계 없음
```

### 5.3 StandardChunker 구현

```python
# chunking/standard.py
from typing import List
from chunking.base import BaseChunker, ChunkConfig
from core.types import Chunk
from core.result import Result, Ok, Err
from core.errors import SystemError, ErrorLevel
from config.constants import ErrorCode, ChunkType

class StandardChunker(BaseChunker):
    """표준 슬라이딩 윈도우 청킹"""
    
    def chunk(
        self, 
        text: str, 
        doc_id: str, 
        filename: str
    ) -> Result[List[Chunk], SystemError]:
        """
        슬라이딩 윈도우 방식으로 청킹
        
        알고리즘:
        1. chunk_size - chunk_overlap = step 계산
        2. 0부터 step 간격으로 이동하며 청크 생성
        3. 각 청크 경계를 단어 경계로 스냅
        4. 최소 크기 미만 청크는 이전 청크에 병합
        """
        try:
            if not text:
                return Ok([])
            
            # 텍스트가 chunk_size보다 작으면 단일 청크
            if len(text) <= self.config.chunk_size:
                chunk = self._create_chunk(
                    text=text,
                    doc_id=doc_id,
                    filename=filename,
                    start_offset=0,
                    chunk_id=0,
                    chunk_type=ChunkType.STANDARD
                )
                return Ok([chunk])
            
            chunks = []
            step = self._calculate_step()
            chunk_id = 0
            position = 0
            
            while position < len(text):
                # 청크 범위 계산
                start = position
                end = min(len(text), position + self.config.chunk_size)
                
                # 경계 스냅
                if end < len(text):
                    end = self._snap_boundary(text, start, end)
                
                # 청크 생성
                chunk_text = text[start:end]
                
                # 최소 크기 체크
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunk = self._create_chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        filename=filename,
                        start_offset=start,
                        chunk_id=chunk_id,
                        chunk_type=ChunkType.STANDARD,
                        extra={
                            "boundary_snapped": end != position + self.config.chunk_size,
                            "is_final": end >= len(text)
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                elif chunks:
                    # 너무 작으면 이전 청크에 병합
                    last_chunk = chunks[-1]
                    merged_text = last_chunk.text + " " + chunk_text
                    chunks[-1] = self._create_chunk(
                        text=merged_text,
                        doc_id=doc_id,
                        filename=filename,
                        start_offset=last_chunk.start_offset,
                        chunk_id=last_chunk.chunk_id,
                        chunk_type=ChunkType.STANDARD,
                        extra={**last_chunk.extra, "merged": True}
                    )
                
                # 다음 위치로 이동
                if end >= len(text):
                    break
                position = start + step
            
            return Ok(chunks)
            
        except Exception as e:
            return Err(SystemError(
                code=ErrorCode.FAILURE,
                level=ErrorLevel.ERROR,
                message=f"Chunking failed: {str(e)}",
                context={"doc_id": doc_id, "text_length": len(text)}
            ))
```

### 5.4 NumericChunker 구현

```python
# chunking/numeric.py
from typing import List, Tuple
from dataclasses import dataclass
import bisect

from chunking.base import BaseChunker, ChunkConfig
from core.types import Chunk
from core.result import Result, Ok, Err
from core.errors import SystemError
from config.constants import ChunkType
from measurements.extractor import extract_measurements

@dataclass
class NumericChunkConfig(ChunkConfig):
    """숫자 중심 청킹 설정"""
    context_window: int = 3
    min_measurements: int = 1
    expansion_limit: int = 1500
    emphasis_repeat: int = 2

class NumericChunker(BaseChunker):
    """숫자 중심 확장 청킹 (O(N log N) 최적화)"""
    
    def __init__(self, config: NumericChunkConfig, base_chunks: List[Chunk]):
        super().__init__(config)
        self.config: NumericChunkConfig = config
        self.base_chunks = base_chunks
        self._build_index()
    
    def _build_index(self) -> None:
        """청크 인덱스 구축 (O(N log N))"""
        # 시작 위치 기준 정렬된 리스트
        self.sorted_chunks = sorted(
            self.base_chunks, 
            key=lambda c: c.start_offset
        )
        self.start_offsets = [c.start_offset for c in self.sorted_chunks]
        
        # numeric flag 미리 계산
        self.numeric_flags = []
        for chunk in self.sorted_chunks:
            measurements = extract_measurements(chunk.text)
            self.numeric_flags.append(len(measurements) >= self.config.min_measurements)
    
    def chunk(
        self, 
        text: str, 
        doc_id: str, 
        filename: str
    ) -> Result[List[Chunk], SystemError]:
        """
        숫자 포함 청크에 대해 문맥 확장
        
        최적화:
        1. 인덱스 구축: O(N log N)
        2. 이웃 탐색: O(log N) per chunk
        3. 총 복잡도: O(N log N)
        """
        try:
            expanded_chunks = []
            
            for i, chunk in enumerate(self.sorted_chunks):
                # 숫자 미포함 청크는 스킵
                if not self.numeric_flags[i]:
                    continue
                
                # 이웃 청크 찾기 (O(log N))
                neighbors = self._find_neighbors_binary(
                    chunk, 
                    self.config.context_window
                )
                
                # 텍스트 확장
                expanded_text = self._build_expanded_text(
                    chunk, 
                    neighbors
                )
                
                # 확장 청크 생성
                if expanded_text != chunk.text:
                    expanded_chunk = self._create_chunk(
                        text=expanded_text,
                        doc_id=doc_id,
                        filename=filename,
                        start_offset=chunk.start_offset,
                        chunk_id=len(expanded_chunks),
                        chunk_type=ChunkType.NUMERIC_EXPANDED,
                        extra={
                            "original_chunk_id": chunk.chunk_id,
                            "original_length": len(chunk.text),
                            "expanded_length": len(expanded_text),
                            "context_chunks": len(neighbors),
                            "measurements": extract_measurements(chunk.text)
                        }
                    )
                    expanded_chunks.append(expanded_chunk)
            
            return Ok(expanded_chunks)
            
        except Exception as e:
            return Err(SystemError(
                code=ErrorCode.FAILURE,
                level=ErrorLevel.ERROR,
                message=f"Numeric chunking failed: {str(e)}",
                context={"doc_id": doc_id}
            ))
    
    def _find_neighbors_binary(
        self, 
        target: Chunk, 
        window: int
    ) -> List[Chunk]:
        """
        이진 탐색으로 이웃 청크 찾기 (O(log N))
        
        Args:
            target: 대상 청크
            window: 탐색 범위 (청크 개수)
        
        Returns:
            이웃 청크 리스트
        """
        # 대상 청크의 인덱스 찾기
        idx = bisect.bisect_left(self.start_offsets, target.start_offset)
        
        # 앞뒤로 window 범위만큼 이웃 선택
        start_idx = max(0, idx - window)
        end_idx = min(len(self.sorted_chunks), idx + window + 1)
        
        neighbors = []
        for i in range(start_idx, end_idx):
            if i != idx:  # 자기 자신 제외
                neighbors.append(self.sorted_chunks[i])
        
        return neighbors
    
    def _build_expanded_text(
        self, 
        original: Chunk, 
        neighbors: List[Chunk]
    ) -> str:
        """문맥 확장 텍스트 생성"""
        parts = []
        
        # 앞쪽 문맥 (원본보다 앞에 있는 이웃)
        前_neighbors = [
            n for n in neighbors 
            if n.start_offset < original.start_offset
        ]
        for neighbor in sorted(前_neighbors, key=lambda n: n.start_offset)[-2:]:
            parts.append(neighbor.text[-200:])  # 뒤쪽 200자
        
        # 원본 텍스트
        parts.append(original.text)
        
        # 뒤쪽 문맥 (원본보다 뒤에 있는 이웃)
        後_neighbors = [
            n for n in neighbors 
            if n.start_offset > original.start_offset
        ]
        for neighbor in sorted(後_neighbors, key=lambda n: n.start_offset)[:2]:
            parts.append(neighbor.text[:200])  # 앞쪽 200자
        
        expanded = '\n'.join(parts)
        
        # 숫자 강조
        measurements = extract_measurements(original.text)
        if measurements:
            numeric_values = [num for num, unit in measurements]
            emphasis = ' ' + ' '.join(numeric_values) * self.config.emphasis_repeat
            expanded += emphasis
        
        # 확장 한도 체크
        if len(expanded) > self.config.expansion_limit:
            expanded = expanded[:self.config.expansion_limit]
        
        return expanded
```

### 5.5 ChunkIndexBuilder

```python
# chunking/index.py
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

from core.types import Chunk

@dataclass
class ChunkIndex:
    """청크 인덱스 구조"""
    by_document: Dict[str, List[Chunk]]
    by_offset: Dict[str, List[Tuple[int, Chunk]]]  # doc_id -> sorted offsets
    by_type: Dict[str, List[Chunk]]
    by_measurement: Dict[str, List[Chunk]]  # unit -> chunks
    
    def get_by_range(
        self, 
        doc_id: str, 
        start: int, 
        end: int
    ) -> List[Chunk]:
        """범위 내 청크 검색 (O(log N))"""
        import bisect
        
        if doc_id not in self.by_offset:
            return []
        
        offsets_chunks = self.by_offset[doc_id]
        offsets = [offset for offset, _ in offsets_chunks]
        
        # 이진 탐색으로 범위 찾기
        start_idx = bisect.bisect_left(offsets, start)
        end_idx = bisect.bisect_right(offsets, end)
        
        return [chunk for _, chunk in offsets_chunks[start_idx:end_idx]]

class ChunkIndexBuilder:
    """청크 인덱스 빌더"""
    
    @staticmethod
    def build(chunks: List[Chunk]) -> ChunkIndex:
        """청크 리스트에서 인덱스 구축"""
        by_document = defaultdict(list)
        by_offset = defaultdict(list)
        by_type = defaultdict(list)
        by_measurement = defaultdict(list)
        
        for chunk in chunks:
            # Document index
            by_document[chunk.doc_id].append(chunk)
            
            # Offset index (정렬 유지)
            by_offset[chunk.doc_id].append((chunk.start_offset, chunk))
            
            # Type index
            by_type[chunk.chunk_type].append(chunk)
            
            # Measurement index
            if 'measurements' in chunk.extra:
                for num, unit in chunk.extra['measurements']:
                    by_measurement[unit].append(chunk)
        
        # Offset 정렬
        for doc_id in by_offset:
            by_offset[doc_id].sort(key=lambda x: x[0])
        
        return ChunkIndex(
            by_document=dict(by_document),
            by_offset=dict(by_offset),
            by_type=dict(by_type),
            by_measurement=dict(by_measurement)
        )
```

---

## 6. 질문 분류 시스템

### 6.1 간소화된 분류 체계

```python
# query/classifier.py
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Set
import re

from config.loader import get_config
from config.constants import QuestionType

@dataclass
class QuestionFeatures:
    """질문 특징"""
    has_number: bool
    has_unit: bool
    has_domain_keyword: bool
    token_count: int
    key_token_count: int
    patterns_matched: Set[str]

@dataclass
class QuestionAnalysis:
    """질문 분석 결과"""
    question_type: str
    subtype: Optional[str]  # 세부 분류
    features: QuestionFeatures
    confidence: float  # 분류 신뢰도
    
    # 검색 파라미터
    vector_weight: float
    bm25_weight: float
    dynamic_k: int

class QuestionClassifier:
    """간소화된 질문 분류기 (6개 핵심 카테고리)"""
    
    def __init__(self):
        self.config = get_config()
        self._compile_patterns()
        self._load_domain_dict()
    
    def _compile_patterns(self) -> None:
        """정규식 패턴 컴파일 (캐싱)"""
        self.patterns = {
            'definition': re.compile(
                r'(정의|무엇|란|의미|개념|설명|이란)'
            ),
            'procedure': re.compile(
                r'(방법|절차|순서|어떻게|과정|단계)'
            ),
            'comparison': re.compile(
                r'(비교|vs|더|높|낮|차이|장단점|우열)'
            ),
            'problem': re.compile(
                r'(문제|오류|이상|고장|원인|해결|대응|대책)'
            ),
        }
    
    def _load_domain_dict(self) -> None:
        """도메인 사전 로드"""
        # 도메인 사전에서 단위, 키워드 로드
        from measurements.extractor import get_units
        self.units = get_units()
        self.domain_keywords = [
            "정수장", "공정", "수질", "처리", "탁도", "응집",
            "여과", "소독", "염소", "pH", "알칼리도"
        ]
    
    def classify(self, question: str) -> QuestionAnalysis:
        """
        질문 분류 (간소화된 6종)
        
        분류 우선순위:
        1. NUMERIC: 숫자/단위/도메인 키워드 포함
        2. DEFINITION: 정의 패턴 매칭
        3. PROCEDURE: 절차 패턴 매칭
        4. COMPARISON: 비교 패턴 매칭
        5. PROBLEM: 문제 패턴 매칭
        6. GENERAL: 기타
        """
        # 특징 추출
        features = self._extract_features(question)
        
        # 분류 로직
        qtype, subtype, confidence = self._determine_type(question, features)
        
        # 검색 가중치
        vector_weight, bm25_weight = self.config.weights.get_weights(qtype)
        
        # 동적 K
        dynamic_k = self._calculate_dynamic_k(qtype, features)
        
        return QuestionAnalysis(
            question_type=qtype,
            subtype=subtype,
            features=features,
            confidence=confidence,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            dynamic_k=dynamic_k
        )
    
    def _extract_features(self, question: str) -> QuestionFeatures:
        """질문 특징 추출"""
        # 숫자 존재 여부
        has_number = bool(re.search(r'\d', question))
        
        # 단위 존재 여부
        has_unit = any(unit in question.lower() for unit in self.units)
        
        # 도메인 키워드 존재 여부
        has_domain_keyword = any(
            kw in question for kw in self.domain_keywords
        )
        
        # 토큰 수 계산
        tokens = [t for t in question.split() if len(t) >= 2]
        token_count = len(tokens)
        
        # 핵심 토큰 수
        key_tokens = [
            t for t in tokens 
            if any(kw in t for kw in self.domain_keywords)
        ]
        key_token_count = len(key_tokens)
        
        # 매칭된 패턴
        patterns_matched = set()
        for name, pattern in self.patterns.items():
            if pattern.search(question):
                patterns_matched.add(name)
        
        return QuestionFeatures(
            has_number=has_number,
            has_unit=has_unit,
            has_domain_keyword=has_domain_keyword,
            token_count=token_count,
            key_token_count=key_token_count,
            patterns_matched=patterns_matched
        )
    
    def _determine_type(
        self, 
        question: str, 
        features: QuestionFeatures
    ) -> Tuple[str, Optional[str], float]:
        """
        질문 유형 결정
        
        Returns:
            (question_type, subtype, confidence)
        """
        # 1. NUMERIC 우선 판단
        numeric_score = (
            int(features.has_number) * 0.4 +
            int(features.has_unit) * 0.4 +
            int(features.has_domain_keyword) * 0.2
        )
        if numeric_score >= 0.5:
            return QuestionType.NUMERIC, None, numeric_score
        
        # 2. 패턴 기반 분류
        if 'definition' in features.patterns_matched:
            return QuestionType.DEFINITION, None, 0.8
        
        if 'procedure' in features.patterns_matched:
            # Subtype으로 세분화
            if '운영' in question or '제어' in question:
                subtype = "operational"
            elif '시스템' in question or '접속' in question:
                subtype = "system_info"
            else:
                subtype = None
            return QuestionType.PROCEDURE, subtype, 0.8
        
        if 'comparison' in features.patterns_matched:
            return QuestionType.COMPARISON, None, 0.8
        
        if 'problem' in features.patterns_matched:
            return QuestionType.PROBLEM, None, 0.8
        
        # 3. GENERAL
        return QuestionType.GENERAL, None, 0.5
    
    def _calculate_dynamic_k(
        self, 
        qtype: str, 
        features: QuestionFeatures
    ) -> int:
        """동적 K 계산"""
        thresholds = self.config.thresholds
        
        # 기본 K 값
        if qtype == QuestionType.NUMERIC:
            base_k = thresholds.k_numeric
        elif qtype == QuestionType.DEFINITION:
            # 질문 길이에 비례
            base_k = max(4, min(6, features.token_count // 3 + 4))
        else:
            base_k = thresholds.k_default
        
        # 복잡도 보너스
        complexity_bonus = (
            features.token_count // 10 +
            features.key_token_count // 6
        )
        
        # 최종 K (범위 제한)
        final_k = max(
            thresholds.k_min,
            min(thresholds.k_max, base_k + complexity_bonus)
        )
        
        return int(final_k)
```

### 6.2 질문 분석기

```python
# query/analyzer.py
from dataclasses import dataclass
from typing import List

from query.classifier import QuestionClassifier, QuestionAnalysis
from query.tokenizer import Tokenizer
from core.result import Result, Ok, Err
from core.errors import SystemError

@dataclass
class EnrichedAnalysis(QuestionAnalysis):
    """확장된 분석 결과"""
    tokens: List[str]
    key_tokens: List[str]
    normalized_question: str

class QuestionAnalyzer:
    """질문 분석 통합 클래스"""
    
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.tokenizer = Tokenizer()
    
    def analyze(self, question: str) -> Result[EnrichedAnalysis, SystemError]:
        """질문 전체 분석"""
        try:
            # 정규화
            normalized = self.tokenizer.normalize(question)
            
            # 토큰화
            tokens = self.tokenizer.tokenize(normalized)
            key_tokens = self.tokenizer.extract_key_tokens(tokens)
            
            # 분류
            analysis = self.classifier.classify(normalized)
            
            # 결과 통합
            enriched = EnrichedAnalysis(
                **analysis.__dict__,
                tokens=tokens,
                key_tokens=key_tokens,
                normalized_question=normalized
            )
            
            return Ok(enriched)
            
        except Exception as e:
            return Err(SystemError(
                code=ErrorCode.FAILURE,
                level=ErrorLevel.ERROR,
                message=f"Question analysis failed: {str(e)}",
                context={"question": question}
            ))
```

---

## 7. 검색 및 필터링 파이프라인

### 7.1 하이브리드 검색

```python
# retrieval/hybrid.py
from typing import List, Tuple
from dataclasses import dataclass
import concurrent.futures

from retrieval.vector import VectorSearchEngine
from retrieval.bm25 import BM25SearchEngine
from core.types import RetrievedSpan
from core.result import Result, Ok, Err
from core.errors import SystemError
from query.analyzer import EnrichedAnalysis

@dataclass
class SearchResult:
    """검색 결과"""
    spans: List[RetrievedSpan]
    vector_time_ms: float
    bm25_time_ms: float
    merge_time_ms: float
    total_candidates: int

class HybridSearchEngine:
    """하이브리드 검색 엔진 (Vector + BM25)"""
    
    def __init__(self):
        self.vector_engine = VectorSearchEngine()
        self.bm25_engine = BM25SearchEngine()
        self.config = get_config()
    
    def search(
        self, 
        query: str, 
        analysis: EnrichedAnalysis,
        k: int
    ) -> Result[SearchResult, SystemError]:
        """
        하이브리드 검색 실행
        
        병렬 처리:
        - Vector Search와 BM25를 동시 실행
        - RRF (Reciprocal Rank Fusion)으로 병합
        """
        try:
            if self.config.options.optimization['use_parallel']:
                return self._search_parallel(query, analysis, k)
            else:
                return self._search_sequential(query, analysis, k)
        
        except Exception as e:
            return Err(SystemError(
                code=ErrorCode.FAILURE,
                level=ErrorLevel.ERROR,
                message=f"Hybrid search failed: {str(e)}",
                context={"query": query, "k": k}
            ))
    
    def _search_parallel(
        self, 
        query: str, 
        analysis: EnrichedAnalysis, 
        k: int
    ) -> Result[SearchResult, SystemError]:
        """병렬 검색"""
        import time
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Vector 검색 시작
            future_vector = executor.submit(
                self.vector_engine.search,
                query,
                k * 2  # 후보는 2배로
            )
            
            # BM25 검색 시작
            future_bm25 = executor.submit(
                self.bm25_engine.search,
                query,
                k * 2
            )
            
            # 결과 수집
            start_merge = time.time()
            vector_result = future_vector.result()
            bm25_result = future_bm25.result()
            
            # RRF 병합
            merged_spans = self._rrf_merge(
                vector_result.spans,
                bm25_result.spans,
                analysis.vector_weight,
                analysis.bm25_weight,
                k
            )
            merge_time_ms = (time.time() - start_merge) * 1000
            
            return Ok(SearchResult(
                spans=merged_spans,
                vector_time_ms=vector_result.time_ms,
                bm25_time_ms=bm25_result.time_ms,
                merge_time_ms=merge_time_ms,
                total_candidates=len(vector_result.spans) + len(bm25_result.spans)
            ))
    
    def _rrf_merge(
        self,
        vector_spans: List[RetrievedSpan],
        bm25_spans: List[RetrievedSpan],
        vector_weight: float,
        bm25_weight: float,
        k: int
    ) -> List[RetrievedSpan]:
        """
        Reciprocal Rank Fusion 병합
        
        RRF 공식:
        score(d) = Σ weight / (rank + 60)
        """
        K_CONST = 60
        span_scores = {}
        
        # Vector 점수
        for rank, span in enumerate(vector_spans):
            key = self._span_key(span)
            rrf_score = vector_weight / (rank + K_CONST)
            span_scores[key] = span_scores.get(key, 0) + rrf_score
            
            # aux_scores에 기록
            if key not in span_scores:
                span.aux_scores['vector'] = span.score
                span.aux_scores['vector_rank'] = rank
        
        # BM25 점수
        for rank, span in enumerate(bm25_spans):
            key = self._span_key(span)
            rrf_score = bm25_weight / (rank + K_CONST)
            span_scores[key] = span_scores.get(key, 0) + rrf_score
            
            span.aux_scores['bm25'] = span.score
            span.aux_scores['bm25_rank'] = rank
        
        # 통합 딕셔너리 생성
        all_spans = {}
        for span in vector_spans + bm25_spans:
            key = self._span_key(span)
            if key not in all_spans:
                all_spans[key] = span
                all_spans[key].score = span_scores[key]
        
        # 점수 정렬
        sorted_spans = sorted(
            all_spans.values(),
            key=lambda s: s.score,
            reverse=True
        )
        
        return sorted_spans[:k]
    
    @staticmethod
    def _span_key(span: RetrievedSpan) -> str:
        """Span 고유 키 생성"""
        return f"{span.doc_id}|{span.page}|{span.start_offset}"
```

### 7.2 필터링 파이프라인

```python
# filtering/pipeline.py
from typing import List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

from core.types import RetrievedSpan
from core.result import Result, Ok
from core.errors import SystemError
from filtering.pre_filter import PreFilter
from filtering.confidence import ConfidenceFilter
from filtering.diversity import DiversityFilter
from filtering.scoring import ScoringEngine
from config.constants import FilterStage

@dataclass
class FilterConfig:
    """필터 파이프라인 설정"""
    enable_pre_filter: bool = True
    enable_rerank: bool = True
    enable_confidence: bool = True
    enable_diversity: bool = True
    
    # 동적 파이프라인
    min_spans_threshold: int = 5  # 이하면 필터 완화

class FilterPipeline:
    """동적 필터링 파이프라인"""
    
    def __init__(self):
        self.config = get_config()
        self.filter_config = FilterConfig()
        
        # 필터 인스턴스
        self.pre_filter = PreFilter()
        self.confidence_filter = ConfidenceFilter()
        self.diversity_filter = DiversityFilter()
        self.scoring_engine = ScoringEngine()
    
    def filter(
        self,
        spans: List[RetrievedSpan],
        question: str,
        reranker: Optional[Any] = None
    ) -> Result[List[RetrievedSpan], SystemError]:
        """
        다단계 필터링 실행
        
        동적 파이프라인:
        - spans 개수가 적으면 자동으로 일부 필터 비활성화
        - confidence 너무 낮으면 rerank 재평가
        """
        try:
            current_spans = spans
            stage_results = {}
            
            # Stage 0: Pre-filter
            if self.filter_config.enable_pre_filter:
                result = self.pre_filter.filter(current_spans, question)
                if result.is_ok():
                    current_spans = result.unwrap()
                    stage_results[FilterStage.PRE_FILTER] = len(current_spans)
            
            # Stage 1: Rerank (if enabled and reranker provided)
            if self.filter_config.enable_rerank and reranker:
                result = self._apply_rerank(current_spans, question, reranker)
                if result.is_ok():
                    current_spans = result.unwrap()
                    stage_results[FilterStage.RERANK] = len(current_spans)
            
            # 동적 조정: 너무 적으면 필터 완화
            if len(current_spans) < self.filter_config.min_spans_threshold:
                # Confidence 필터만 적용
                if self.filter_config.enable_confidence:
                    result = self.confidence_filter.filter(
                        current_spans,
                        relaxed=True  # 완화 모드
                    )
                    if result.is_ok():
                        current_spans = result.unwrap()
            else:
                # Stage 2: Confidence filter
                if self.filter_config.enable_confidence:
                    result = self.confidence_filter.filter(current_spans)
                    if result.is_ok():
                        current_spans = result.unwrap()
                        stage_results[FilterStage.CONFIDENCE] = len(current_spans)
                        
                        # Confidence 재평가 로직
                        avg_conf = sum(s.calibrated_conf for s in current_spans) / len(current_spans)
                        if avg_conf < 0.3 and reranker:
                            # Rerank 재평가
                            result = self._apply_rerank(current_spans, question, reranker)
                            if result.is_ok():
                                current_spans = result.unwrap()
                
                # Stage 3: Diversity filter
                if self.filter_config.enable_diversity:
                    result = self.diversity_filter.filter(current_spans)
                    if result.is_ok():
                        current_spans = result.unwrap()
                        stage_results[FilterStage.DIVERSITY] = len(current_spans)
            
            # 로깅
            self._log_filter_stages(stage_results, len(spans), len(current_spans))
            
            return Ok(current_spans)
        
        except Exception as e:
            return Err(SystemError(
                code=ErrorCode.FAILURE,
                level=ErrorLevel.ERROR,
                message=f"Filter pipeline failed: {str(e)}"
            ))
    
    def _apply_rerank(
        self,
        spans: List[RetrievedSpan],
        question: str,
        reranker: Any
    ) -> Result[List[RetrievedSpan], SystemError]:
        """Rerank 적용"""
        # Rerank 점수 계산 (lazy evaluation)
        for span in spans:
            if 'rerank' not in span.aux_scores:
                score = reranker.score(question, span.text)
                span.aux_scores['rerank'] = score
        
        # 정규화 및 필터링
        return self.scoring_engine.apply_rerank_threshold(spans)
    
    def _log_filter_stages(
        self,
        stage_results: dict,
        initial: int,
        final: int
    ) -> None:
        """필터 단계별 결과 로깅"""
        from monitoring.logger import get_logger
        logger = get_logger()
        
        logger.info({
            "event": "filter_pipeline_complete",
            "initial_spans": initial,
            "final_spans": final,
            "reduction_rate": (initial - final) / initial if initial > 0 else 0,
            "stage_results": stage_results
        })
```

### 7.3 Lazy Scoring Engine

```python
# filtering/scoring.py
from typing import List, Dict, Optional
import statistics

from core.types import RetrievedSpan
from core.result import Result, Ok
from config.loader import get_config

class ScoringEngine:
    """
    점수 계산 엔진 (Lazy Evaluation)
    
    특징:
    - 한 번 계산된 점수는 aux_scores에 캐싱
    - 필요할 때만 계산 (on-demand)
    """
    
    def __init__(self):
        self.config = get_config()
    
    def apply_rerank_threshold(
        self,
        spans: List[RetrievedSpan]
    ) -> Result[List[RetrievedSpan], SystemError]:
        """Rerank 임계값 적용"""
        # Rerank 점수 정규화 (lazy)
        for span in spans:
            if 'rerank_norm' not in span.aux_scores:
                span.aux_scores['rerank_norm'] = self._normalize_rerank(span, spans)
        
        # 임계값 필터링
        threshold = self.config.thresholds.rerank_threshold
        filtered = [
            s for s in spans
            if s.aux_scores['rerank_norm'] >= threshold
        ]
        
        return Ok(filtered)
    
    def calculate_confidence(
        self,
        spans: List[RetrievedSpan],
        use_robust: bool = True
    ) -> Result[List[RetrievedSpan], SystemError]:
        """
        신뢰도 계산 (Robust Z-score)
        
        개선점:
        - 기존 Z-score 대신 Robust Z-score 사용
        - Median + MAD (Median Absolute Deviation)
        """
        # 출처별 정규화 (lazy)
        per_source_scores = self._collect_source_scores(spans)
        
        # 정규화
        for span in spans:
            if 'confidence' not in span.aux_scores:
                norm_scores = []
                for source, score in span.aux_scores.items():
                    if source in per_source_scores:
                        normalized = self._normalize_source_score(
                            score,
                            per_source_scores[source]
                        )
                        norm_scores.append(normalized)
                
                # 평균 정규화 점수
                avg_norm = statistics.mean(norm_scores) if norm_scores else 0
                span.aux_scores['avg_norm'] = avg_norm
        
        # Z-score 계산
        all_norms = [s.aux_scores['avg_norm'] for s in spans]
        
        if use_robust:
            # Robust Z-score
            median = statistics.median(all_norms)
            mad = statistics.median([abs(x - median) for x in all_norms])
            
            for span in spans:
                if mad > 0:
                    z = 0.6745 * (span.aux_scores['avg_norm'] - median) / mad
                else:
                    z = 0
                
                # 클리핑 및 정규화
                z_clipped = max(-3, min(3, z))
                confidence = (z_clipped + 3) / 6
                span.calibrated_conf = confidence
        else:
            # 기존 Z-score
            mu = statistics.mean(all_norms)
            sigma = statistics.pstdev(all_norms)
            
            for span in spans:
                if sigma > 0:
                    z = (span.aux_scores['avg_norm'] - mu) / sigma
                else:
                    z = 0
                
                z_clipped = max(-3, min(3, z))
                confidence = (z_clipped + 3) / 6
                span.calibrated_conf = confidence
        
        return Ok(spans)
    
    def _normalize_rerank(
        self,
        span: RetrievedSpan,
        all_spans: List[RetrievedSpan]
    ) -> float:
        """Rerank 점수 정규화 (Min-Max)"""
        scores = [s.aux_scores.get('rerank', 0) for s in all_spans]
        vmin, vmax = min(scores), max(scores)
        
        if vmax - vmin < 1e-9:
            return 0.5
        
        score = span.aux_scores.get('rerank', 0)
        return (score - vmin) / (vmax - vmin)
    
    def _collect_source_scores(
        self,
        spans: List[RetrievedSpan]
    ) -> Dict[str, List[float]]:
        """출처별 점수 수집"""
        from collections import defaultdict
        per_source = defaultdict(list)
        
        for span in spans:
            for source, score in span.aux_scores.items():
                if isinstance(score, (int, float)):
                    per_source[source].append(score)
        
        return dict(per_source)
    
    def _normalize_source_score(
        self,
        score: float,
        source_scores: List[float]
    ) -> float:
        """출처별 점수 정규화"""
        vmin = min(source_scores)
        vmax = max(source_scores)
        
        if vmax - vmin < 1e-9:
            return 0.5
        
        return (score - vmin) / (vmax - vmin)
```

---

## 8. 캐싱 시스템

### 8.1 CacheHub (중앙 캐시 관리)

```python
# caching/hub.py
from typing import Any, Optional, Dict, Callable
from dataclasses import dataclass, field
from threading import RLock
from enum import Enum

from caching.strategies import LRUCache, FIFOCache, LFUCache
from config.loader import get_config
from config.constants import CacheStrategy

@dataclass
class CacheStats:
    """캐시 통계"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class CacheHub:
    """
    중앙 캐시 관리 허브
    
    특징:
    - 여러 캐시 통합 관리
    - Thread-safe
    - 전략별 캐시 생성
    - 통계 수집
    """
    
    _instance: Optional['CacheHub'] = None
    _lock = RLock()
    
    def __init__(self):
        self.config = get_config()
        self.caches: Dict[str, Any] = {}
        self.stats: Dict[str, CacheStats] = {}
        self._global_lock = RLock()
    
    @classmethod
    def get_instance(cls) -> 'CacheHub':
        """싱글톤 인스턴스"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_cache(
        self,
        name: str,
        max_size: int = 256,
        strategy: str = "lru"
    ) -> Any:
        """
        캐시 가져오기 (없으면 생성)
        
        Args:
            name: 캐시 이름
            max_size: 최대 크기
            strategy: 캐시 전략 (lru, fifo, lfu)
        """
        with self._global_lock:
            if name not in self.caches:
                # 전략별 캐시 생성
                if strategy == CacheStrategy.LRU:
                    cache = LRUCache(max_size)
                elif strategy == CacheStrategy.FIFO:
                    cache = FIFOCache(max_size)
                elif strategy == CacheStrategy.LFU:
                    cache = LFUCache(max_size)
                else:
                    cache = LRUCache(max_size)
                
                self.caches[name] = cache
                self.stats[name] = CacheStats(max_size=max_size)
            
            return self.caches[name]
    
    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        with self._global_lock:
            if cache_name not in self.caches:
                return None
            
            cache = self.caches[cache_name]
            value = cache.get(key)
            
            # 통계 업데이트
            if value is not None:
                self.stats[cache_name].hits += 1
            else:
                self.stats[cache_name].misses += 1
            
            return value
    
    def put(self, cache_name: str, key: str, value: Any) -> None:
        """캐시에 값 저장"""
        with self._global_lock:
            if cache_name not in self.caches:
                # 기본 캐시 생성
                self.get_cache(cache_name)
            
            cache = self.caches[cache_name]
            evicted = cache.put(key, value)
            
            # 통계 업데이트
            if evicted:
                self.stats[cache_name].evictions += 1
            self.stats[cache_name].size = len(cache)
    
    def get_stats(self, cache_name: Optional[str] = None) -> Dict[str, CacheStats]:
        """캐시 통계 조회"""
        with self._global_lock:
            if cache_name:
                return {cache_name: self.stats.get(cache_name, CacheStats())}
            return self.stats.copy()
    
    def clear(self, cache_name: Optional[str] = None) -> None:
        """캐시 클리어"""
        with self._global_lock:
            if cache_name:
                if cache_name in self.caches:
                    self.caches[cache_name].clear()
                    self.stats[cache_name] = CacheStats(
                        max_size=self.stats[cache_name].max_size
                    )
            else:
                for cache in self.caches.values():
                    cache.clear()
                self.stats = {
                    name: CacheStats(max_size=stats.max_size)
                    for name, stats in self.stats.items()
                }
    
    def get_memory_usage(self) -> Dict[str, int]:
        """메모리 사용량 추정"""
        import sys
        usage = {}
        
        with self._global_lock:
            for name, cache in self.caches.items():
                size = sys.getsizeof(cache)
                # 대략적인 크기 추정
                usage[name] = size
        
        return usage

# 전역 접근 함수
def get_cache_hub() -> CacheHub:
    """캐시 허브 인스턴스 가져오기"""
    return CacheHub.get_instance()
```

### 8.2 캐시 전략 구현

```python
# caching/strategies.py
from typing import Any, Optional, Dict, OrderedDict
from collections import OrderedDict as OD, defaultdict
from threading import RLock

class LRUCache:
    """LRU (Least Recently Used) 캐시"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OD = OD()
        self.lock = RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """값 가져오기 (최근 사용으로 이동)"""
        with self.lock:
            if key in self.cache:
                # 최근 사용으로 이동
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """값 저장 (eviction 발생 시 True 반환)"""
        with self.lock:
            evicted = False
            
            if key in self.cache:
                # 기존 키 업데이트
                self.cache.move_to_end(key)
            else:
                # 새 키 추가
                if len(self.cache) >= self.max_size:
                    # 가장 오래된 항목 제거
                    self.cache.popitem(last=False)
                    evicted = True
            
            self.cache[key] = value
            return evicted
    
    def clear(self) -> None:
        """캐시 클리어"""
        with self.lock:
            self.cache.clear()
    
    def __len__(self) -> int:
        return len(self.cache)

class FIFOCache:
    """FIFO (First In First Out) 캐시"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.order: list = []
        self.lock = RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """값 가져오기"""
        with self.lock:
            return self.cache.get(key)
    
    def put(self, key: str, value: Any) -> bool:
        """값 저장"""
        with self.lock:
            evicted = False
            
            if key not in self.cache:
                if len(self.cache) >= self.max_size:
                    # 가장 먼저 들어온 항목 제거
                    oldest = self.order.pop(0)
                    del self.cache[oldest]
                    evicted = True
                
                self.order.append(key)
            
            self.cache[key] = value
            return evicted
    
    def clear(self) -> None:
        """캐시 클리어"""
        with self.lock:
            self.cache.clear()
            self.order.clear()
    
    def __len__(self) -> int:
        return len(self.cache)

class LFUCache:
    """LFU (Least Frequently Used) 캐시"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.freq: Dict[str, int] = defaultdict(int)
        self.lock = RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """값 가져오기 (빈도 증가)"""
        with self.lock:
            if key in self.cache:
                self.freq[key] += 1
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """값 저장"""
        with self.lock:
            evicted = False
            
            if key not in self.cache:
                if len(self.cache) >= self.max_size:
                    # 빈도가 가장 낮은 항목 제거
                    lfu_key = min(self.freq, key=self.freq.get)
                    del self.cache[lfu_key]
                    del self.freq[lfu_key]
                    evicted = True
                
                self.freq[key] = 1
            else:
                self.freq[key] += 1
            
            self.cache[key] = value
            return evicted
    
    def clear(self) -> None:
        """캐시 클리어"""
        with self.lock:
            self.cache.clear()
            self.freq.clear()
    
    def __len__(self) -> int:
        return len(self.cache)
```

### 8.3 캐시 사용 예시

```python
# Example usage
from caching.hub import get_cache_hub

# 도메인 사전 캐싱
cache_hub = get_cache_hub()
domain_cache = cache_hub.get_cache("domain_dict", max_size=10, strategy="lru")

def load_domain_dict(path: str) -> dict:
    """도메인 사전 로드 (캐싱)"""
    # 캐시 확인
    cache_key = f"{path}_{os.path.getmtime(path)}"
    cached = cache_hub.get("domain_dict", cache_key)
    
    if cached is not None:
        return cached
    
    # 파일 로드
    with open(path, 'r') as f:
        data = json.load(f)
    
    # 캐시 저장
    cache_hub.put("domain_dict", cache_key, data)
    
    return data

# 검색 결과 캐싱
search_cache = cache_hub.get_cache("search_results", max_size=256, strategy="lru")

def search_with_cache(query: str, k: int) -> List[RetrievedSpan]:
    """검색 with 캐싱"""
    cache_key = f"{query}|{k}"
    cached = cache_hub.get("search_results", cache_key)
    
    if cached is not None:
        return cached
    
    # 실제 검색
    results = hybrid_search(query, k)
    
    # 캐시 저장
    cache_hub.put("search_results", cache_key, results)
    
    return results

# 캐시 통계 조회
stats = cache_hub.get_stats()
for name, stat in stats.items():
    print(f"{name}: hit_rate={stat.hit_rate:.2%}, size={stat.size}/{stat.max_size}")
```

---

## 9. 에러 처리 전략

### 9.1 계층별 에러 처리

```python
# core/errors.py (확장)
from typing import Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import traceback

T = TypeVar('T')
E = TypeVar('E')

@dataclass
class ErrorContext:
    """에러 컨텍스트"""
    timestamp: datetime = field(default_factory=datetime.now)
    trace: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def add_trace(self) -> 'ErrorContext':
        """스택 트레이스 추가"""
        self.trace = traceback.format_exc()
        return self

@dataclass
class SystemError:
    """확장된 시스템 에러"""
    code: int
    level: ErrorLevel
    message: str
    detail: Optional[str] = None
    context: ErrorContext = field(default_factory=ErrorContext)
    recoverable: bool = True
    recovery_action: Optional[str] = None  # retry, fallback, skip, abort
    
    def with_recovery(self, action: str) -> 'SystemError':
        """복구 액션 설정"""
        self.recovery_action = action
        return self
    
    def to_dict(self) -> dict:
        """딕셔너리 변환"""
        return {
            "code": self.code,
            "level": self.level.value,
            "message": self.message,
            "detail": self.detail,
            "recoverable": self.recoverable,
            "recovery_action": self.recovery_action,
            "timestamp": self.context.timestamp.isoformat(),
            "metadata": self.context.metadata
        }

class ErrorHandler:
    """중앙 에러 핸들러"""
    
    def __init__(self):
        from monitoring.logger import get_logger
        self.logger = get_logger()
        self.error_counts = defaultdict(int)
    
    def handle(
        self,
        error: SystemError,
        context: Optional[dict] = None
    ) -> None:
        """
        에러 처리
        
        1. 로깅
        2. 통계 수집
        3. 알림 (Critical인 경우)
        """
        # 컨텍스트 추가
        if context:
            error.context.metadata.update(context)
        
        # 로깅
        log_data = error.to_dict()
        
        if error.level == ErrorLevel.CRITICAL:
            self.logger.critical(log_data)
            self._send_alert(error)
        elif error.level == ErrorLevel.ERROR:
            self.logger.error(log_data)
        elif error.level == ErrorLevel.WARNING:
            self.logger.warning(log_data)
        else:
            self.logger.info(log_data)
        
        # 통계
        self.error_counts[error.code] += 1
    
    def _send_alert(self, error: SystemError) -> None:
        """Critical 에러 알림"""
        # Slack, Email 등으로 알림 전송
        pass
    
    def get_error_stats(self) -> dict:
        """에러 통계 반환"""
        return dict(self.error_counts)

# 전역 에러 핸들러
_error_handler = ErrorHandler()

def handle_error(error: SystemError, context: Optional[dict] = None) -> None:
    """에러 처리"""
    _error_handler.handle(error, context)
```

### 9.2 복구 전략

```python
# core/recovery.py
from typing import Callable, TypeVar, Optional
from functools import wraps
import time

from core.errors import SystemError, ErrorLevel
from core.result import Result, Ok, Err
from config.constants import ErrorCode, RecoveryAction

T = TypeVar('T')

def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """재시도 데코레이터"""
    def decorator(func: Callable[..., Result[T, SystemError]]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Result[T, SystemError]:
            last_error = None
            
            for attempt in range(max_attempts):
                result = func(*args, **kwargs)
                
                if result.is_ok():
                    return result
                
                error = result.unwrap_err()
                last_error = error
                
                # 복구 불가능하면 즉시 반환
                if not error.recoverable:
                    return result
                
                # 재시도 가능하면 대기
                if error.recovery_action == RecoveryAction.RETRY:
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        time.sleep(wait_time)
                        continue
                else:
                    # 다른 복구 액션이면 재시도 중단
                    return result
            
            # 모든 시도 실패
            return Err(last_error)
        
        return wrapper
    return decorator

def with_fallback(fallback_func: Callable[..., T]):
    """폴백 데코레이터"""
    def decorator(func: Callable[..., Result[T, SystemError]]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Result[T, SystemError]:
            result = func(*args, **kwargs)
            
            if result.is_err():
                error = result.unwrap_err()
                
                if error.recovery_action == RecoveryAction.FALLBACK:
                    try:
                        fallback_value = fallback_func(*args, **kwargs)
                        return Ok(fallback_value)
                    except Exception as e:
                        # Fallback도 실패
                        error.detail = f"Fallback failed: {str(e)}"
                        return Err(error)
            
            return result
        
        return wrapper
    return decorator

# 사용 예시
@with_retry(max_attempts=3, delay=1.0)
def call_llm_api(prompt: str) -> Result[str, SystemError]:
    """LLM API 호출 (재시도)"""
    try:
        response = ollama_generate(prompt)
        return Ok(response)
    except TimeoutError:
        return Err(SystemError(
            code=ErrorCode.TIMEOUT_ERROR,
            level=ErrorLevel.ERROR,
            message="LLM API timeout",
            recoverable=True
        ).with_recovery(RecoveryAction.RETRY))
    except Exception as e:
        return Err(SystemError(
            code=ErrorCode.FAILURE,
            level=ErrorLevel.ERROR,
            message=f"LLM API failed: {str(e)}",
            recoverable=False
        ))

@with_fallback(lambda text: text)  # 실패 시 원본 반환
def correct_text_with_llm(text: str) -> Result[str, SystemError]:
    """LLM 텍스트 교정 (폴백)"""
    result = call_llm_api(text)
    
    if result.is_err():
        error = result.unwrap_err()
        error.recovery_action = RecoveryAction.FALLBACK
        return Err(error)
    
    return result
```

---

## 10. 메타데이터 최적화

### 10.1 경량화 구조

```python
# chunking/metadata.py
from dataclasses import dataclass
from typing import Optional, List, Tuple
from weakref import WeakValueDictionary

@dataclass(frozen=True, slots=True)
class DocumentMeta:
    """문서 메타데이터 (공유)"""
    doc_id: str
    filename: str
    total_pages: int
    doc_type: str
    
    def __hash__(self):
        return hash(self.doc_id)

@dataclass(frozen=True, slots=True)
class ChunkMetaLight:
    """
    경량화된 청크 메타데이터
    
    특징:
    - __slots__로 메모리 최적화
    - 문서 메타는 참조로 관리
    - 불변 객체 (frozen=True)
    """
    chunk_id: int
    start_offset: int
    length: int
    chunk_type: str
    page: Optional[int] = None
    section: Optional[str] = None
    paragraph_id: Optional[int] = None
    has_measurements: bool = False
    numeric_expanded: bool = False

@dataclass
class Chunk:
    """최적화된 청크"""
    # 필수 필드
    text: str
    meta: ChunkMetaLight
    doc_meta_id: str  # DocumentMeta 참조 ID
    
    # 선택 필드 (필요시만 저장)
    measurements: Optional[List[Tuple[str, str]]] = None
    extra: Optional[dict] = None
    
    @property
    def doc_meta(self) -> DocumentMeta:
        """문서 메타 가져오기"""
        return MetadataRegistry.get_doc_meta(self.doc_meta_id)

class MetadataRegistry:
    """
    메타데이터 레지스트리 (싱글톤)
    
    목적:
    - 문서 메타데이터 중복 제거
    - WeakRef로 메모리 자동 관리
    """
    
    _instance: Optional['MetadataRegistry'] = None
    
    def __init__(self):
        self.doc_metas: WeakValueDictionary = WeakValueDictionary()
    
    @classmethod
    def get_instance(cls) -> 'MetadataRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def register_doc_meta(cls, meta: DocumentMeta) -> str:
        """문서 메타 등록"""
        registry = cls.get_instance()
        doc_id = meta.doc_id
        registry.doc_metas[doc_id] = meta
        return doc_id
    
    @classmethod
    def get_doc_meta(cls, doc_id: str) -> Optional[DocumentMeta]:
        """문서 메타 가져오기"""
        registry = cls.get_instance()
        return registry.doc_metas.get(doc_id)
    
    @classmethod
    def get_memory_saved(cls) -> dict:
        """메모리 절약량 추정"""
        import sys
        registry = cls.get_instance()
        
        num_docs = len(registry.doc_metas)
        bytes_per_meta = sys.getsizeof(DocumentMeta("", "", 0, ""))
        
        # 중복 제거로 절약된 메모리 (대략적)
        # 평균 청크 수를 100개로 가정
        avg_chunks_per_doc = 100
        saved = num_docs * (avg_chunks_per_doc - 1) * bytes_per_meta
        
        return {
            "num_documents": num_docs,
            "bytes_saved_estimate": saved,
            "mb_saved_estimate": saved / (1024 * 1024)
        }

# 사용 예시
def create_optimized_chunk(
    text: str,
    doc_id: str,
    filename: str,
    chunk_id: int,
    start_offset: int
) -> Chunk:
    """최적화된 청크 생성"""
    # 문서 메타 등록 (중복 제거)
    doc_meta = DocumentMeta(
        doc_id=doc_id,
        filename=filename,
        total_pages=100,
        doc_type="technical"
    )
    doc_meta_id = MetadataRegistry.register_doc_meta(doc_meta)
    
    # 청크 메타 생성
    chunk_meta = ChunkMetaLight(
        chunk_id=chunk_id,
        start_offset=start_offset,
        length=len(text),
        chunk_type="standard"
    )
    
    # 청크 생성
    return Chunk(
        text=text,
        meta=chunk_meta,
        doc_meta_id=doc_meta_id
    )
```

---

설계문서가 매우 방대해지고 있습니다. 나머지 섹션(성능 모니터링, 테스트 전략)을 계속 작성할까요, 아니면 여기까지 확인하시고 수정/보완이 필요한 부분이 있으신지 말씀해주시겠어요?

