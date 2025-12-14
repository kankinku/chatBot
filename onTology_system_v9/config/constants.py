"""
Application Constants
변경되지 않는 상수값들을 중앙 관리
코드 내 하드코딩 방지
"""
from dataclasses import dataclass
from pathlib import Path
from config.settings import settings


@dataclass(frozen=True)
class CachePaths:
    """캐시 파일 경로 상수"""
    MARKET_CACHE: Path = settings.CACHE_DIR / "market_cache.json"
    CUSTOM_MAPPING: Path = settings.CACHE_DIR / "custom_market_mapping.json"
    ALL_PAIRS: Path = settings.CACHE_DIR / "all_pairs.json"
    FUNDAMENTALS: Path = settings.CACHE_DIR / "fundamentals.json"
    MOMENTUM: Path = settings.CACHE_DIR / "momentum.json"
    PRICE_DATA: Path = settings.CACHE_DIR / "price_data.json"


@dataclass(frozen=True)
class APIEndpoints:
    """외부 API 엔드포인트"""
    OLLAMA_GENERATE: str = "/api/generate"
    OLLAMA_TAGS: str = "/api/tags"


@dataclass(frozen=True)
class DefaultParams:
    """기본 파라미터 값"""
    # Pair Trading
    CORRELATION_THRESHOLD: float = 0.7
    ROLLING_WINDOW: int = 120
    MOMENTUM_WINDOW: int = 126
    BACKTEST_LOOKBACK: int = 63
    MAX_HOLDING_DAYS: int = 60
    ENTRY_Z_SCORE: float = 1.5
    EXIT_Z_SCORE: float = 0.0
    
    # Simulation
    MAX_PROPAGATION_DEPTH: int = 3
    DECAY_FACTOR: float = 0.9
    MIN_IMPACT_THRESHOLD: float = 0.1
    
    # Market Data
    DEFAULT_START_DATE: str = "2020-01-01"
    CACHE_EXPIRY_HOURS: int = 6


@dataclass(frozen=True)
class LogMessages:
    """표준화된 로그 메시지 포맷"""
    # System
    STARTUP = "[System] {} started on port {}"
    SHUTDOWN = "[System] {} shutting down"
    
    # Pipeline
    M1_INIT = "[M1] Initialized with Ollama model: {} at {}"
    M1_SUCCESS = "[M1] Successfully extracted {} fragments"
    M1_FALLBACK = "[M1] Using Mock Fallback"
    
    # Graph
    GRAPH_SAVED = "[Graph] Saved {} nodes to {}"
    GRAPH_LOADED = "[Graph] Loaded {} nodes from {}"
    
    # Market Data
    MARKET_INIT = "[Market] Initializing data sync..."
    MARKET_CACHE_HIT = "[Market] Cache hit for {}"
    MARKET_FETCH = "[Market] Fetching {} from API"


@dataclass(frozen=True)
class LLMConstraints:
    """LLM 출력 제약 사항 - 모든 프롬프트에 적용"""
    
    # 공통 제약 사항 (모든 LLM 프롬프트에 포함)
    COMMON_RULES: str = """
[출력 형식 제약 - 반드시 준수]
1. 언어: 한국어와 영어만 사용 (다른 언어 절대 금지)
2. 이모지(😀🚀📈 등) 절대 사용 금지
3. 마크다운 강조 금지: **, ##, *, >, ``` 등 마크다운 문법 사용 금지
4. 제목/소제목 금지: "AI 분석 리포트", "제1장" 등 제목 형식 사용 금지
5. 평문 텍스트로만 작성할 것
6. 표가 필요한 경우에만 마크다운 표(| | |) 사용 허용
"""

    # 분석 리포트용 추가 제약
    ANALYSIS_RULES: str = """
[분석 스타일]
- 어조: 감정적 미사여구 배제, 냉정하고 단호하게 작성
- 종결어미: ~함. ~임. ~됨. 체로 종결
- 불필요한 수식어 없이 핵심만 간결하게 작성
"""

    # 표 형식 가이드
    TABLE_FORMAT: str = """
[표 작성 규칙]
- 표가 필요한 경우 마크다운 표 형식 사용: | 열1 | 열2 |
- 표 헤더와 구분선 포함: | --- | --- |
- 표는 데이터 요약에만 사용, 불필요한 표 작성 금지
"""
