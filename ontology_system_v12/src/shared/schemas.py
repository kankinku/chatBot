"""
Advisor System 핵심 스키마 정의
Evidence Layer, Time Series, Event 등을 위한 공통 데이터 모델

버전: 1.0
"""
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
import uuid
import hashlib


def generate_id(prefix: str) -> str:
    """고유 ID 생성"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def generate_content_hash(content: str) -> str:
    """콘텐츠 해시 생성 (중복 방지용)"""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# Enums
# =============================================================================

class RegimeType(str, Enum):
    """시장 레짐 타입"""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    INFLATION_UP = "inflation_up"
    DISINFLATION = "disinflation"
    GROWTH_UP = "growth_up"
    GROWTH_DOWN = "growth_down"
    LIQUIDITY_ABUNDANT = "liquidity_abundant"
    LIQUIDITY_TIGHT = "liquidity_tight"
    UNKNOWN = "unknown"


class EvidenceDirection(str, Enum):
    """Evidence 방향성"""
    PRO = "pro"           # 관계 지지
    CON = "con"           # 관계 반박
    NEUTRAL = "neutral"   # 중립


class FeatureType(str, Enum):
    """Feature 유형"""
    RAW = "raw"                    # 원시값
    SPREAD = "spread"              # 스프레드
    ROC = "roc"                    # Rate of Change
    ZSCORE = "zscore"              # Z-score
    YOY = "yoy"                    # Year over Year
    MOM = "mom"                    # Month over Month
    RATIO = "ratio"                # 비율
    CORRELATION = "correlation"    # 상관계수
    VOLATILITY = "volatility"      # 변동성
    PERCENTILE = "percentile"      # 백분위


class SourceType(str, Enum):
    """데이터 소스 유형"""
    API = "api"
    FILE = "file"
    SCRAPE = "scrape"
    MANUAL = "manual"


class DeltaMethod(str, Enum):
    """Delta 수집 방식"""
    SINCE_TIMESTAMP = "since_timestamp"
    CURSOR = "cursor"
    ETAG = "etag"
    HASH_DIFF = "hash_diff"
    FULL_REPLACE = "full_replace"


# =============================================================================
# Time Series 관련 스키마
# =============================================================================

class Observation(BaseModel):
    """
    시계열 관측값 (Append-only)
    
    Primary Key: (series_id, timestamp, as_of)
    """
    observation_id: str = Field(default_factory=lambda: generate_id("OBS"))
    
    # 식별자
    series_id: str                                    # 시계열 ID (예: SOFR, VIX)
    timestamp: datetime                               # 관측 시점
    as_of: datetime = Field(default_factory=datetime.now)  # 수집/수정 시점 (버전 관리)
    
    # 값
    value: float
    unit: Optional[str] = None                        # 단위 (%, bps, etc.)
    
    # 메타데이터
    source_id: str = ""                               # 데이터 소스 ID
    is_revision: bool = False                         # 수정 데이터 여부
    original_value: Optional[float] = None            # 수정 전 값 (수정인 경우)
    quality_flag: str = "ok"                          # ok, estimated, missing, error
    
    class Config:
        frozen = True  # Immutable


class TimeSeriesMetadata(BaseModel):
    """시계열 메타데이터"""
    series_id: str
    name: str
    description: Optional[str] = None
    source_id: str
    frequency: str = "daily"                          # daily, weekly, monthly, quarterly
    unit: Optional[str] = None
    start_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)


# =============================================================================
# Event 관련 스키마
# =============================================================================

class Event(BaseModel):
    """
    비정형 이벤트 (Append-only)
    
    Primary Key: (source_id, external_id) 또는 content_hash
    """
    event_id: str = Field(default_factory=lambda: generate_id("EVT"))
    
    # 식별자
    source_id: str                                    # 데이터 소스 ID
    external_id: Optional[str] = None                 # 원본 시스템 ID
    content_hash: str = ""                            # 콘텐츠 해시 (중복 방지)
    
    # 시점
    occurred_at: datetime                             # 이벤트 발생 시점
    collected_at: datetime = Field(default_factory=datetime.now)  # 수집 시점
    
    # 내용
    title: str
    content: Optional[str] = None
    event_type: str = "general"                       # news, policy, earnings, etc.
    
    # 추출된 정보
    entities: List[str] = Field(default_factory=list)  # 관련 엔티티 ID
    sentiment: Optional[float] = None                  # -1 ~ 1
    importance: float = 0.5                            # 0 ~ 1
    
    # 메타데이터
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # 리비전 관리
    revision: int = 1
    is_latest: bool = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.content_hash and self.content:
            object.__setattr__(self, 'content_hash', generate_content_hash(self.content))


# =============================================================================
# Feature 관련 스키마
# =============================================================================

class FeatureSpec(BaseModel):
    """Feature 정의 스펙"""
    feature_id: str
    name: str
    feature_type: FeatureType
    
    # 입력
    input_series: List[str]                           # 입력 시계열 ID
    
    # 계산 파라미터
    window_days: int = 0                              # 윈도우 크기 (일)
    params: Dict[str, Any] = Field(default_factory=dict)  # 추가 파라미터
    
    # 메타데이터
    description: Optional[str] = None
    unit: Optional[str] = None


class FeatureValue(BaseModel):
    """
    계산된 Feature 값
    
    Primary Key: (feature_id, timestamp, as_of)
    """
    feature_value_id: str = Field(default_factory=lambda: generate_id("FV"))
    
    # 식별자
    feature_id: str
    timestamp: datetime
    as_of: datetime = Field(default_factory=datetime.now)
    
    # 값
    value: float
    
    # 메타데이터
    computation_time_ms: Optional[float] = None
    input_observations_count: int = 0


# =============================================================================
# Evidence 관련 스키마
# =============================================================================

class EvidenceSpec(BaseModel):
    """
    Edge-Evidence 매핑 스펙
    특정 관계 타입/패턴에 대해 어떤 Feature로 검증할지 정의
    """
    spec_id: str = Field(default_factory=lambda: generate_id("ESPEC"))
    
    # 적용 대상 Edge 패턴
    edge_pattern: Dict[str, Any] = Field(default_factory=dict)
    # 예: {"head_type": "Indicator", "tail_type": "Asset", "relation_type": "Affect"}
    
    # 검증에 사용할 Features
    evidence_features: List[Dict[str, Any]] = Field(default_factory=list)
    # 예: [{"feature": "SOFR_ROC_30D", "direction": "positive", "weight": 0.4}]
    
    # 임계값
    thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "strong_pro": 0.7,
        "weak_pro": 0.3,
        "neutral_low": -0.3,
        "neutral_high": 0.3,
        "weak_con": -0.3,
        "strong_con": -0.7,
    })
    
    # 시차 탐색
    lag_days: List[int] = Field(default_factory=lambda: [0, 30, 60])
    
    # 레짐별 가중치 조정
    regime_applicability: Dict[str, float] = Field(default_factory=dict)
    
    # 활성화 여부
    is_active: bool = True


class EvidenceScore(BaseModel):
    """
    Edge에 대한 Evidence 점수 (시계열 저장)
    """
    evidence_id: str = Field(default_factory=lambda: generate_id("EV"))
    
    # 대상 Edge
    edge_id: str
    head_id: str
    tail_id: str
    relation_type: str
    
    # 점수
    pro_score: float = 0.0                            # 지지 점수 (0~1)
    con_score: float = 0.0                            # 반박 점수 (0~1)
    total_score: float = 0.0                          # 종합 점수 (-1~1)
    
    # 시점
    timestamp: datetime = Field(default_factory=datetime.now)
    as_of: datetime = Field(default_factory=datetime.now)
    
    # 레짐 컨텍스트
    regime: Optional[RegimeType] = None
    regime_adjustment: float = 1.0
    
    # 추적 정보 (설명 가능성)
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    # 예: [{"feature": "SOFR_ROC_30D", "value": 0.12, "contribution": 0.3}]
    
    # 최적 시차
    best_lag_days: int = 0
    
    # 신뢰도
    confidence: float = 0.5                           # 0~1


class AccumulatedEvidence(BaseModel):
    """
    누적/평활화된 Edge Evidence (업데이트 값 생성용)
    """
    edge_id: str
    
    # 누적 점수 (EMA/EWMA)
    accumulated_score: float = 0.0
    score_volatility: float = 0.0                     # 점수 변동성
    
    # 통계
    observation_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    
    # 시간 정보
    first_observed: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # 권장 업데이트
    suggested_confidence_delta: float = 0.0           # edge confidence에 적용할 변화량


# =============================================================================
# Regime 관련 스키마
# =============================================================================

class RegimeCondition(BaseModel):
    """레짐 조건 정의"""
    feature: str
    operator: str                                     # >, <, >=, <=, ==, between
    threshold: Union[float, List[float]]


class RegimeSpec(BaseModel):
    """레짐 정의 스펙"""
    regime_id: str = Field(default_factory=lambda: generate_id("REG"))
    name: str
    regime_type: RegimeType
    
    conditions: List[RegimeCondition] = Field(default_factory=list)
    priority: int = 1                                 # 높을수록 우선
    
    description: Optional[str] = None
    is_active: bool = True


class RegimeDetectionResult(BaseModel):
    """레짐 탐지 결과"""
    detection_id: str = Field(default_factory=lambda: generate_id("RD"))
    
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # 탐지된 레짐 (확률 포함)
    detected_regimes: Dict[RegimeType, float] = Field(default_factory=dict)
    # 예: {RegimeType.RISK_ON: 0.7, RegimeType.INFLATION_UP: 0.5}
    
    primary_regime: Optional[RegimeType] = None
    primary_probability: float = 0.0
    
    # 불확실성
    uncertainty: float = 0.0                          # 0~1, 높을수록 불확실
    
    # 사용된 Feature 값
    feature_snapshot: Dict[str, float] = Field(default_factory=dict)


# =============================================================================
# Source 관련 스키마
# =============================================================================

class SourceSpec(BaseModel):
    """데이터 소스 정의"""
    source_id: str
    name: str
    source_type: SourceType
    
    # Delta 수집 방식
    delta_method: DeltaMethod = DeltaMethod.SINCE_TIMESTAMP
    
    # 연결 정보
    endpoint: Optional[str] = None
    credentials_key: Optional[str] = None             # 환경변수 or secret key
    
    # 수집 설정
    refresh_interval_minutes: int = 60
    retry_count: int = 3
    timeout_seconds: int = 30
    
    # 메타데이터
    provides_series: List[str] = Field(default_factory=list)
    is_active: bool = True


class FetchState(BaseModel):
    """소스별 수집 상태 (마지막 위치)"""
    source_id: str
    stream: str = "default"                           # 소스 내 스트림 구분
    
    # 마지막 수집 위치
    last_timestamp: Optional[datetime] = None
    last_cursor: Optional[str] = None
    last_etag: Optional[str] = None
    last_hash: Optional[str] = None
    
    # 상태
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    consecutive_failures: int = 0
    
    # 통계
    total_fetched: int = 0
    total_errors: int = 0


# =============================================================================
# Dependency 관련 스키마
# =============================================================================

class DependencyEdge(BaseModel):
    """의존성 그래프 엣지"""
    from_id: str
    from_type: str                                    # series, feature, evidence, edge
    to_id: str
    to_type: str
    dependency_type: str = "input"                    # input, triggers, updates


class DependencyGraph(BaseModel):
    """의존성 그래프 (증분 업데이트용)"""
    edges: List[DependencyEdge] = Field(default_factory=list)
    
    # 인덱스 (런타임에 구성)
    forward_index: Dict[str, List[str]] = Field(default_factory=dict)   # A → [B, C]
    backward_index: Dict[str, List[str]] = Field(default_factory=dict)  # B → [A]
