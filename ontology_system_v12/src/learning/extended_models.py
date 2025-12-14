"""
Extended Training Sample
Evidence trace, Regime snapshot을 포함한 확장 학습 샘플

8단계: Learning/Policy 연결 강화
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class ExtendedTrainingSample(BaseModel):
    """
    확장된 학습 샘플
    
    기존 TrainingSample에 다음을 추가:
    - Evidence trace: 당시 evidence 상태
    - Regime snapshot: 당시 regime 상태
    - Conclusion label: 당시 조언 결론
    - Outcome metrics: 이후 실제 결과
    """
    sample_id: str = Field(default_factory=lambda: generate_id("ESAMP"))
    
    # 기존 필드
    text: str
    fragment_id: Optional[str] = None
    task_type: str
    labels: Dict[str, Any] = Field(default_factory=dict)
    source: str
    label_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    source_edge_id: Optional[str] = None
    
    # === 신규 필드 (8단계) ===
    
    # Evidence 정보
    evidence_trace: Dict[str, float] = Field(default_factory=dict)
    """
    당시 evidence 상태
    예: {"SOFR_ROC_30D": 0.12, "VIX_ZSCORE": 1.5, ...}
    """
    
    evidence_scores: Dict[str, float] = Field(default_factory=dict)
    """
    Edge별 evidence score
    예: {"EDGE_001": 0.7, "EDGE_002": -0.3, ...}
    """
    
    # Regime 정보
    regime_snapshot: Dict[str, Any] = Field(default_factory=dict)
    """
    당시 regime 상태
    예: {
        "primary_regime": "risk_on",
        "probability": 0.8,
        "detected_regimes": {"risk_on": 0.8, "growth_up": 0.6}
    }
    """
    
    # 결론 정보
    conclusion_label: Optional[str] = None
    """당시 조언 결론 (예: "positive", "negative", "neutral")"""
    
    conclusion_confidence: Optional[float] = None
    """결론에 대한 시스템 confidence"""
    
    conclusion_path: List[str] = Field(default_factory=list)
    """사용된 추론 경로 (노드 리스트)"""
    
    # 결과 정보
    outcome_metrics: Dict[str, float] = Field(default_factory=dict)
    """
    이후 실제 결과 (lookahead)
    예: {
        "return_30d": 0.05,
        "volatility_30d": 0.15,
        "max_drawdown_30d": -0.03
    }
    """
    
    outcome_date: Optional[datetime] = None
    """결과 측정 날짜"""
    
    # 평가 정보
    was_correct: Optional[bool] = None
    """결론이 실제로 맞았는지 (나중에 채워짐)"""
    
    error_magnitude: Optional[float] = None
    """오차 크기 (예측 vs 실제)"""


class ConclusionOutcomePair(BaseModel):
    """결론-결과 페어 (학습용)"""
    pair_id: str = Field(default_factory=lambda: generate_id("COP"))
    
    # 결론 시점
    conclusion_date: datetime
    conclusion_direction: str  # +, -, neutral
    conclusion_confidence: float
    
    # 결론 컨텍스트
    regime_at_conclusion: Optional[str] = None
    evidence_scores_at_conclusion: Dict[str, float] = Field(default_factory=dict)
    
    # 결과 (lookahead 후)
    outcome_date: datetime
    actual_direction: str  # +, -, neutral
    actual_magnitude: float
    
    # 평가
    is_correct: bool
    confidence_error: float  # |confidence - accuracy|
    
    # 분류
    regime_correct: bool = True  # 레짐 판단이 맞았는지
    evidence_aligned: bool = True  # evidence와 결과가 일치했는지


class PolicyEvaluationResult(BaseModel):
    """정책 평가 결과"""
    evaluation_id: str = Field(default_factory=lambda: generate_id("PEVAL"))
    policy_version: str
    evaluation_date: datetime = Field(default_factory=datetime.now)
    
    # 평가 기간
    period_start: datetime
    period_end: datetime
    
    # 샘플 통계
    total_samples: int = 0
    correct_samples: int = 0
    accuracy: float = 0.0
    
    # 세부 메트릭
    precision_by_confidence: Dict[str, float] = Field(default_factory=dict)
    """confidence 구간별 precision"""
    
    accuracy_by_regime: Dict[str, float] = Field(default_factory=dict)
    """regime별 accuracy"""
    
    calibration_score: float = 0.0
    """confidence calibration 점수"""
    
    # 개선 여부
    improved_from_baseline: bool = False
    improvement_delta: float = 0.0
    
    # 권장 사항
    recommendations: List[str] = Field(default_factory=list)


class PolicyOptimizationTarget(BaseModel):
    """정책 최적화 대상"""
    target_id: str = Field(default_factory=lambda: generate_id("POT"))
    
    # 최적화 대상 파라미터
    parameter_name: str
    parameter_type: str  # "weight", "threshold", "decay"
    
    # 현재 값
    current_value: float
    
    # 탐색 범위
    min_value: float
    max_value: float
    step_size: float = 0.1
    
    # 중요도
    sensitivity: float = 0.5  # 이 파라미터 변화가 결과에 미치는 영향
    
    # 제약조건
    constraints: Dict[str, Any] = Field(default_factory=dict)


class ExtendedPolicyConfig(BaseModel):
    """
    확장된 정책 설정 (8단계)
    
    기존 PolicyConfig에 다음을 추가:
    - Evidence 가중치
    - Regime 적용 강도
    - 결론 임계값
    """
    config_id: str = Field(default_factory=lambda: generate_id("EPOL"))
    version: str
    
    # 기존 EES 가중치
    ees_weights: Dict[str, float] = Field(default_factory=lambda: {
        "domain": 0.4,
        "personal": 0.2,
        "semantic": 0.15,
        "temporal": 0.1,
        "validation": 0.1,
        "graph": 0.05,
    })
    
    # 기존 PCS 가중치
    pcs_weights: Dict[str, float] = Field(default_factory=lambda: {
        "domain_proximity": 0.25,
        "semantic_strength": 0.3,
        "user_origin": 0.2,
        "consistency": 0.25,
    })
    
    # === 신규 (8단계) ===
    
    # Evidence 가중치
    evidence_weights: Dict[str, float] = Field(default_factory=lambda: {
        "pro_score": 0.4,
        "con_score": 0.3,
        "lag_penalty": 0.1,
        "trace_count": 0.2,
    })
    
    # Regime 적용 강도
    regime_applicability: Dict[str, float] = Field(default_factory=lambda: {
        "risk_on": 1.0,
        "risk_off": 1.0,
        "inflation_up": 1.0,
        "disinflation": 1.0,
        "growth_up": 1.0,
        "growth_down": 1.0,
        "liquidity_abundant": 1.0,
        "liquidity_tight": 1.0,
    })
    
    # 결론 임계값
    conclusion_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "strong_positive": 0.7,
        "weak_positive": 0.3,
        "neutral_low": -0.3,
        "neutral_high": 0.3,
        "weak_negative": -0.3,
        "strong_negative": -0.7,
        "min_confidence": 0.4,  # 결론 생성 최소 confidence
    })
    
    # 기존 Thresholds
    thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "domain_candidate": 0.55,
        "personal_candidate": 0.35,
        "drift_signal": 0.6,
        "promotion": 0.8,
    })
    
    # Path reasoning
    path_length_penalty: float = 0.1
    max_path_length: int = 4
    
    # 메타
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = False
    notes: str = ""
    
    # 학습 이력
    trained_from: Optional[str] = None  # 어떤 데이터로 학습되었는지
    training_samples_count: int = 0
    validation_accuracy: float = 0.0
