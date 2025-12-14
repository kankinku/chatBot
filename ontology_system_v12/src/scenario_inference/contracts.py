"""
핵심 내부 객체 - 학습용 계약 스키마

모든 결정/추론/검증의 표준 데이터 구조
"""
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import hashlib


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ============================================================
# Enums
# ============================================================

class DecisionType(str, Enum):
    """결정 유형 (5종)"""
    MODE = "mode"                    # 추론 모드 선택
    PATH_SELECT = "path_select"      # 경로 후보 선택/필터링
    PLAN_SELECT = "plan_select"      # 증거 계획 선택
    FUSION_SCORE = "fusion_score"    # 합성 점수/가중치 적용
    OUTPUT_FILTER = "output_filter"  # 최종 출력 필터링


class InferenceMode(str, Enum):
    """추론 모드"""
    GRAPH_ONLY = "graph_only"              # 그래프 구조만
    GRAPH_EVIDENCE = "graph_evidence"       # 그래프 + 수치증거
    DOMAIN_ONLY = "domain_only"             # 도메인 지식만
    DOMAIN_PERSONAL = "domain_personal"     # 도메인 + 퍼스널 보조
    FULL = "full"                           # 전체 모드
    LIGHT = "light"                         # 경량 모드 (비용 최소)


class VerificationOutcome(str, Enum):
    """검증 결과 유형"""
    PASS_DIRECTION = "pass_direction"      # 방향 적중
    FAIL_DIRECTION = "fail_direction"      # 방향 반대
    NEUTRAL_NO_EFFECT = "neutral_no_effect"  # 효과 없음
    CONDITIONAL = "conditional"            # 조건부 (레짐 의존)
    DATA_MISSING = "data_missing"          # 데이터 결측


# ============================================================
# Layer 1: ScenarioState (입력 상태)
# ============================================================

class QueryFeatures(BaseModel):
    """질의 특성"""
    query_type: str = "unknown"              # scenario, prediction, analysis, comparison
    target_entities: List[str] = Field(default_factory=list)
    entity_count: int = 0
    entity_types: List[str] = Field(default_factory=list)
    time_range_days: Optional[int] = None    # 질의의 시간 범위
    has_condition: bool = False              # 조건절 포함 여부


class KGFeatures(BaseModel):
    """KG 구조 특성"""
    target_entity_degrees: Dict[str, int] = Field(default_factory=dict)
    path_candidate_count: int = 0
    avg_path_length: float = 0.0
    domain_edge_ratio: float = 1.0           # 도메인 / 전체
    personal_edge_ratio: float = 0.0
    conflict_edge_ratio: float = 0.0         # 충돌 엣지 비율
    drift_flag_ratio: float = 0.0            # 드리프트 플래그 비율


class EvidenceCoverage(BaseModel):
    """증거 커버리지 특성"""
    related_edge_count: int = 0
    evidence_freshness_days: float = 0.0     # 평균 최신성 (일)
    coverage_ratio: float = 0.0              # 증거 있는 엣지 비율
    avg_evidence_strength: float = 0.0       # 평균 강도
    uncertainty_ratio: float = 0.0           # 불확실 증거 비율


class TimeSeriesFeatures(BaseModel):
    """시계열/레짐 특성 (경량)"""
    volatility_regime: str = "medium"        # low, medium, high
    trend_signal: str = "neutral"            # up, down, neutral, reversal
    missing_ratio: float = 0.0               # 결측률
    regime_changes_30d: int = 0              # 최근 30일 레짐 변화 횟수


class BudgetFeatures(BaseModel):
    """비용 제약"""
    max_tests: int = 10                      # 최대 테스트 수
    max_time_seconds: float = 30.0           # 최대 허용 시간
    allow_external_calls: bool = True        # 외부 API 호출 허용


class ScenarioState(BaseModel):
    """
    시나리오 추론 입력 상태
    
    모든 결정 지점에서 참조하는 컨텍스트
    """
    state_id: str = Field(default_factory=lambda: generate_id("STATE"))
    as_of_time: datetime
    
    # 스냅샷 참조
    kg_snapshot_ref: Optional[str] = None
    ts_snapshot_ref: Optional[str] = None
    evidence_snapshot_ref: Optional[str] = None
    
    # 질의 정보
    query_id: str = ""
    query_text: str = ""
    query_features: QueryFeatures = Field(default_factory=QueryFeatures)
    
    # KG 특성
    kg_features: KGFeatures = Field(default_factory=KGFeatures)
    
    # 증거 커버리지
    evidence_coverage: EvidenceCoverage = Field(default_factory=EvidenceCoverage)
    
    # 시계열/레짐
    ts_features: TimeSeriesFeatures = Field(default_factory=TimeSeriesFeatures)
    
    # 비용 제약
    budget: BudgetFeatures = Field(default_factory=BudgetFeatures)
    
    def to_feature_vector(self) -> Dict[str, float]:
        """학습용 피처 벡터 추출"""
        return {
            "entity_count": self.query_features.entity_count,
            "has_condition": 1.0 if self.query_features.has_condition else 0.0,
            "path_candidate_count": self.kg_features.path_candidate_count,
            "avg_path_length": self.kg_features.avg_path_length,
            "domain_edge_ratio": self.kg_features.domain_edge_ratio,
            "conflict_edge_ratio": self.kg_features.conflict_edge_ratio,
            "evidence_coverage": self.evidence_coverage.coverage_ratio,
            "evidence_freshness": self.evidence_coverage.evidence_freshness_days,
            "volatility_high": 1.0 if self.ts_features.volatility_regime == "high" else 0.0,
            "max_tests": self.budget.max_tests,
        }
    
    def compute_hash(self) -> str:
        """상태 해시 (재현성 검증용)"""
        content = f"{self.as_of_time.isoformat()}_{self.query_text}_{self.kg_snapshot_ref}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================================
# Layer 1: DecisionRecord (결정 단위 로그)
# ============================================================

class CandidateSummary(BaseModel):
    """후보 요약"""
    candidate_id: str
    features: Dict[str, float] = Field(default_factory=dict)
    score: float = 0.0


class DecisionRecord(BaseModel):
    """
    결정 단위 로그
    
    모든 결정 지점에서 생성되어 Trace에 누적
    """
    decision_id: str = Field(default_factory=lambda: generate_id("DEC"))
    decision_type: DecisionType
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # 후보 목록
    candidates: List[CandidateSummary] = Field(default_factory=list)
    candidate_count: int = 0
    
    # 선택 결과
    chosen_id: Optional[str] = None
    chosen_ids: List[str] = Field(default_factory=list)  # 다중 선택 시
    
    # 점수/근거
    scores: Dict[str, float] = Field(default_factory=dict)  # 후보별 점수
    selection_rationale: str = ""
    
    # 정책 정보
    policy_version: str = ""
    policy_type: str = ""  # rule, learned, hybrid
    
    # 적용된 제약
    constraints_applied: List[str] = Field(default_factory=list)
    constraints_violated: List[str] = Field(default_factory=list)
    
    # 비용
    cost_estimate: float = 0.0
    cost_actual: Optional[float] = None
    
    # 메타
    confidence: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Layer 1: EvidencePlan (증거 계획)
# ============================================================

class EvidenceTest(BaseModel):
    """개별 증거 테스트"""
    test_id: str = Field(default_factory=lambda: generate_id("TEST"))
    target_edge_id: str = ""
    feature_id: str = ""
    test_type: str = "direction"  # direction, correlation, threshold, volatility
    window_days: int = 30
    alternative_features: List[str] = Field(default_factory=list)
    expected_direction: str = "+"  # +, -, neutral


class EvidencePlan(BaseModel):
    """
    증거 수집/검증 계획
    
    어떤 테스트를 어떤 순서로 실행할지 정의
    """
    plan_id: str = Field(default_factory=lambda: generate_id("PLAN"))
    plan_type: str = "standard"  # minimal, standard, comprehensive, robust
    
    # 테스트 목록
    tests: List[EvidenceTest] = Field(default_factory=list)
    test_count: int = 0
    
    # 비용 상한
    max_cost: float = 10.0
    max_time_seconds: float = 10.0
    
    # 강건성 옵션
    include_robustness_check: bool = False
    alternative_window_sizes: List[int] = Field(default_factory=list)
    
    # 우선순위
    priority_edges: List[str] = Field(default_factory=list)
    
    # 조기 종료 조건
    early_stop_on_contradiction: bool = True
    min_coverage_threshold: float = 0.5
    
    def estimate_cost(self) -> float:
        """비용 추정"""
        base_cost = len(self.tests) * 1.0
        robustness_cost = len(self.alternative_window_sizes) * 0.5 if self.include_robustness_check else 0
        return base_cost + robustness_cost


# ============================================================
# Layer 2: ScenarioTrace (추론 과정 전체)
# ============================================================

class ScenarioTrace(BaseModel):
    """
    시나리오 추론 과정 전체 기록
    
    모든 결정/후보/점수/근거/비용을 표준 스키마로 저장
    학습 데이터 생성의 원천
    """
    trace_id: str = Field(default_factory=lambda: generate_id("TRACE"))
    as_of_time: datetime
    query_id: str
    correlation_id: Optional[str] = None  # 요청 추적용
    
    # 상태 참조
    state_hash: str = ""
    kg_snapshot_ref: Optional[str] = None
    ts_snapshot_ref: Optional[str] = None
    evidence_snapshot_ref: Optional[str] = None
    
    # 결정 기록들 (시간순)
    decision_records: List[DecisionRecord] = Field(default_factory=list)
    
    # 증거 실행 결과
    evidence_trace: Dict[str, Any] = Field(default_factory=dict)
    """
    {
        "tests_executed": [...],
        "windows_used": [...],
        "alternative_results": [...],
        "robustness_scores": {...}
    }
    """
    
    # 최종 시나리오들 (top-K)
    final_scenarios: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_breakdown: Dict[str, float] = Field(default_factory=dict)
    
    # 실패/경고 플래그
    failure_flags: List[str] = Field(default_factory=list)
    """
    가능한 플래그:
    - "leak_suspected": 누수 의심
    - "data_insufficient": 데이터 부족
    - "conflict_severe": 심각한 충돌
    - "evidence_weak": 약한 증거
    - "budget_exceeded": 비용 초과
    """
    
    # 비용/시간
    total_cost: float = 0.0
    total_time_seconds: float = 0.0
    
    # 메타
    pipeline_version: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    
    def add_decision(self, record: DecisionRecord) -> None:
        """결정 기록 추가"""
        self.decision_records.append(record)
    
    def get_decisions_by_type(self, decision_type: DecisionType) -> List[DecisionRecord]:
        """유형별 결정 조회"""
        return [d for d in self.decision_records if d.decision_type == decision_type]


# ============================================================
# Layer 2: VerificationReport (후행 검증)
# ============================================================

class VerificationTarget(BaseModel):
    """검증 대상"""
    target_id: str
    entity_id: str
    metric_type: str  # return, direction, threshold, volatility
    expected_direction: str  # +, -, neutral
    threshold: Optional[float] = None
    window_days: int = 30


class VerificationReport(BaseModel):
    """
    후행 검증 리포트
    
    시나리오가 실제로 맞았는지 검증
    """
    report_id: str = Field(default_factory=lambda: generate_id("VRPT"))
    scenario_id: str
    trace_id: str
    
    # 시간
    scenario_as_of: datetime
    evaluation_horizon_days: int = 30
    verified_at: datetime = Field(default_factory=datetime.now)
    
    # 검증 대상
    targets: List[VerificationTarget] = Field(default_factory=list)
    
    # 실제 결과
    outcome_values: Dict[str, float] = Field(default_factory=dict)
    direction_hits: Dict[str, bool] = Field(default_factory=dict)
    
    # 전체 판정
    overall_outcome: VerificationOutcome = VerificationOutcome.DATA_MISSING
    accuracy: float = 0.0  # 방향 적중률
    
    # 실패 분석
    failure_types: List[str] = Field(default_factory=list)
    """
    - "direction_opposite": 방향 반대
    - "no_effect": 효과 없음
    - "regime_dependent": 레짐 의존
    - "data_missing": 데이터 결측
    - "magnitude_wrong": 크기 오류
    """
    
    # 메타
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Layer 2: RewardBreakdown (다면 점수)
# ============================================================

class RewardBreakdown(BaseModel):
    """
    다면 보상 분해
    
    단일 스칼라가 아닌 분해된 보상 (Goodhart 방지)
    **반드시 저장해야 디버깅 가능**
    """
    breakdown_id: str = Field(default_factory=lambda: generate_id("RWD"))
    trace_id: str
    verification_report_id: Optional[str] = None
    
    # === 각 차원별 점수 (0~1) ===
    
    # 정확도: 방향/부호 적중 (핵심 타깃 우선 가중)
    accuracy: float = 0.0
    accuracy_details: Dict[str, float] = Field(default_factory=dict)
    
    # 캘리브레이션: confidence와 실제 적중의 정합 (과신 페널티)
    calibration: float = 0.0
    calibration_error: float = 0.0  # |confidence - accuracy|
    
    # 강건성: 윈도우/대체지표/지연 변화에도 유지되는지
    robustness: float = 0.0
    robustness_details: Dict[str, float] = Field(default_factory=dict)
    
    # 비용: 실행 비용 (테스트 수, 시간, 외부 호출) 역수
    cost_efficiency: float = 1.0  # 1 - normalized_cost
    actual_cost: float = 0.0
    
    # 제약 위반 페널티
    constraint_penalty: float = 0.0
    constraint_violations: List[str] = Field(default_factory=list)
    """
    - "evidence_insufficient": 증거 부족
    - "forbidden_path": 금지 경로 사용
    - "conflict_ignored": 충돌 무시
    - "leak_suspected": 누수 의심
    """
    
    # === 가중치 (학습 대상) ===
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "accuracy": 0.40,
        "calibration": 0.20,
        "robustness": 0.15,
        "cost_efficiency": 0.10,
        "constraint_penalty": 0.15,
    })
    
    # === 최종 스칼라 보상 ===
    scalar_reward: float = 0.0
    
    def compute_scalar(self) -> float:
        """가중합으로 스칼라 보상 계산"""
        w = self.weights
        self.scalar_reward = (
            w.get("accuracy", 0.4) * self.accuracy +
            w.get("calibration", 0.2) * self.calibration +
            w.get("robustness", 0.15) * self.robustness +
            w.get("cost_efficiency", 0.1) * self.cost_efficiency -
            w.get("constraint_penalty", 0.15) * self.constraint_penalty
        )
        return self.scalar_reward
    
    def to_dict(self) -> Dict[str, float]:
        """전체 breakdown을 dict로"""
        return {
            "accuracy": self.accuracy,
            "calibration": self.calibration,
            "robustness": self.robustness,
            "cost_efficiency": self.cost_efficiency,
            "constraint_penalty": self.constraint_penalty,
            "scalar_reward": self.scalar_reward,
        }


# ============================================================
# EvidencePlan 템플릿 (공식 지원 5개)
# ============================================================

EVIDENCE_PLAN_TEMPLATES = {
    "minimal": EvidencePlan(
        plan_type="minimal",
        max_cost=3.0,
        max_time_seconds=5.0,
        include_robustness_check=False,
        early_stop_on_contradiction=True,
        min_coverage_threshold=0.3,
    ),
    "standard": EvidencePlan(
        plan_type="standard",
        max_cost=10.0,
        max_time_seconds=15.0,
        include_robustness_check=False,
        early_stop_on_contradiction=True,
        min_coverage_threshold=0.5,
    ),
    "comprehensive": EvidencePlan(
        plan_type="comprehensive",
        max_cost=25.0,
        max_time_seconds=30.0,
        include_robustness_check=True,
        alternative_window_sizes=[15, 60],
        early_stop_on_contradiction=False,
        min_coverage_threshold=0.7,
    ),
    "robust": EvidencePlan(
        plan_type="robust",
        max_cost=40.0,
        max_time_seconds=45.0,
        include_robustness_check=True,
        alternative_window_sizes=[7, 15, 30, 60, 90],
        early_stop_on_contradiction=False,
        min_coverage_threshold=0.8,
    ),
    "quick_check": EvidencePlan(
        plan_type="quick_check",
        max_cost=2.0,
        max_time_seconds=3.0,
        include_robustness_check=False,
        early_stop_on_contradiction=True,
        min_coverage_threshold=0.2,
    ),
}
