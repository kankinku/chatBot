"""
Inference Schemas

시나리오 추론 서비스에 사용되는 데이터 구조 (DTO)
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

# ============================================================
# Inference 요청/응답
# ============================================================

class InferenceRequest(BaseModel):
    """시나리오 추론 요청"""
    query: str = Field(..., min_length=1, description="질의 텍스트")
    as_of_date: Optional[str] = Field(
        None, 
        description="기준일 (YYYY-MM-DD). 미지정시 현재 시점"
    )
    budget_max_tests: int = Field(10, ge=1, le=100, description="최대 테스트 수")
    budget_max_time: float = Field(30.0, ge=1.0, le=300.0, description="최대 시간(초)")
    include_trace: bool = Field(False, description="상세 추론 과정 포함 여부")


class ScenarioResult(BaseModel):
    """시나리오 결과"""
    scenario_id: str
    description: str
    direction: str  # positive, negative, neutral
    confidence: float
    evidence_summary: Dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    """시나리오 추론 응답"""
    trace_id: str
    query: str
    as_of_time: datetime
    mode_selected: str
    scenarios: List[ScenarioResult]
    total_cost: float
    processing_time_seconds: float
    failure_flags: List[str] = Field(default_factory=list)
    trace_detail: Optional[Dict[str, Any]] = None


# ============================================================
# Trace 조회
# ============================================================

class TraceDetailResponse(BaseModel):
    """추론 과정 상세 조회 응답"""
    trace_id: str
    query_id: str
    query_text: str
    as_of_time: datetime
    decision_records: List[Dict[str, Any]]
    evidence_trace: Dict[str, Any]
    final_scenarios: List[Dict[str, Any]]
    failure_flags: List[str]
    total_cost: float
    total_time_seconds: float
    created_at: datetime


# ============================================================
# Verification / Feedback
# ============================================================

class FeedbackInput(BaseModel):
    """검증 피드백 입력"""
    trace_id: str = Field(..., description="대상 Trace ID")
    target_entity: str = Field(..., description="검증 대상 엔티티")
    actual_outcome: str = Field(..., description="실제 결과 (+, -, neutral)")
    actual_value: Optional[float] = Field(None, description="실제 수치값 (옵션)")
    feedback_note: str = Field("", description="추가 메모")


class FeedbackResponse(BaseModel):
    """피드백 처리 응답"""
    feedback_id: str
    trace_id: str
    accuracy_computed: float
    is_direction_hit: bool
    recorded_at: datetime
