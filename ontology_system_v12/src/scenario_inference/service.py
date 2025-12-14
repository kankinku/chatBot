"""
Inference Service

시나리오 추론 비즈니스 로직
"""
import logging
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from src.scenario_inference.contracts import (
    ScenarioState, ScenarioTrace, DecisionRecord, BudgetFeatures
)
from src.scenario_inference.layer0_guards import AsOfContext, AsOfContextGuard, SnapshotResolver
from src.scenario_inference.layer1_inference import (
    ScenarioStateBuilder, InferenceModeSelector, PathCandidateGenerator,
    EvidencePlanGenerator, FusionScorer, ScoredScenario, ModeDecision
)
from src.scenario_inference.layer2_trace import ScenarioTraceRecorder
from src.scenario_inference.schemas import InferenceRequest, InferenceResponse, ScenarioResult
from src.infrastructure.market_data_client import MarketDataClient

logger = logging.getLogger(__name__)


class InferenceService:
    """
    시나리오 추론 서비스
    
    책임:
    - Layer 0~2 파이프라인 오케스트레이션
    - 추론 실행 및 결과 반환
    """
    
    def __init__(
        self,
        trace_recorder: Optional[ScenarioTraceRecorder] = None,
        db_path: str = "./data/traces.db"
    ):
        self.trace_recorder = trace_recorder or ScenarioTraceRecorder(db_path=db_path)
        self.state_builder = ScenarioStateBuilder()
        self.mode_selector = InferenceModeSelector()
        self.path_generator = PathCandidateGenerator()
        self.plan_generator = EvidencePlanGenerator()
        self.fusion_scorer = FusionScorer()
        self.market_client = MarketDataClient()
        
        # 통계
        self._total_inferences = 0
        self._total_time = 0.0

    def _collect_evidence(
        self,
        paths,
        plan
    ) -> Dict[str, Any]:
        """
        증거 수집
        - MarketDataClient를 통해 실제 데이터 조회
        - 경로 상의 엔티티에 대한 시장 데이터 분석
        """
        results = {}
        
        # 1. 전체 시장 상황 조회
        market_context = self.market_client.get_market_indicators()
        
        for path in paths:
            path_evidence = {
                "strength": 0.5, # 기본값
                "robustness": 0.5,
                "data_points": 0,
                "trends": {},
                "market_context": market_context
            }
            
            # 경로 상의 노드들 확인
            for node in path.nodes:
                ticker = self.market_client.get_ticker(node)
                if not ticker:
                    # 부분 일치 검색 (간단히)
                    # MarketDataClient의 내부 map 접근
                    for k, v in self.market_client._ticker_map.items():
                        if k in node:
                            ticker = v
                            break
                            
                if ticker:
                    # 데이터 조회 (최근 30일)
                    df = self.market_client.get_price_history(ticker, days=30)
                    if not df.empty:
                        # 간단한 트렌드 분석
                        start_price = df['Close'].iloc[0]
                        end_price = df['Close'].iloc[-1]
                        if isinstance(start_price, pd.Series): # yfinance 리턴 타입 주의
                            start_price = start_price.iloc[0]
                            end_price = end_price.iloc[0]
                            
                        # float 변환
                        start_price = float(start_price)
                        end_price = float(end_price)
                        
                        change = (end_price - start_price) / start_price
                        
                        path_evidence["trends"][node] = {
                            "change_30d": change,
                            "current_price": end_price,
                            "direction": "up" if change > 0 else "down"
                        }
                        path_evidence["data_points"] += 1
            
            # 데이터가 있으면 점수 보정
            if path_evidence["data_points"] > 0:
                path_evidence["strength"] = 0.7 + (min(path_evidence["data_points"], 3) * 0.05)
                path_evidence["robustness"] = 0.8
            
            results[path.path_id] = path_evidence
            
        return results
    
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        시나리오 추론 실행
        
        Args:
            request: 추론 요청
        
        Returns:
            InferenceResponse
        """
        start_time = time.time()
        
        # 1. 기준 시점 설정
        as_of_time = self._parse_as_of(request.as_of_date)
        
        with AsOfContext(as_of_time):
            # 2. 상태 구축
            budget = BudgetFeatures(
                max_tests=request.budget_max_tests,
                max_time_seconds=request.budget_max_time,
            )
            
            state = self.state_builder.build(
                query_text=request.query,
                as_of_time=as_of_time,
                budget=budget,
            )
            
            # 3. Trace 시작
            trace = self.trace_recorder.start_trace(state)
            
            try:
                # 4. 모드 선택
                mode_decision, mode_record = self.mode_selector.select(state)
                self.trace_recorder.record_decision(mode_record)
                
                # 5. 경로 생성
                paths, path_record = self.path_generator.generate(state, mode_decision.mode)
                self.trace_recorder.record_decision(path_record)
                
                # 6. 증거 계획 생성
                plan, plan_record = self.plan_generator.generate(paths, state, mode_decision.mode)
                self.trace_recorder.record_decision(plan_record)
                
                # 7. 증거 수집 (현재 stub - 실제로는 feature store에서 조회)
                evidence_results = self._collect_evidence(paths, plan)
                
                # 8. 융합 점수화
                scenarios, score_record = self.fusion_scorer.score(
                    paths, evidence_results, state, top_k=3
                )
                self.trace_recorder.record_decision(score_record)
                
                # 결과 변환
                scenario_results = [
                    self._convert_scenario(s) for s in scenarios
                ]
                
                # 실패 플래그 수집
                failure_flags = self._collect_failure_flags(state, evidence_results)
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                failure_flags = ["inference_error"]
                scenario_results = []
                mode_decision = ModeDecision(
                    mode=mode_decision.mode if 'mode_decision' in dir() else "unknown",
                    rationale="Error",
                    confidence=0.0,
                    estimated_cost=0.0,
                )
            
            # 9. Trace 종료
            elapsed = time.time() - start_time
            final_trace = self.trace_recorder.finalize_trace(
                total_cost=self._estimate_cost(elapsed)
            )
            
            # 통계 업데이트
            self._total_inferences += 1
            self._total_time += elapsed
            
            # 10. 응답 생성
            response = InferenceResponse(
                trace_id=final_trace.trace_id,
                query=request.query,
                as_of_time=as_of_time,
                mode_selected=mode_decision.mode.value if hasattr(mode_decision.mode, 'value') else str(mode_decision.mode),
                scenarios=scenario_results,
                total_cost=final_trace.total_cost,
                processing_time_seconds=elapsed,
                failure_flags=failure_flags,
                trace_detail=self._get_trace_detail(final_trace) if request.include_trace else None,
            )
            
            return response
    
    def get_trace(self, trace_id: str) -> Optional[ScenarioTrace]:
        """Trace 조회"""
        return self.trace_recorder.get_trace(trace_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """서비스 통계"""
        return {
            "total_inferences": self._total_inferences,
            "avg_processing_time": self._total_time / max(1, self._total_inferences),
        }
    
    def _parse_as_of(self, date_str: Optional[str]) -> datetime:
        """기준일 파싱"""
        if not date_str:
            return datetime.now()
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return datetime.now()
    
    
    def _convert_scenario(self, scored: ScoredScenario) -> ScenarioResult:
        """ScoredScenario → ScenarioResult 변환"""
        # 방향 결정
        direction = "neutral"
        if scored.final_score > 0.6:
            direction = "positive"
        elif scored.final_score < 0.4:
            direction = "negative"
        
        return ScenarioResult(
            scenario_id=scored.scenario_id,
            description=f"Path {scored.path.path_id}: {' → '.join(scored.path.nodes)}",
            direction=direction,
            confidence=scored.confidence,
            evidence_summary=scored.evidence_result,
        )
    
    def _collect_failure_flags(
        self,
        state: ScenarioState,
        evidence_results: Dict[str, Any]
    ) -> List[str]:
        """실패 플래그 수집"""
        flags = []
        
        # 증거 약함
        if state.evidence_coverage.coverage_ratio < 0.3:
            flags.append("evidence_weak")
        
        # 충돌 심함
        if state.kg_features.conflict_edge_ratio > 0.2:
            flags.append("conflict_severe")
        
        return flags
    
    def _estimate_cost(self, elapsed: float) -> float:
        """비용 추정"""
        # 시간 기반 비용 (초당 1 단위)
        return elapsed * 1.0
    
    def _get_trace_detail(self, trace: ScenarioTrace) -> Dict[str, Any]:
        """Trace 상세 정보"""
        return {
            "decision_records": [
                r.model_dump() for r in trace.decision_records
            ],
            "evidence_trace": trace.evidence_trace,
            "final_scenarios": trace.final_scenarios,
        }
