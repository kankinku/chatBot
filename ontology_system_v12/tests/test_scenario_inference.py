"""
Scenario Inference 통합 테스트

테스트 범위:
- Layer 0: AsOfContextGuard, SnapshotResolver
- Layer 1: ScenarioStateBuilder, ModeSelector, PathGenerator, PlanGenerator, FusionScorer
- Layer 2: TraceRecorder, DelayedVerifier, Scorecard, DatasetBuilder
- Layer 3: Learners (Mode, Path, Plan, Weight), SafetyGate
"""
import pytest
import tempfile
import os
from datetime import datetime, timedelta


class TestLayer0Guards:
    """Layer 0: 공통 계약 테스트"""
    
    def test_as_of_context_guard_basic(self):
        """AsOfContextGuard 기본 동작"""
        from src.scenario_inference.layer0_guards import AsOfContextGuard, AsOfContext
        
        guard = AsOfContextGuard(enable_strict_mode=False)
        
        # 컨텍스트 없이 조회 시 False
        assert guard.validate_query("timeseries", "SOFR") == False
        
        # 컨텍스트 설정 후 조회 허용
        with AsOfContext(datetime(2024, 12, 1)):
            result = guard.validate_query("timeseries", "SOFR", datetime(2024, 11, 15))
            assert result == True
    
    def test_as_of_context_guard_leak_detection(self):
        """미래 데이터 누수 감지"""
        from src.scenario_inference.layer0_guards import AsOfContextGuard, AsOfContext
        
        guard = AsOfContextGuard(enable_strict_mode=False)
        
        with AsOfContext(datetime(2024, 12, 1)):
            # 미래 데이터 접근 시도
            result = guard.validate_query("timeseries", "SOFR", datetime(2024, 12, 15))
            assert result == False
        
        # 위반 리포트
        report = guard.generate_leak_report()
        assert report["total_violations"] > 0
        assert report["is_clean"] == False
    
    def test_snapshot_resolver(self):
        """SnapshotResolver 테스트"""
        from src.scenario_inference.layer0_guards import SnapshotResolver
        
        resolver = SnapshotResolver()
        as_of = datetime(2024, 12, 1)
        
        # 모든 스냅샷 해석
        snapshots = resolver.resolve_all(as_of)
        
        assert "kg" in snapshots
        assert "timeseries" in snapshots
        assert "evidence" in snapshots
        assert snapshots["kg"].as_of_time == as_of


class TestLayer1Inference:
    """Layer 1: Scenario Inference Core 테스트"""
    
    def test_scenario_state_builder(self):
        """ScenarioStateBuilder 테스트"""
        from src.scenario_inference.layer1_inference import ScenarioStateBuilder
        
        builder = ScenarioStateBuilder()
        
        state = builder.build(
            query_text="금리가 오르면 성장주는 어떻게 될까요?",
            as_of_time=datetime(2024, 12, 1),
            query_id="Q001",
        )
        
        assert state.query_text == "금리가 오르면 성장주는 어떻게 될까요?"
        assert "금리" in state.query_features.target_entities
        assert state.query_features.has_condition == True
    
    def test_inference_mode_selector(self):
        """InferenceModeSelector 테스트"""
        from src.scenario_inference.layer1_inference import (
            ScenarioStateBuilder, InferenceModeSelector
        )
        from src.scenario_inference.contracts import InferenceMode
        
        builder = ScenarioStateBuilder()
        state = builder.build("테스트 질의", datetime(2024, 12, 1))
        
        selector = InferenceModeSelector()
        decision, record = selector.select(state)
        
        assert decision.mode in list(InferenceMode)
        assert record.decision_type.value == "mode"
        assert len(record.candidates) > 0
    
    def test_path_candidate_generator(self):
        """PathCandidateGenerator 테스트"""
        from src.scenario_inference.layer1_inference import (
            ScenarioStateBuilder, PathCandidateGenerator, InferenceMode
        )
        
        builder = ScenarioStateBuilder()
        state = builder.build("테스트", datetime(2024, 12, 1))
        
        generator = PathCandidateGenerator()
        paths, record = generator.generate(state, InferenceMode.FULL)
        
        assert len(paths) > 0
        assert record.decision_type.value == "path_select"
    
    def test_evidence_plan_generator(self):
        """EvidencePlanGenerator 테스트"""
        from src.scenario_inference.layer1_inference import (
            ScenarioStateBuilder, PathCandidateGenerator,
            EvidencePlanGenerator, InferenceMode
        )
        
        builder = ScenarioStateBuilder()
        state = builder.build("테스트", datetime(2024, 12, 1))
        
        generator = PathCandidateGenerator()
        paths, _ = generator.generate(state, InferenceMode.GRAPH_EVIDENCE)
        
        plan_gen = EvidencePlanGenerator()
        plan, record = plan_gen.generate(paths, state, InferenceMode.GRAPH_EVIDENCE)
        
        assert plan.plan_type in ["minimal", "standard", "comprehensive", "robust", "quick_check"]
        assert record.decision_type.value == "plan_select"
    
    def test_fusion_scorer(self):
        """FusionScorer 테스트"""
        from src.scenario_inference.layer1_inference import (
            ScenarioStateBuilder, PathCandidateGenerator, FusionScorer, InferenceMode
        )
        
        builder = ScenarioStateBuilder()
        state = builder.build("테스트", datetime(2024, 12, 1))
        
        generator = PathCandidateGenerator()
        paths, _ = generator.generate(state, InferenceMode.FULL)
        
        scorer = FusionScorer()
        evidence_results = {p.path_id: {"strength": 0.6, "robustness": 0.5} for p in paths}
        scored, record = scorer.score(paths, evidence_results, state, top_k=3)
        
        assert len(scored) <= 3
        assert record.decision_type.value == "fusion_score"


class TestLayer2Trace:
    """Layer 2: Trace & Verification 테스트"""
    
    def test_scenario_trace_recorder(self):
        """ScenarioTraceRecorder 테스트"""
        from src.scenario_inference.layer2_trace import ScenarioTraceRecorder
        from src.scenario_inference.layer1_inference import ScenarioStateBuilder
        from src.scenario_inference.contracts import DecisionRecord, DecisionType
        
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ScenarioTraceRecorder(
                db_path=os.path.join(tmpdir, "traces.db")
            )
            
            builder = ScenarioStateBuilder()
            state = builder.build("테스트", datetime(2024, 12, 1), query_id="Q001")
            
            # 트레이스 시작
            trace = recorder.start_trace(state)
            
            # 결정 기록
            decision = DecisionRecord(
                decision_type=DecisionType.MODE,
                chosen_id="graph_evidence",
                confidence=0.8,
            )
            recorder.record_decision(decision)
            
            # 완료
            final_trace = recorder.finalize_trace(total_cost=5.0)
            
            assert final_trace.trace_id is not None
            assert len(final_trace.decision_records) == 1
            assert final_trace.total_cost == 5.0
            
            # 조회
            retrieved = recorder.get_trace(final_trace.trace_id)
            assert retrieved is not None
    
    def test_delayed_verifier(self):
        """DelayedVerifier 테스트"""
        from src.scenario_inference.layer2_trace import DelayedVerifier
        from src.scenario_inference.contracts import ScenarioTrace, VerificationTarget
        
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = DelayedVerifier(
                db_path=os.path.join(tmpdir, "verifications.db")
            )
            
            trace = ScenarioTrace(
                as_of_time=datetime(2024, 11, 1),
                query_id="Q001",
            )
            
            targets = [
                VerificationTarget(
                    target_id="T1", entity_id="SPY",
                    metric_type="return", expected_direction="+",
                )
            ]
            
            # 결과 조회 함수
            def outcome_fetcher(entity_id, date):
                return 0.05  # 5% 수익
            
            report = verifier.verify_now(trace, "SCEN_001", targets, outcome_fetcher)
            
            assert report.accuracy == 1.0  # 방향 적중
    
    def test_scorecard(self):
        """Scorecard 테스트"""
        from src.scenario_inference.layer2_scoring import Scorecard
        from src.scenario_inference.contracts import (
            ScenarioTrace, VerificationReport, VerificationOutcome
        )
        
        scorecard = Scorecard()
        
        trace = ScenarioTrace(
            as_of_time=datetime(2024, 12, 1),
            query_id="Q001",
            total_cost=10.0,
            failure_flags=["evidence_weak"],
        )
        
        report = VerificationReport(
            scenario_id="SCEN_001",
            trace_id=trace.trace_id,
            scenario_as_of=datetime(2024, 12, 1),
            accuracy=0.7,
            overall_outcome=VerificationOutcome.PASS_DIRECTION,
        )
        
        breakdown = scorecard.compute(trace, report)
        
        assert breakdown.accuracy == 0.7
        assert breakdown.constraint_penalty > 0  # failure_flag로 인해
        assert 0 <= breakdown.scalar_reward <= 1
    
    def test_scenario_dataset_builder(self):
        """ScenarioDatasetBuilder 테스트"""
        from src.scenario_inference.layer2_scoring import ScenarioDatasetBuilder
        from src.scenario_inference.contracts import (
            ScenarioTrace, DecisionRecord, DecisionType, RewardBreakdown
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ScenarioDatasetBuilder(
                db_path=os.path.join(tmpdir, "datasets.db")
            )
            
            # 테스트 데이터
            trace = ScenarioTrace(
                as_of_time=datetime(2024, 12, 1),
                query_id="Q001",
            )
            trace.add_decision(DecisionRecord(
                decision_type=DecisionType.MODE,
                chosen_id="full",
            ))
            
            reward = RewardBreakdown(
                trace_id=trace.trace_id,
                accuracy=0.7, calibration=0.6, robustness=0.5,
            )
            reward.compute_scalar()
            
            datasets = builder.build_from_traces([trace], [reward])
            
            assert DecisionType.MODE in datasets
            assert len(datasets[DecisionType.MODE].samples) == 1


class TestLayer3Learners:
    """Layer 3: Decision Learners 테스트"""
    
    def test_mode_learner(self):
        """ModeLearner 테스트"""
        from src.scenario_inference.layer3_learners import ModeLearner
        from src.scenario_inference.contracts import InferenceMode
        
        learner = ModeLearner()
        
        # 학습 데이터
        samples = [
            {"chosen_id": "full", "reward": 0.8},
            {"chosen_id": "light", "reward": 0.3},
            {"chosen_id": "full", "reward": 0.9},
        ]
        
        learner.update(samples)
        
        policy = learner.get_policy()
        assert "full" in policy
        assert policy["full"] > policy.get("light", 0)
    
    def test_path_selector_learner(self):
        """PathSelectorLearner 테스트"""
        from src.scenario_inference.layer3_learners import PathSelectorLearner
        
        learner = PathSelectorLearner()
        
        samples = [
            {"chosen_id": "PATH_D_1", "reward": 0.8},  # domain
            {"chosen_id": "PATH_P_1", "reward": 0.2},  # personal
        ]
        
        learner.update(samples)
        
        # domain 경로는 보너스
        assert learner.get_bonus("PATH_D_2") > 0
        # personal 경로는 페널티
        assert learner.get_penalty("PATH_P_2") > 0
    
    def test_evidence_planner_learner(self):
        """EvidencePlannerLearner 테스트"""
        from src.scenario_inference.layer3_learners import EvidencePlannerLearner
        from src.scenario_inference.contracts import ScenarioState, BudgetFeatures
        
        learner = EvidencePlannerLearner()
        
        samples = [
            {"chosen_id": "standard", "reward": 0.7, "reward_breakdown": {"cost_efficiency": 0.6}},
            {"chosen_id": "comprehensive", "reward": 0.5, "reward_breakdown": {"cost_efficiency": 0.3}},
        ]
        
        learner.update(samples)
        
        # 낮은 비용 제약
        state = ScenarioState(
            as_of_time=datetime(2024, 12, 1),
            budget=BudgetFeatures(max_tests=5),
        )
        
        selected = learner.select_plan(state)
        assert selected in ["minimal", "quick_check", "standard"]
    
    def test_weight_learner(self):
        """WeightLearner 테스트"""
        from src.scenario_inference.layer3_learners import WeightLearner
        
        learner = WeightLearner()
        
        # 충분한 샘플
        samples = [{"reward": 0.7} for _ in range(25)]
        
        learner.update(samples)
        
        policy = learner.get_policy()
        assert "ees_domain" in policy
        assert sum(policy.values()) > 0
    
    def test_safety_gate(self):
        """SafetyGate 테스트"""
        from src.scenario_inference.layer3_learners import SafetyGate, PolicyBundle
        
        gate = SafetyGate(min_samples=10)
        
        baseline = PolicyBundle(version="v1.0", validation_score=0.5)
        candidate = PolicyBundle(version="v1.1", validation_score=0.55)
        
        # 검증 결과
        validation_results = [
            {"accuracy": 0.7, "calibration": 0.8, "reward": 0.6}
            for _ in range(20)
        ]
        
        decision = gate.evaluate(candidate, baseline, validation_results)
        
        assert decision.action in ["activate", "reject", "rollback"]


class TestEndToEndFlow:
    """엔드투엔드 흐름 테스트"""
    
    def test_full_inference_pipeline(self):
        """전체 추론 파이프라인 테스트"""
        from src.scenario_inference.layer0_guards import AsOfContext
        from src.scenario_inference.layer1_inference import (
            ScenarioStateBuilder, InferenceModeSelector,
            PathCandidateGenerator, EvidencePlanGenerator, FusionScorer
        )
        from src.scenario_inference.contracts import InferenceMode
        
        with AsOfContext(datetime(2024, 12, 1)):
            # 1. 상태 구축
            builder = ScenarioStateBuilder()
            state = builder.build(
                query_text="금리 인상 시 주식시장 영향",
                as_of_time=datetime(2024, 12, 1),
            )
            
            # 2. 모드 선택
            mode_selector = InferenceModeSelector()
            mode_decision, mode_record = mode_selector.select(state)
            
            # 3. 경로 생성
            path_gen = PathCandidateGenerator()
            paths, path_record = path_gen.generate(state, mode_decision.mode)
            
            # 4. 증거 계획
            plan_gen = EvidencePlanGenerator()
            plan, plan_record = plan_gen.generate(paths, state, mode_decision.mode)
            
            # 5. 점수화
            scorer = FusionScorer()
            evidence = {p.path_id: {"strength": 0.6} for p in paths}
            scenarios, score_record = scorer.score(paths, evidence, state)
            
            # 검증
            assert len(scenarios) > 0
            assert mode_record.decision_type.value == "mode"
            assert path_record.decision_type.value == "path_select"
            assert plan_record.decision_type.value == "plan_select"
            assert score_record.decision_type.value == "fusion_score"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
