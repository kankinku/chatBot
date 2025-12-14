"""
Advisor System Step 5-8 통합 테스트

테스트 범위:
- Step 5: Incremental Update Orchestrator
- Step 6: Replay/Backtest
- Step 7: Scenario Simulation
- Step 8: Extended Learning/Policy
"""
import pytest
import tempfile
import os
from datetime import datetime, timedelta


class TestStep5Orchestration:
    """Step 5: Incremental Update Orchestrator 테스트"""
    
    def test_dependency_graph_manager(self):
        """DependencyGraphManager 테스트"""
        from src.orchestration.dependency_graph_manager import (
            DependencyGraphManager, NodeType
        )
        
        manager = DependencyGraphManager()
        
        # 의존성 추가: SOFR → SOFR_ROC_30D → evidence_001
        manager.add_dependency(
            "SOFR_ROC_30D", NodeType.FEATURE,
            "SOFR", NodeType.SERIES
        )
        manager.add_dependency(
            "evidence_001", NodeType.EVIDENCE,
            "SOFR_ROC_30D", NodeType.FEATURE
        )
        
        # 영향 범위 산출
        affected = manager.get_affected_downstream("SOFR")
        
        assert "SOFR_ROC_30D" in affected[NodeType.FEATURE]
        assert "evidence_001" in affected[NodeType.EVIDENCE]
    
    def test_cache_invalidator(self):
        """CacheInvalidator 테스트"""
        from src.orchestration.dependency_graph_manager import (
            DependencyGraphManager, NodeType
        )
        from src.orchestration.cache_invalidator import CacheInvalidator
        
        manager = DependencyGraphManager()
        manager.add_dependency("SOFR_ROC_30D", NodeType.FEATURE, "SOFR", NodeType.SERIES)
        
        invalidator = CacheInvalidator(manager)
        
        # 무효화
        event = invalidator.invalidate(["SOFR"])
        
        assert "SOFR_ROC_30D" in event.affected_features
        assert len(invalidator.get_pending_features()) > 0
    
    def test_incremental_orchestrator(self):
        """IncrementalUpdateOrchestrator 테스트"""
        from src.orchestration.dependency_graph_manager import (
            DependencyGraphManager, NodeType
        )
        from src.orchestration.cache_invalidator import CacheInvalidator
        from src.orchestration.incremental_orchestrator import (
            IncrementalUpdateOrchestrator
        )
        
        manager = DependencyGraphManager()
        manager.add_dependency("SOFR_ROC_30D", NodeType.FEATURE, "SOFR", NodeType.SERIES)
        
        invalidator = CacheInvalidator(manager)
        orchestrator = IncrementalUpdateOrchestrator(manager, invalidator)
        
        # Ingestion 결과 처리
        result = orchestrator.process_ingestion_result(["SOFR", "VIX"])
        
        assert result.success
        assert result.series_count == 2


class TestStep6Replay:
    """Step 6: Replay/Backtest 테스트"""
    
    def test_snapshot_manager(self):
        """SnapshotManager 테스트"""
        from src.replay.snapshot_manager import SnapshotManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(
                db_path=os.path.join(tmpdir, "snapshots.db")
            )
            
            # 스냅샷 생성
            snapshot = manager.create_snapshot(
                as_of=datetime(2024, 12, 14),
                metadata={"test": True}
            )
            
            assert snapshot.snapshot_id is not None
            
            # 조회
            retrieved = manager.get_snapshot(snapshot.snapshot_id)
            assert retrieved is not None
            assert retrieved.metadata.get("test") == True
    
    def test_replay_runner(self):
        """ReplayRunner 테스트"""
        from src.replay.snapshot_manager import SnapshotManager
        from src.replay.replay_runner import ReplayRunner
        
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_manager = SnapshotManager(
                db_path=os.path.join(tmpdir, "snapshots.db")
            )
            runner = ReplayRunner(snapshot_manager)
            
            # 단일 시점 재현
            step = runner.run_point_in_time(
                datetime(2024, 12, 14),
                query="금리가 오르면 성장주는?"
            )
            
            assert step.step_date == datetime(2024, 12, 14)
    
    def test_metrics_calculator(self):
        """MetricsCalculator 테스트"""
        from src.replay.metrics import MetricsCalculator
        
        calculator = MetricsCalculator()
        
        # 테스트 데이터
        conclusions = [
            {"date": datetime(2024, 12, 1), "direction": "+", "confidence": 0.7},
            {"date": datetime(2024, 12, 2), "direction": "-", "confidence": 0.6},
            {"date": datetime(2024, 12, 3), "direction": "+", "confidence": 0.8},
        ]
        outcomes = [
            {"date": datetime(2024, 12, 1), "return": 0.05},
            {"date": datetime(2024, 12, 2), "return": -0.03},
            {"date": datetime(2024, 12, 3), "return": 0.02},
        ]
        
        # Calibration
        calibration = calculator.calculate_calibration(conclusions, outcomes)
        assert calibration.metric_name == "calibration"
        
        # Stability
        stability = calculator.calculate_stability(conclusions)
        assert stability.metric_name == "stability"


class TestStep7Scenario:
    """Step 7: Scenario Simulation 테스트"""
    
    def test_shock_spec_registry(self):
        """ShockSpecRegistry 테스트"""
        from src.scenario.shock_spec_registry import (
            ShockSpecRegistry, ShockType
        )
        
        registry = ShockSpecRegistry()
        
        # 기본 프리셋 확인
        presets = registry.list_presets()
        assert len(presets) > 0
        
        # 프리셋 조회
        fed_hike = registry.get_preset("fed_rate_hike_50bp")
        assert fed_hike is not None
        assert len(fed_hike.shocks) > 0
        
        # Shock 생성
        shock = registry.create_shock(
            target_node="VIX",
            shock_value=0.2,
            shock_type=ShockType.RELATIVE,
            description="VIX +20%"
        )
        assert shock.shock_id is not None
    
    def test_scenario_simulator(self):
        """ScenarioSimulator 테스트"""
        from src.scenario.shock_spec_registry import ShockSpecRegistry
        from src.scenario.scenario_simulator import ScenarioSimulator
        
        registry = ShockSpecRegistry()
        simulator = ScenarioSimulator(registry)
        
        # 프리셋 시뮬레이션
        result = simulator.simulate_preset("fed_rate_hike_50bp")
        
        assert result.scenario_name == "Fed Rate Hike +50bp"
        assert len(result.applied_shocks) > 0
        assert len(result.node_impacts) > 0
        assert result.summary_direction in ["+", "-", "mixed"]
    
    def test_scenario_report(self):
        """시나리오 리포트 생성 테스트"""
        from src.scenario.shock_spec_registry import ShockSpecRegistry
        from src.scenario.scenario_simulator import ScenarioSimulator
        
        registry = ShockSpecRegistry()
        simulator = ScenarioSimulator(registry)
        
        result = simulator.simulate_preset("vix_spike")
        report = simulator.generate_report(result)
        
        assert "VIX Spike" in report
        assert "## 적용된 Shock" in report


class TestStep8ExtendedLearning:
    """Step 8: Extended Learning/Policy 테스트"""
    
    def test_extended_training_sample(self):
        """ExtendedTrainingSample 테스트"""
        from src.learning.extended_models import ExtendedTrainingSample
        
        sample = ExtendedTrainingSample(
            text="금리가 상승하면 성장주는 하락합니다.",
            task_type="reasoning",
            labels={"direction": "-"},
            source="replay",
            evidence_trace={"SOFR_ROC_30D": 0.12, "VIX_ZSCORE": 1.5},
            regime_snapshot={"primary_regime": "risk_on", "probability": 0.8},
            conclusion_label="-",
            conclusion_confidence=0.7,
            outcome_metrics={"return_30d": -0.05},
            was_correct=True,
        )
        
        assert sample.evidence_trace["SOFR_ROC_30D"] == 0.12
        assert sample.regime_snapshot["primary_regime"] == "risk_on"
        assert sample.was_correct == True
    
    def test_extended_policy_config(self):
        """ExtendedPolicyConfig 테스트"""
        from src.learning.extended_models import ExtendedPolicyConfig
        
        config = ExtendedPolicyConfig(version="v2.0")
        
        # 기존 필드
        assert "domain" in config.ees_weights
        
        # 신규 필드 (8단계)
        assert "pro_score" in config.evidence_weights
        assert "risk_on" in config.regime_applicability
        assert "strong_positive" in config.conclusion_thresholds
    
    def test_extended_policy_learner(self):
        """ExtendedPolicyLearner 테스트"""
        from src.learning.extended_models import (
            ExtendedTrainingSample, ExtendedPolicyConfig
        )
        from src.learning.extended_policy_learner import ExtendedPolicyLearner
        
        # 검증 샘플 생성
        samples = []
        for i in range(10):
            sample = ExtendedTrainingSample(
                text=f"Sample {i}",
                task_type="reasoning",
                labels={},
                source="test",
                conclusion_confidence=0.6 + (i % 3) * 0.1,
                was_correct=(i % 3 != 0),  # 66% 정확도
            )
            samples.append(sample)
        
        learner = ExtendedPolicyLearner(
            validation_samples=samples
        )
        
        # 현재 설정 평가
        score = learner.evaluate_config(learner.current_config)
        assert 0 <= score <= 1
        
        # 최적화 (작은 반복)
        result = learner.optimize_random_search(n_iterations=5)
        assert result.iterations == 5
    
    def test_extended_dataset_builder(self):
        """ExtendedDatasetBuilder 테스트"""
        from src.learning.extended_dataset_builder import ExtendedDatasetBuilder
        
        builder = ExtendedDatasetBuilder(lookahead_days=30)
        
        # 테스트 결론/결과 데이터
        conclusions = [
            {
                "date": datetime(2024, 11, 1),
                "direction": "+",
                "confidence": 0.7,
                "query": "금리가 하락하면 주식은?",
            },
            {
                "date": datetime(2024, 11, 15),
                "direction": "-",
                "confidence": 0.6,
                "query": "VIX가 급등하면?",
            },
        ]
        outcomes = [
            {"date": datetime(2024, 12, 1), "return": 0.05},
            {"date": datetime(2024, 12, 15), "return": -0.03},
        ]
        
        # 빌드
        result = builder.build_from_conclusions(conclusions, outcomes)
        
        assert result.success
        assert result.total_conclusions == 2
        assert len(result.samples) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
