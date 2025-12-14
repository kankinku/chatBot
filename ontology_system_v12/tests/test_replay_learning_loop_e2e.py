import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, ANY

from src.replay.replay_runner import ReplayRunner
from src.replay.snapshot_manager import SnapshotManager, SystemSnapshot
from src.learning.policy_optimizer import PolicyOptimizer
from src.policy.contracts import PolicyBundle
from src.policy.policy_registry import get_policy_registry
from src.reasoning.models import ReasoningConclusion, ReasoningResult, PathReasoningResult, RetrievedPath
from src.reasoning.pipeline import ReasoningPipeline
from src.replay.outcome_contracts import Outcome
from src.common.asof_context import get_context

class MockReasoningPipeline:
    def __init__(self):
        self.edge_fusion = MagicMock()
        self.graph_retrieval = MagicMock()
        self.graph_retrieval.domain = None
        self.confidence_filter = None
    
    def reason(self, query, as_of=None, context=None):
        # Verify context is set correctly
        ctx = get_context()
        assert ctx.mode == "REPLAY_SNAPSHOT_ONLY" or ctx.mode == "REPLAY" # Depending on impl
        assert ctx.as_of is not None
        
        # Return dummy verified conclusion
        return ReasoningConclusion(
            query_id="Q_MOCK",
            original_query=query or "test",
            conclusion_text="Upward trend detected",
            explanation_text="Test explanation",
            direction="+",  # Valid ReasoningDirection value
            confidence=0.8,
            strongest_path_description="A->B->C",
            evidence_summary="Strong evidence from domain"
        )
        
def _setup_components():
    # 1. Policy Registry
    registry = get_policy_registry()
    initial_policy = PolicyBundle(
        policy_id="test_policy",
        version=1,
        params={"fusion": {"w_domain": 0.5}}
    )
    registry.register_policy(initial_policy)
    registry.activate_policy("test_policy")
    
    # 2. Snapshot Manager & Repo
    snapshot_manager = MagicMock(spec=SnapshotManager)
    
    def get_snap(date):
        return SystemSnapshot(
            snapshot_id=f"SNAP_{date.isoformat()}",
            as_of=date,
            created_at=datetime.utcnow(),
            graph_relations=[{"src_id": "A", "dst_id": "B", "rel_type": "Cause"}]
        )
    snapshot_manager.get_snapshot_at.side_effect = get_snap
    snapshot_manager.create_snapshot.side_effect = get_snap
    
    # 3. Outcome Fetcher
    def fetch_outcome(date):
        return Outcome(
            target_id="T1",
            as_of=date,
            horizon_date=date,
            direction_actual=1, # UP
            magnitude_actual=0.05,
            drawdown=0.0,
            raw_value_start=100,
            raw_value_end=105
        )
        
    return registry, snapshot_manager, fetch_outcome

@pytest.fixture
def setup_components():
    return _setup_components()

def test_replay_learning_loop_e2e(setup_components):
    registry, snapshot_manager, outcome_fetcher = setup_components
    
    optimizer = PolicyOptimizer(learning_rate=0.1)
    pipeline = MockReasoningPipeline()
    
    runner = ReplayRunner(
        snapshot_manager=snapshot_manager,
        reasoning_engine=pipeline,
        outcome_fetcher=outcome_fetcher,
        optimizer=optimizer
    )
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3) # 3 steps
    
    # Run
    result = runner.run_rolling(
        start_date=start_date,
        end_date=end_date,
        step_days=1,
        query="What is the trend?",
        policy_id="test_policy",
        lookahead_days=1
    )
    
    # Verification
    
    # 1. Check if policy version evolved
    active_policy = registry.get_active_policy()
    assert active_policy.policy_id == "test_policy"
    # Starting v1, +3 updates => v4 ideally, or at least > v1
    assert active_policy.version > 1, f"Policy version should increase. Current: {active_policy.version}"
    
    # 2. Check optimizer history
    assert len(optimizer.history) == 3
    assert optimizer.history[0]["reward"] is not None
    
    # 3. Check replay result
    assert result.replay_id.startswith("RPL_")
    assert len(result.steps) == 3
    
    print("\n[Passed] Closed Loop Learning Verified: Policy evolved from v1 to", active_policy.version)
