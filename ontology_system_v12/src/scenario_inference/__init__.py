"""
Scenario Inference Core - 추론 과정 강화 시스템

Layer 0: 공통 계약 (AsOfContextGuard, SnapshotResolver)
Layer 1: Scenario Inference Core (결정 지점 분해)
Layer 2: Trace & Verification (학습 가능성)
Layer 3: Decision Learners (결정별 학습기)
"""
from .contracts import (
    ScenarioState,
    DecisionRecord,
    ScenarioTrace,
    VerificationReport,
    RewardBreakdown,
    DecisionType,
    EvidencePlan,
)
from .layer0_guards import AsOfContextGuard, SnapshotResolver
from .layer1_inference import (
    ScenarioStateBuilder,
    InferenceModeSelector,
    PathCandidateGenerator,
    EvidencePlanGenerator,
    FusionScorer,
)
from .layer2_trace import (
    ScenarioTraceRecorder,
    DelayedVerifier, 
)
from .layer2_scoring import (
    Scorecard,
    ScenarioDatasetBuilder,
)
from .layer3_learners import (
    ModeLearner,
    PathSelectorLearner,
    EvidencePlannerLearner,
    WeightLearner,
    SafetyGate,
)

__all__ = [
    # Contracts
    "ScenarioState",
    "DecisionRecord",
    "ScenarioTrace",
    "VerificationReport",
    "RewardBreakdown",
    "DecisionType",
    "EvidencePlan",
    # Layer 0
    "AsOfContextGuard",
    "SnapshotResolver",
    # Layer 1
    "ScenarioStateBuilder",
    "InferenceModeSelector",
    "PathCandidateGenerator",
    "EvidencePlanGenerator",
    "FusionScorer",
    # Layer 2
    "ScenarioTraceRecorder",
    "DelayedVerifier",
    "Scorecard",
    "ScenarioDatasetBuilder",
    # Layer 3
    "ModeLearner",
    "PathSelectorLearner",
    "EvidencePlannerLearner",
    "WeightLearner",
    "SafetyGate",
]
