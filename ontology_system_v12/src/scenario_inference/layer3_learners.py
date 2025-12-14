"""
Layer 3: Decision Learners
결정별 학습기 - 추론 과정 강화의 본체
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import random
import json
from pathlib import Path

from src.scenario_inference.contracts import (
    ScenarioState, DecisionType, InferenceMode, EvidencePlan,
    EVIDENCE_PLAN_TEMPLATES,
)

logger = logging.getLogger(__name__)


@dataclass
class PolicyBundle:
    """정책 번들"""
    version: str
    mode_policy: Dict[str, float] = field(default_factory=dict)
    path_policy: Dict[str, float] = field(default_factory=dict)
    plan_policy: Dict[str, float] = field(default_factory=dict)
    score_weights: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    validation_score: float = 0.0


class ModeLearner:
    """
    모드 학습기
    어떤 상황에서 어떤 모드를 쓰는 게 기대효용(품질-비용)이 좋은지 학습
    """
    
    def __init__(self, policy_version: str = "mode_v1"):
        self.policy_version = policy_version
        self.mode_scores: Dict[str, Dict[str, float]] = {}  # context -> mode -> score
        self._history: List[Dict] = []
    
    def update(self, samples: List[Dict]) -> None:
        """샘플로부터 학습"""
        for sample in samples:
            chosen = sample.get("chosen_id", "")
            reward = sample.get("reward", 0)
            
            # 컨텍스트 해시 (간단화)
            ctx = "default"
            
            if ctx not in self.mode_scores:
                self.mode_scores[ctx] = {}
            
            # 지수이동평균 업데이트
            alpha = 0.1
            current = self.mode_scores[ctx].get(chosen, 0.5)
            self.mode_scores[ctx][chosen] = current * (1 - alpha) + reward * alpha
            
            self._history.append({"mode": chosen, "reward": reward})
    
    def get_policy(self) -> Dict[str, float]:
        """현재 정책 반환"""
        if "default" not in self.mode_scores:
            # 기본 정책
            return {m.value: 0.5 for m in InferenceMode}
        return self.mode_scores["default"]
    
    def select_mode(self, state: ScenarioState, epsilon: float = 0.1) -> InferenceMode:
        """모드 선택 (epsilon-greedy)"""
        policy = self.get_policy()
        
        if random.random() < epsilon:
            # 탐색
            return random.choice(list(InferenceMode))
        
        # 최고 점수 모드 선택
        best_mode = max(policy.items(), key=lambda x: x[1])[0]
        return InferenceMode(best_mode)


class PathSelectorLearner:
    """
    경로 선택 학습기
    스푸리어스 경로를 벌점, 유효 패턴을 가산
    """
    
    def __init__(self, policy_version: str = "path_v1"):
        self.policy_version = policy_version
        self.path_penalties: Dict[str, float] = {}  # pattern -> penalty
        self.valid_patterns: Dict[str, float] = {}  # pattern -> bonus
    
    def update(self, samples: List[Dict]) -> None:
        """샘플로부터 학습"""
        for sample in samples:
            chosen_id = sample.get("chosen_id", "")
            reward = sample.get("reward", 0)
            
            # 경로 패턴 추출 (간단화: path_id 기반)
            pattern = self._extract_pattern(chosen_id)
            
            alpha = 0.1
            if reward > 0.6:
                # 좋은 경로
                current = self.valid_patterns.get(pattern, 0)
                self.valid_patterns[pattern] = current * (1 - alpha) + reward * alpha
            elif reward < 0.3:
                # 나쁜 경로
                current = self.path_penalties.get(pattern, 0)
                self.path_penalties[pattern] = current * (1 - alpha) + (1 - reward) * alpha
    
    def _extract_pattern(self, path_id: str) -> str:
        """경로 패턴 추출 (간단화)"""
        if "D_" in path_id:
            return "domain"
        elif "P_" in path_id:
            return "personal"
        return "unknown"
    
    def get_penalty(self, path_id: str) -> float:
        """경로 페널티 조회"""
        pattern = self._extract_pattern(path_id)
        return self.path_penalties.get(pattern, 0)
    
    def get_bonus(self, path_id: str) -> float:
        """경로 보너스 조회"""
        pattern = self._extract_pattern(path_id)
        return self.valid_patterns.get(pattern, 0)
    
    def get_policy(self) -> Dict[str, Any]:
        return {
            "penalties": self.path_penalties.copy(),
            "bonuses": self.valid_patterns.copy(),
        }


class EvidencePlannerLearner:
    """
    증거 계획 학습기
    어떤 EvidencePlan이 정보 대비 비용이 좋은가 학습
    """
    
    def __init__(self, policy_version: str = "plan_v1"):
        self.policy_version = policy_version
        self.plan_scores: Dict[str, float] = {
            "minimal": 0.5, "standard": 0.5, "comprehensive": 0.5,
            "robust": 0.5, "quick_check": 0.5,
        }
    
    def update(self, samples: List[Dict]) -> None:
        """샘플로부터 학습"""
        for sample in samples:
            chosen = sample.get("chosen_id", "standard")
            reward = sample.get("reward", 0)
            breakdown = sample.get("reward_breakdown", {})
            
            # 비용 효율성 반영
            cost_eff = breakdown.get("cost_efficiency", 0.5)
            robustness = breakdown.get("robustness", 0.5)
            
            # 종합 점수
            combined = reward * 0.5 + cost_eff * 0.3 + robustness * 0.2
            
            alpha = 0.1
            current = self.plan_scores.get(chosen, 0.5)
            self.plan_scores[chosen] = current * (1 - alpha) + combined * alpha
    
    def select_plan(self, state: ScenarioState) -> str:
        """계획 선택"""
        # 비용 제약 고려
        max_cost = state.budget.max_tests
        
        candidates = []
        for plan_name, template in EVIDENCE_PLAN_TEMPLATES.items():
            if template.max_cost <= max_cost:
                candidates.append((plan_name, self.plan_scores.get(plan_name, 0.5)))
        
        if not candidates:
            return "minimal"
        
        # 최고 점수 선택
        return max(candidates, key=lambda x: x[1])[0]
    
    def get_policy(self) -> Dict[str, float]:
        return self.plan_scores.copy()


class WeightLearner:
    """
    가중치 학습기 (PolicyLearner 확장)
    FusionScorer의 가중치/임계값을 워크포워드 기준으로 최적화
    """
    
    def __init__(self, policy_version: str = "weight_v1"):
        self.policy_version = policy_version
        self.weights = {
            "ees_domain": 0.35, "ees_personal": 0.15,
            "evidence_strength": 0.30, "evidence_robustness": 0.10,
            "path_length_penalty": 0.10,
        }
        self._history: List[Dict] = []
    
    def update(self, samples: List[Dict]) -> None:
        """샘플로부터 학습 (그리드 서치 기반)"""
        self._history.extend(samples)
        
        if len(self._history) < 20:
            return  # 충분한 샘플 필요
        
        # 최근 샘플로 최적화
        recent = self._history[-100:]
        
        best_weights = self.weights.copy()
        best_score = self._evaluate_weights(best_weights, recent)
        
        # 간단한 그리드 서치
        for key in self.weights:
            for delta in [-0.05, 0.05]:
                trial = best_weights.copy()
                trial[key] = max(0, min(1, trial[key] + delta))
                
                # 정규화
                total = sum(trial.values())
                if total > 0:
                    trial = {k: v / total for k, v in trial.items()}
                
                score = self._evaluate_weights(trial, recent)
                if score > best_score:
                    best_score = score
                    best_weights = trial
        
        self.weights = best_weights
    
    def _evaluate_weights(self, weights: Dict[str, float], 
                          samples: List[Dict]) -> float:
        """가중치 평가 (평균 보상)"""
        if not samples:
            return 0
        
        total = sum(s.get("reward", 0) for s in samples)
        return total / len(samples)
    
    def get_policy(self) -> Dict[str, float]:
        return self.weights.copy()


@dataclass
class GateDecision:
    """게이트 결정"""
    action: str  # "activate", "reject", "rollback"
    reason: str
    metrics: Dict[str, float]


class SafetyGate:
    """
    안전 게이트 / 배포 게이트
    새 정책 번들 배포 전, 워크포워드+엠바고 평가 통과 시만 활성화
    """
    
    def __init__(
        self,
        min_samples: int = 50,
        min_improvement: float = 0.02,
        max_degradation: float = 0.05,
        calibration_threshold: float = 0.3,
    ):
        self.min_samples = min_samples
        self.min_improvement = min_improvement
        self.max_degradation = max_degradation
        self.calibration_threshold = calibration_threshold
        self._history: List[Tuple[str, GateDecision]] = []
    
    def evaluate(
        self,
        candidate: PolicyBundle,
        baseline: PolicyBundle,
        validation_results: List[Dict],
    ) -> GateDecision:
        """
        정책 번들 평가
        
        Args:
            candidate: 후보 정책
            baseline: 기존 정책
            validation_results: 검증 결과
        
        Returns:
            GateDecision
        """
        if len(validation_results) < self.min_samples:
            return GateDecision(
                action="reject",
                reason=f"샘플 부족: {len(validation_results)} < {self.min_samples}",
                metrics={"sample_count": len(validation_results)},
            )
        
        # 성능 계산
        avg_accuracy = sum(r.get("accuracy", 0) for r in validation_results) / len(validation_results)
        avg_calibration = sum(r.get("calibration", 0) for r in validation_results) / len(validation_results)
        avg_reward = sum(r.get("reward", 0) for r in validation_results) / len(validation_results)
        
        metrics = {
            "accuracy": avg_accuracy,
            "calibration": avg_calibration,
            "reward": avg_reward,
            "sample_count": len(validation_results),
            "candidate_version": candidate.version,
            "baseline_version": baseline.version,
        }
        
        # 개선 확인
        improvement = candidate.validation_score - baseline.validation_score
        
        # 캘리브레이션 체크
        calibration_ok = avg_calibration > (1 - self.calibration_threshold)
        
        if improvement >= self.min_improvement and calibration_ok:
            decision = GateDecision(
                action="activate",
                reason=f"개선됨: +{improvement:.3f}, 캘리브레이션 OK",
                metrics=metrics,
            )
        elif improvement < -self.max_degradation:
            decision = GateDecision(
                action="reject",
                reason=f"성능 하락: {improvement:.3f}",
                metrics=metrics,
            )
        elif not calibration_ok:
            decision = GateDecision(
                action="reject",
                reason=f"캘리브레이션 불량: {avg_calibration:.3f}",
                metrics=metrics,
            )
        else:
            decision = GateDecision(
                action="reject",
                reason=f"개선 불충분: {improvement:.3f} < {self.min_improvement}",
                metrics=metrics,
            )
        
        self._history.append((candidate.version, decision))
        return decision
    
    def check_leak_audit(self, audit_report: Dict) -> bool:
        """누수 감사 확인"""
        is_clean = audit_report.get("is_clean", False)
        violation_rate = audit_report.get("violation_rate", 1.0)
        
        if not is_clean:
            logger.warning(f"Leak audit failed: violation_rate={violation_rate}")
            return False
        
        return True
    
    def rollback(self, current_version: str, reason: str) -> GateDecision:
        """롤백"""
        decision = GateDecision(
            action="rollback",
            reason=reason,
            metrics={"rolled_back_version": current_version},
        )
        self._history.append((current_version, decision))
        return decision
    
    def get_history(self) -> List[Tuple[str, GateDecision]]:
        return self._history.copy()
