"""
Extended Policy Learner
Evidence + Regime 가중치를 포함한 확장 정책 학습기

8단계: Learning/Policy 연결 강화
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import random

from src.learning.extended_models import (
    ExtendedTrainingSample,
    ExtendedPolicyConfig,
    PolicyOptimizationTarget,
    PolicyEvaluationResult,
    ConclusionOutcomePair,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """최적화 결과"""
    success: bool
    iterations: int
    best_config: ExtendedPolicyConfig
    best_score: float
    improvement: float
    optimization_history: List[Dict] = field(default_factory=list)


class ExtendedPolicyLearner:
    """
    확장된 정책 학습기
    
    탐색 대상:
    - 기존: EES weights, PCS weights, Thresholds
    - 신규: Evidence weights, Regime applicability, Conclusion thresholds
    
    알고리즘:
    - Grid Search (작은 파라미터 공간)
    - Random Search (큰 파라미터 공간)
    - Bayesian Optimization (향후 확장)
    """
    
    def __init__(
        self,
        current_config: Optional[ExtendedPolicyConfig] = None,
        validation_samples: Optional[List[ExtendedTrainingSample]] = None,
    ):
        """
        Args:
            current_config: 현재 적용 중인 정책
            validation_samples: 검증용 샘플
        """
        self.current_config = current_config or ExtendedPolicyConfig(version="v1.0")
        self.validation_samples = validation_samples or []
        
        # 최적화 대상
        self._targets: List[PolicyOptimizationTarget] = []
        self._build_default_targets()
    
    def _build_default_targets(self) -> None:
        """기본 최적화 대상 구축"""
        self._targets = [
            # EES weights
            PolicyOptimizationTarget(
                parameter_name="ees_weights.domain",
                parameter_type="weight",
                current_value=0.4,
                min_value=0.2,
                max_value=0.6,
                step_size=0.05,
                sensitivity=0.7,
            ),
            PolicyOptimizationTarget(
                parameter_name="ees_weights.personal",
                parameter_type="weight",
                current_value=0.2,
                min_value=0.1,
                max_value=0.4,
                step_size=0.05,
                sensitivity=0.5,
            ),
            
            # Evidence weights (신규)
            PolicyOptimizationTarget(
                parameter_name="evidence_weights.pro_score",
                parameter_type="weight",
                current_value=0.4,
                min_value=0.2,
                max_value=0.6,
                step_size=0.05,
                sensitivity=0.8,
            ),
            PolicyOptimizationTarget(
                parameter_name="evidence_weights.con_score",
                parameter_type="weight",
                current_value=0.3,
                min_value=0.1,
                max_value=0.5,
                step_size=0.05,
                sensitivity=0.6,
            ),
            
            # Regime applicability (신규)
            PolicyOptimizationTarget(
                parameter_name="regime_applicability.risk_on",
                parameter_type="weight",
                current_value=1.0,
                min_value=0.5,
                max_value=1.5,
                step_size=0.1,
                sensitivity=0.5,
            ),
            PolicyOptimizationTarget(
                parameter_name="regime_applicability.risk_off",
                parameter_type="weight",
                current_value=1.0,
                min_value=0.5,
                max_value=1.5,
                step_size=0.1,
                sensitivity=0.6,
            ),
            
            # Conclusion thresholds (신규)
            PolicyOptimizationTarget(
                parameter_name="conclusion_thresholds.strong_positive",
                parameter_type="threshold",
                current_value=0.7,
                min_value=0.5,
                max_value=0.9,
                step_size=0.05,
                sensitivity=0.5,
            ),
            PolicyOptimizationTarget(
                parameter_name="conclusion_thresholds.min_confidence",
                parameter_type="threshold",
                current_value=0.4,
                min_value=0.2,
                max_value=0.6,
                step_size=0.05,
                sensitivity=0.7,
            ),
        ]
    
    def add_target(self, target: PolicyOptimizationTarget) -> None:
        """최적화 대상 추가"""
        self._targets.append(target)
    
    def set_validation_samples(self, samples: List[ExtendedTrainingSample]) -> None:
        """검증 샘플 설정"""
        self.validation_samples = samples
    
    def evaluate_config(
        self,
        config: ExtendedPolicyConfig,
        samples: Optional[List[ExtendedTrainingSample]] = None,
    ) -> float:
        """
        정책 설정 평가
        
        Returns:
            점수 (0~1, 높을수록 좋음)
        """
        samples = samples or self.validation_samples
        
        if not samples:
            return 0.5  # 기본값
        
        correct = 0
        total = 0
        calibration_error = 0.0
        
        for sample in samples:
            if sample.was_correct is None:
                continue
            
            total += 1
            
            # 정확도 계산
            if sample.was_correct:
                correct += 1
            
            # Calibration error
            if sample.conclusion_confidence is not None:
                expected_acc = sample.conclusion_confidence
                actual_acc = 1.0 if sample.was_correct else 0.0
                calibration_error += abs(expected_acc - actual_acc)
        
        if total == 0:
            return 0.5
        
        accuracy = correct / total
        avg_calibration_error = calibration_error / total
        
        # 최종 점수 (accuracy 70%, calibration 30%)
        score = accuracy * 0.7 + (1 - avg_calibration_error) * 0.3
        
        return score
    
    def optimize_random_search(
        self,
        n_iterations: int = 50,
        samples: Optional[List[ExtendedTrainingSample]] = None,
    ) -> OptimizationResult:
        """
        Random Search 최적화
        
        Args:
            n_iterations: 탐색 반복 횟수
            samples: 검증 샘플
        """
        samples = samples or self.validation_samples
        
        best_config = self.current_config.model_copy()
        best_score = self.evaluate_config(best_config, samples)
        history = []
        
        for i in range(n_iterations):
            # 랜덤 설정 생성
            candidate = self._generate_random_config()
            
            # 평가
            score = self.evaluate_config(candidate, samples)
            
            history.append({
                "iteration": i,
                "score": score,
                "improved": score > best_score,
            })
            
            if score > best_score:
                best_score = score
                best_config = candidate
                logger.info(f"Iteration {i}: New best score = {score:.4f}")
        
        improvement = best_score - self.evaluate_config(self.current_config, samples)
        
        return OptimizationResult(
            success=improvement > 0,
            iterations=n_iterations,
            best_config=best_config,
            best_score=best_score,
            improvement=improvement,
            optimization_history=history,
        )
    
    def _generate_random_config(self) -> ExtendedPolicyConfig:
        """랜덤 설정 생성"""
        config = self.current_config.model_copy(deep=True)
        
        # 각 타겟에 대해 랜덤 값 설정
        for target in self._targets:
            parts = target.parameter_name.split(".")
            
            if len(parts) == 2:
                dict_name, key = parts
                current_dict = getattr(config, dict_name, {})
                
                if isinstance(current_dict, dict) and key in current_dict:
                    # 랜덤 값 생성 (범위 내)
                    new_value = random.uniform(target.min_value, target.max_value)
                    # Step size에 맞게 반올림
                    steps = round((new_value - target.min_value) / target.step_size)
                    new_value = target.min_value + steps * target.step_size
                    new_value = min(target.max_value, max(target.min_value, new_value))
                    
                    current_dict[key] = new_value
        
        # 버전 업데이트
        config.version = f"v{random.randint(100, 999)}"
        
        return config
    
    def optimize_grid_search(
        self,
        target_params: List[str],
        samples: Optional[List[ExtendedTrainingSample]] = None,
    ) -> OptimizationResult:
        """
        Grid Search 최적화 (선택된 파라미터만)
        
        Args:
            target_params: 최적화할 파라미터 이름 리스트
            samples: 검증 샘플
        """
        samples = samples or self.validation_samples
        
        # 대상 파라미터 필터링
        targets = [t for t in self._targets if t.parameter_name in target_params]
        
        if not targets:
            return OptimizationResult(
                success=False,
                iterations=0,
                best_config=self.current_config,
                best_score=0,
                improvement=0,
            )
        
        # 그리드 생성 (간소화: 첫 2개 타겟만)
        targets = targets[:2]
        
        best_config = self.current_config.model_copy()
        best_score = self.evaluate_config(best_config, samples)
        history = []
        iterations = 0
        
        # 간단한 2D 그리드 탐색
        t1 = targets[0]
        t2 = targets[1] if len(targets) > 1 else None
        
        v1_range = self._get_value_range(t1)
        v2_range = self._get_value_range(t2) if t2 else [0]
        
        for v1 in v1_range:
            for v2 in v2_range:
                config = self._apply_values(t1, v1, t2, v2)
                score = self.evaluate_config(config, samples)
                
                history.append({
                    "iteration": iterations,
                    "params": {t1.parameter_name: v1, t2.parameter_name if t2 else "": v2},
                    "score": score,
                })
                
                if score > best_score:
                    best_score = score
                    best_config = config
                
                iterations += 1
        
        improvement = best_score - self.evaluate_config(self.current_config, samples)
        
        return OptimizationResult(
            success=improvement > 0,
            iterations=iterations,
            best_config=best_config,
            best_score=best_score,
            improvement=improvement,
            optimization_history=history,
        )
    
    def _get_value_range(self, target: PolicyOptimizationTarget) -> List[float]:
        """값 범위 생성"""
        values = []
        v = target.min_value
        while v <= target.max_value:
            values.append(round(v, 4))
            v += target.step_size
        return values
    
    def _apply_values(
        self,
        t1: PolicyOptimizationTarget,
        v1: float,
        t2: Optional[PolicyOptimizationTarget],
        v2: float,
    ) -> ExtendedPolicyConfig:
        """값 적용"""
        config = self.current_config.model_copy(deep=True)
        
        self._set_param_value(config, t1.parameter_name, v1)
        if t2:
            self._set_param_value(config, t2.parameter_name, v2)
        
        return config
    
    def _set_param_value(
        self,
        config: ExtendedPolicyConfig,
        param_name: str,
        value: float,
    ) -> None:
        """파라미터 값 설정"""
        parts = param_name.split(".")
        if len(parts) == 2:
            dict_name, key = parts
            d = getattr(config, dict_name, None)
            if isinstance(d, dict):
                d[key] = value
    
    def generate_evaluation_report(
        self,
        config: ExtendedPolicyConfig,
        samples: List[ExtendedTrainingSample],
        period_start: datetime,
        period_end: datetime,
    ) -> PolicyEvaluationResult:
        """정책 평가 리포트 생성"""
        total = 0
        correct = 0
        
        regime_correct: Dict[str, Tuple[int, int]] = {}
        conf_correct: Dict[str, Tuple[int, int]] = {}  # 신뢰도 구간별
        
        for sample in samples:
            if sample.was_correct is None:
                continue
            
            total += 1
            if sample.was_correct:
                correct += 1
            
            # Regime별 집계
            regime = sample.regime_snapshot.get("primary_regime", "unknown")
            if regime not in regime_correct:
                regime_correct[regime] = (0, 0)
            rc, rt = regime_correct[regime]
            regime_correct[regime] = (rc + (1 if sample.was_correct else 0), rt + 1)
            
            # Confidence 구간별 집계
            conf = sample.conclusion_confidence or 0.5
            if conf < 0.4:
                bucket = "low"
            elif conf < 0.7:
                bucket = "medium"
            else:
                bucket = "high"
            
            if bucket not in conf_correct:
                conf_correct[bucket] = (0, 0)
            cc, ct = conf_correct[bucket]
            conf_correct[bucket] = (cc + (1 if sample.was_correct else 0), ct + 1)
        
        accuracy = correct / total if total > 0 else 0
        
        # Regime별 accuracy
        accuracy_by_regime = {
            k: v[0] / v[1] if v[1] > 0 else 0
            for k, v in regime_correct.items()
        }
        
        # Confidence 구간별 precision
        precision_by_confidence = {
            k: v[0] / v[1] if v[1] > 0 else 0
            for k, v in conf_correct.items()
        }
        
        # Calibration score
        calibration_score = self._calculate_calibration_score(samples)
        
        # 권장 사항
        recommendations = self._generate_recommendations(
            accuracy, accuracy_by_regime, precision_by_confidence
        )
        
        return PolicyEvaluationResult(
            policy_version=config.version,
            period_start=period_start,
            period_end=period_end,
            total_samples=total,
            correct_samples=correct,
            accuracy=accuracy,
            precision_by_confidence=precision_by_confidence,
            accuracy_by_regime=accuracy_by_regime,
            calibration_score=calibration_score,
            recommendations=recommendations,
        )
    
    def _calculate_calibration_score(
        self,
        samples: List[ExtendedTrainingSample],
    ) -> float:
        """Calibration score 계산"""
        if not samples:
            return 0.5
        
        total_error = 0.0
        count = 0
        
        for sample in samples:
            if sample.was_correct is None or sample.conclusion_confidence is None:
                continue
            
            expected = sample.conclusion_confidence
            actual = 1.0 if sample.was_correct else 0.0
            total_error += abs(expected - actual)
            count += 1
        
        if count == 0:
            return 0.5
        
        avg_error = total_error / count
        return 1.0 - avg_error
    
    def _generate_recommendations(
        self,
        accuracy: float,
        accuracy_by_regime: Dict[str, float],
        precision_by_confidence: Dict[str, float],
    ) -> List[str]:
        """권장 사항 생성"""
        recommendations = []
        
        if accuracy < 0.5:
            recommendations.append("전체 정확도가 50% 미만입니다. 기본 EES 가중치 재검토 필요")
        
        # 레짐별 불균형
        if accuracy_by_regime:
            min_regime = min(accuracy_by_regime.items(), key=lambda x: x[1])
            max_regime = max(accuracy_by_regime.items(), key=lambda x: x[1])
            
            if max_regime[1] - min_regime[1] > 0.2:
                recommendations.append(
                    f"'{min_regime[0]}' 레짐에서 성능 저하. "
                    f"regime_applicability 조정 권장"
                )
        
        # Confidence calibration
        if precision_by_confidence:
            high_conf = precision_by_confidence.get("high", 0)
            low_conf = precision_by_confidence.get("low", 0)
            
            if high_conf < 0.7:
                recommendations.append(
                    "높은 confidence에서도 정확도가 낮음. "
                    "conclusion_thresholds 상향 조정 권장"
                )
            
            if low_conf > 0.5:
                recommendations.append(
                    "낮은 confidence에서도 정확도가 높음. "
                    "min_confidence 하향 조정 고려"
                )
        
        if not recommendations:
            recommendations.append("현재 정책 설정이 적절합니다.")
        
        return recommendations
    
    def get_stats(self) -> Dict:
        """통계"""
        return {
            "optimization_targets": len(self._targets),
            "validation_samples": len(self.validation_samples),
            "current_config_version": self.current_config.version,
        }
