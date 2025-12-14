"""
Extended Dataset Builder
Evidence trace, Regime snapshot을 포함한 학습 데이터 생성

8단계: Learning/Policy 연결 강화
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.learning.extended_models import (
    ExtendedTrainingSample,
    ConclusionOutcomePair,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetBuildResult:
    """데이터셋 빌드 결과"""
    success: bool
    samples: List[ExtendedTrainingSample]
    pairs: List[ConclusionOutcomePair]
    total_conclusions: int
    matched_outcomes: int
    error_message: Optional[str] = None


class ExtendedDatasetBuilder:
    """
    확장된 데이터셋 빌더
    
    Replay 결과에서 확장된 학습 샘플 생성:
    - Evidence trace 포함
    - Regime snapshot 포함
    - Outcome metrics 포함
    """
    
    def __init__(
        self,
        evidence_store: Optional[Any] = None,
        regime_store: Optional[Any] = None,
        feature_store: Optional[Any] = None,
        timeseries_repo: Optional[Any] = None,
        lookahead_days: int = 30,
    ):
        """
        Args:
            evidence_store: Evidence 저장소
            regime_store: Regime 저장소
            feature_store: Feature 저장소
            timeseries_repo: 시계열 저장소
            lookahead_days: 결과 확인 기간 (일)
        """
        self.evidence_store = evidence_store
        self.regime_store = regime_store
        self.feature_store = feature_store
        self.timeseries_repo = timeseries_repo
        self.lookahead_days = lookahead_days
    
    def build_from_replay(
        self,
        replay_steps: List[Dict],
        target_series: Optional[List[str]] = None,
    ) -> DatasetBuildResult:
        """
        Replay 결과에서 학습 데이터 생성
        
        Args:
            replay_steps: ReplayRunner의 스텝 결과
            target_series: 결과 측정 대상 시계열 (예: ["SPY"])
        
        Returns:
            DatasetBuildResult
        """
        samples = []
        pairs = []
        total_conclusions = 0
        matched_outcomes = 0
        
        for step in replay_steps:
            step_date = step.get("step_date")
            if not step_date:
                continue
            
            conclusion = step.get("conclusion")
            if not conclusion:
                continue
            
            total_conclusions += 1
            
            # Evidence trace 수집
            evidence_trace = self._get_evidence_trace(step_date)
            evidence_scores = self._get_evidence_scores(step_date)
            
            # Regime snapshot 수집
            regime_snapshot = self._get_regime_snapshot(step_date)
            
            # Outcome 수집
            outcome_date = step_date + timedelta(days=self.lookahead_days)
            outcome_metrics = self._get_outcome_metrics(
                step_date, outcome_date, target_series or ["SPY"]
            )
            
            # 정확성 평가
            was_correct = self._evaluate_correctness(
                conclusion.get("direction"),
                outcome_metrics
            )
            
            if outcome_metrics:
                matched_outcomes += 1
            
            # 샘플 생성
            sample = ExtendedTrainingSample(
                text=conclusion.get("query", ""),
                task_type="reasoning",
                labels={"direction": conclusion.get("direction")},
                source="replay",
                label_confidence=conclusion.get("confidence", 0.5),
                evidence_trace=evidence_trace,
                evidence_scores=evidence_scores,
                regime_snapshot=regime_snapshot,
                conclusion_label=conclusion.get("direction"),
                conclusion_confidence=conclusion.get("confidence"),
                conclusion_path=conclusion.get("path", []),
                outcome_metrics=outcome_metrics,
                outcome_date=outcome_date,
                was_correct=was_correct,
            )
            samples.append(sample)
            
            # 페어 생성 (결과가 있는 경우)
            if outcome_metrics and was_correct is not None:
                pair = ConclusionOutcomePair(
                    conclusion_date=step_date,
                    conclusion_direction=conclusion.get("direction", "neutral"),
                    conclusion_confidence=conclusion.get("confidence", 0.5),
                    regime_at_conclusion=regime_snapshot.get("primary_regime"),
                    evidence_scores_at_conclusion=evidence_scores,
                    outcome_date=outcome_date,
                    actual_direction=self._compute_actual_direction(outcome_metrics),
                    actual_magnitude=abs(outcome_metrics.get("return_30d", 0)),
                    is_correct=was_correct,
                    confidence_error=abs(
                        conclusion.get("confidence", 0.5) - 
                        (1.0 if was_correct else 0.0)
                    ),
                )
                pairs.append(pair)
        
        return DatasetBuildResult(
            success=True,
            samples=samples,
            pairs=pairs,
            total_conclusions=total_conclusions,
            matched_outcomes=matched_outcomes,
        )
    
    def build_from_conclusions(
        self,
        conclusions: List[Dict],
        outcomes: List[Dict],
    ) -> DatasetBuildResult:
        """
        결론과 결과 리스트에서 학습 데이터 생성
        
        Args:
            conclusions: 결론 리스트 (date, direction, confidence, ...)
            outcomes: 결과 리스트 (date, return, ...)
        """
        # 결과를 날짜별로 인덱싱
        outcome_map = {o.get("date"): o for o in outcomes if "date" in o}
        
        samples = []
        pairs = []
        
        for conclusion in conclusions:
            date = conclusion.get("date")
            if not date:
                continue
            
            # Lookahead 후 결과 조회
            outcome_date = date + timedelta(days=self.lookahead_days)
            outcome = outcome_map.get(outcome_date, {})
            
            outcome_metrics = {
                f"return_{self.lookahead_days}d": outcome.get("return", 0),
                f"volatility_{self.lookahead_days}d": outcome.get("volatility", 0),
            }
            
            was_correct = self._evaluate_correctness(
                conclusion.get("direction"),
                outcome_metrics
            )
            
            sample = ExtendedTrainingSample(
                text=conclusion.get("query", ""),
                task_type="reasoning",
                labels={"direction": conclusion.get("direction")},
                source="historical",
                label_confidence=conclusion.get("confidence", 0.5),
                evidence_trace=conclusion.get("evidence_trace", {}),
                evidence_scores=conclusion.get("evidence_scores", {}),
                regime_snapshot=conclusion.get("regime_snapshot", {}),
                conclusion_label=conclusion.get("direction"),
                conclusion_confidence=conclusion.get("confidence"),
                outcome_metrics=outcome_metrics,
                outcome_date=outcome_date,
                was_correct=was_correct,
            )
            samples.append(sample)
        
        return DatasetBuildResult(
            success=True,
            samples=samples,
            pairs=pairs,
            total_conclusions=len(conclusions),
            matched_outcomes=len([s for s in samples if s.outcome_metrics]),
        )
    
    def _get_evidence_trace(self, date: datetime) -> Dict[str, float]:
        """Evidence trace 수집"""
        trace = {}
        
        if self.feature_store:
            # 주요 Feature 값 조회
            key_features = [
                "VIX", "SOFR_ROC_30D", "SPY_ROC_20D", 
                "CPI_YOY", "VIX_ZSCORE_30D"
            ]
            for fid in key_features:
                fv = self.feature_store.get_last(fid, date)
                if fv:
                    trace[fid] = fv.value
        
        return trace
    
    def _get_evidence_scores(self, date: datetime) -> Dict[str, float]:
        """Edge별 evidence score 수집"""
        scores = {}
        
        if self.evidence_store:
            # 해당 날짜의 evidence score 조회
            results = self.evidence_store.get_scores_by_timestamp(date)
            for score in results:
                scores[score.edge_id] = score.total_score
        
        return scores
    
    def _get_regime_snapshot(self, date: datetime) -> Dict[str, Any]:
        """Regime snapshot 수집"""
        snapshot = {}
        
        if self.regime_store:
            result = self.regime_store.get_at(date)
            if result:
                snapshot = {
                    "primary_regime": result.primary_regime.value if result.primary_regime else None,
                    "probability": result.primary_probability,
                    "uncertainty": result.uncertainty,
                    "detected_regimes": {
                        k.value: v for k, v in result.detected_regimes.items()
                    } if result.detected_regimes else {},
                }
        
        return snapshot
    
    def _get_outcome_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        target_series: List[str],
    ) -> Dict[str, float]:
        """결과 메트릭 수집"""
        metrics = {}
        
        if not self.timeseries_repo:
            return metrics
        
        for series_id in target_series:
            # 시작/종료 시점 값
            start_obs = self.timeseries_repo.get_last(series_id, start_date)
            end_obs = self.timeseries_repo.get_last(series_id, end_date)
            
            if start_obs and end_obs and start_obs.value != 0:
                ret = (end_obs.value - start_obs.value) / start_obs.value
                metrics[f"return_{self.lookahead_days}d"] = ret
            
            # 변동성 (기간 내)
            observations = self.timeseries_repo.get_range(
                series_id, start_date, end_date
            )
            if len(observations) > 2:
                returns = []
                for i in range(1, len(observations)):
                    if observations[i-1].value != 0:
                        r = (observations[i].value - observations[i-1].value) / observations[i-1].value
                        returns.append(r)
                
                if returns:
                    import statistics
                    vol = statistics.stdev(returns) * (252 ** 0.5)  # 연율화
                    metrics[f"volatility_{self.lookahead_days}d"] = vol
        
        return metrics
    
    def _evaluate_correctness(
        self,
        predicted_direction: Optional[str],
        outcome_metrics: Dict[str, float],
    ) -> Optional[bool]:
        """예측 정확성 평가"""
        if not predicted_direction or not outcome_metrics:
            return None
        
        actual_return = outcome_metrics.get(f"return_{self.lookahead_days}d")
        if actual_return is None:
            return None
        
        # 방향 일치 확인
        if predicted_direction == "+":
            return actual_return > 0
        elif predicted_direction == "-":
            return actual_return < 0
        else:  # neutral
            return abs(actual_return) < 0.02  # 2% 이내면 neutral 정확
    
    def _compute_actual_direction(self, outcome_metrics: Dict[str, float]) -> str:
        """실제 방향 계산"""
        ret = outcome_metrics.get(f"return_{self.lookahead_days}d", 0)
        
        if ret > 0.02:
            return "+"
        elif ret < -0.02:
            return "-"
        else:
            return "neutral"
    
    def get_stats(self) -> Dict:
        """통계"""
        return {
            "lookahead_days": self.lookahead_days,
            "has_evidence_store": self.evidence_store is not None,
            "has_regime_store": self.regime_store is not None,
            "has_feature_store": self.feature_store is not None,
            "has_timeseries_repo": self.timeseries_repo is not None,
        }
