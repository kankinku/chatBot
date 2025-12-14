"""
Feature Builder
증분 Feature 계산

책임:
- 다양한 Feature 유형 계산 (ROC, ZScore, Spread, Correlation 등)
- 증분 계산 (윈도우 기반)
- 벡터화 연산
"""
import logging
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math

from src.shared.schemas import FeatureSpec, FeatureType, FeatureValue
from src.features.feature_spec_registry import FeatureSpecRegistry
from src.storage.timeseries_repository import TimeSeriesRepository

logger = logging.getLogger(__name__)


@dataclass
class ComputationResult:
    """계산 결과"""
    success: bool
    feature_id: str
    values: List[FeatureValue] = field(default_factory=list)
    error_message: Optional[str] = None
    computation_time_ms: float = 0.0


class FeatureBuilder:
    """
    Feature 계산기
    
    증분 계산을 지원합니다:
    - 윈도우 기반 지표는 윈도우 길이만큼 과거 포함하여 재계산
    - 특정 시점부터 특정 시점까지만 계산 가능
    """
    
    def __init__(
        self,
        spec_registry: FeatureSpecRegistry,
        ts_repository: TimeSeriesRepository,
    ):
        """
        Args:
            spec_registry: Feature 스펙 레지스트리
            ts_repository: 시계열 저장소
        """
        self.spec_registry = spec_registry
        self.ts_repository = ts_repository
        
        # 계산 함수 매핑
        self._calculators: Dict[FeatureType, Callable] = {
            FeatureType.RAW: self._calc_raw,
            FeatureType.SPREAD: self._calc_spread,
            FeatureType.ROC: self._calc_roc,
            FeatureType.ZSCORE: self._calc_zscore,
            FeatureType.YOY: self._calc_yoy,
            FeatureType.MOM: self._calc_mom,
            FeatureType.RATIO: self._calc_ratio,
            FeatureType.CORRELATION: self._calc_correlation,
            FeatureType.VOLATILITY: self._calc_volatility,
            FeatureType.PERCENTILE: self._calc_percentile,
        }
    
    def compute(
        self,
        feature_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
    ) -> ComputationResult:
        """
        Feature 계산
        
        Args:
            feature_id: Feature ID
            start: 계산 시작 시점
            end: 계산 종료 시점
            as_of: 데이터 기준 시점 (리플레이용)
        
        Returns:
            ComputationResult
        """
        import time
        start_time = time.time()
        
        spec = self.spec_registry.get(feature_id)
        if not spec:
            return ComputationResult(
                success=False,
                feature_id=feature_id,
                error_message=f"Feature spec not found: {feature_id}",
            )
        
        calculator = self._calculators.get(spec.feature_type)
        if not calculator:
            return ComputationResult(
                success=False,
                feature_id=feature_id,
                error_message=f"No calculator for type: {spec.feature_type}",
            )
        
        try:
            # 윈도우 고려하여 데이터 범위 확장
            data_start = start
            if data_start and spec.window_days > 0:
                data_start = start - timedelta(days=spec.window_days * 2)
            
            # 입력 데이터 로드
            series_data = {}
            for series_id in spec.input_series:
                observations = self.ts_repository.get_range(
                    series_id=series_id,
                    start=data_start,
                    end=end,
                    as_of=as_of,
                )
                series_data[series_id] = {
                    obs.timestamp: obs.value for obs in observations
                }
            
            # 계산
            values = calculator(spec, series_data, start, end)
            
            elapsed = (time.time() - start_time) * 1000
            
            return ComputationResult(
                success=True,
                feature_id=feature_id,
                values=values,
                computation_time_ms=elapsed,
            )
            
        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            return ComputationResult(
                success=False,
                feature_id=feature_id,
                error_message=str(e),
            )
    
    def compute_batch(
        self,
        feature_ids: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[ComputationResult]:
        """배치 계산"""
        return [self.compute(fid, start, end) for fid in feature_ids]
    
    def compute_affected(
        self,
        updated_series: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Tuple[List[ComputationResult], int]:
        """
        업데이트된 시계열에 영향받는 Feature만 계산
        
        Args:
            updated_series: 업데이트된 시계열 ID 리스트
            start: 계산 시작 시점
            end: 계산 종료 시점
        
        Returns:
            (결과 리스트, 영향받은 Feature 수)
        """
        affected_features = set()
        for series_id in updated_series:
            specs = self.spec_registry.get_by_input(series_id)
            for spec in specs:
                affected_features.add(spec.feature_id)
        
        results = self.compute_batch(list(affected_features), start, end)
        return results, len(affected_features)
    
    # =========================================================================
    # 계산 함수들
    # =========================================================================
    
    def _calc_raw(
        self,
        spec: FeatureSpec,
        data: Dict[str, Dict[datetime, float]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[FeatureValue]:
        """원시값 (변환 없음)"""
        series_id = spec.input_series[0]
        series = data.get(series_id, {})
        
        values = []
        for ts, val in sorted(series.items()):
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            values.append(FeatureValue(
                feature_id=spec.feature_id,
                timestamp=ts,
                value=val,
            ))
        return values
    
    def _calc_spread(
        self,
        spec: FeatureSpec,
        data: Dict[str, Dict[datetime, float]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[FeatureValue]:
        """스프레드 (series1 - series2)"""
        if len(spec.input_series) < 2:
            return []
        
        s1, s2 = spec.input_series[0], spec.input_series[1]
        series1 = data.get(s1, {})
        series2 = data.get(s2, {})
        
        # 공통 타임스탬프
        common_ts = set(series1.keys()) & set(series2.keys())
        
        values = []
        for ts in sorted(common_ts):
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            
            spread = series1[ts] - series2[ts]
            # bps 변환 (옵션)
            if spec.unit == "bps":
                spread *= 100
            
            values.append(FeatureValue(
                feature_id=spec.feature_id,
                timestamp=ts,
                value=spread,
            ))
        return values
    
    def _calc_roc(
        self,
        spec: FeatureSpec,
        data: Dict[str, Dict[datetime, float]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[FeatureValue]:
        """Rate of Change (변화율)"""
        series_id = spec.input_series[0]
        series = data.get(series_id, {})
        
        sorted_items = sorted(series.items())
        window = spec.window_days
        
        values = []
        for i, (ts, val) in enumerate(sorted_items):
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            
            # 윈도우 이전 값 찾기
            target_date = ts - timedelta(days=window)
            prev_val = None
            for prev_ts, pv in sorted_items[:i]:
                if prev_ts <= target_date:
                    prev_val = pv
            
            if prev_val is not None and prev_val != 0:
                roc = (val - prev_val) / prev_val
                values.append(FeatureValue(
                    feature_id=spec.feature_id,
                    timestamp=ts,
                    value=roc,
                ))
        
        return values
    
    def _calc_zscore(
        self,
        spec: FeatureSpec,
        data: Dict[str, Dict[datetime, float]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[FeatureValue]:
        """Z-Score"""
        series_id = spec.input_series[0]
        series = data.get(series_id, {})
        
        sorted_items = sorted(series.items())
        window = spec.window_days
        
        values = []
        for i, (ts, val) in enumerate(sorted_items):
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            
            # 윈도우 내 값들
            window_start = ts - timedelta(days=window)
            window_vals = [
                v for t, v in sorted_items[:i+1]
                if t >= window_start
            ]
            
            if len(window_vals) >= 2:
                mean = sum(window_vals) / len(window_vals)
                variance = sum((v - mean) ** 2 for v in window_vals) / len(window_vals)
                std = math.sqrt(variance) if variance > 0 else 1e-10
                
                zscore = (val - mean) / std
                values.append(FeatureValue(
                    feature_id=spec.feature_id,
                    timestamp=ts,
                    value=zscore,
                ))
        
        return values
    
    def _calc_yoy(
        self,
        spec: FeatureSpec,
        data: Dict[str, Dict[datetime, float]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[FeatureValue]:
        """Year over Year 변화율"""
        series_id = spec.input_series[0]
        series = data.get(series_id, {})
        
        sorted_items = sorted(series.items())
        
        values = []
        for i, (ts, val) in enumerate(sorted_items):
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            
            # 1년 전 값 찾기
            target_date = ts - timedelta(days=365)
            prev_val = None
            for prev_ts, pv in sorted_items[:i]:
                if prev_ts <= target_date:
                    prev_val = pv
            
            if prev_val is not None and prev_val != 0:
                yoy = (val - prev_val) / prev_val
                values.append(FeatureValue(
                    feature_id=spec.feature_id,
                    timestamp=ts,
                    value=yoy,
                ))
        
        return values
    
    def _calc_mom(
        self,
        spec: FeatureSpec,
        data: Dict[str, Dict[datetime, float]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[FeatureValue]:
        """Month over Month 변화율"""
        series_id = spec.input_series[0]
        series = data.get(series_id, {})
        
        sorted_items = sorted(series.items())
        
        values = []
        for i, (ts, val) in enumerate(sorted_items):
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            
            # 1개월 전 값 찾기
            target_date = ts - timedelta(days=30)
            prev_val = None
            for prev_ts, pv in sorted_items[:i]:
                if prev_ts <= target_date:
                    prev_val = pv
            
            if prev_val is not None and prev_val != 0:
                mom = (val - prev_val) / prev_val
                values.append(FeatureValue(
                    feature_id=spec.feature_id,
                    timestamp=ts,
                    value=mom,
                ))
        
        return values
    
    def _calc_ratio(
        self,
        spec: FeatureSpec,
        data: Dict[str, Dict[datetime, float]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[FeatureValue]:
        """비율 (series1 / series2)"""
        if len(spec.input_series) < 2:
            return []
        
        s1, s2 = spec.input_series[0], spec.input_series[1]
        series1 = data.get(s1, {})
        series2 = data.get(s2, {})
        
        common_ts = set(series1.keys()) & set(series2.keys())
        
        values = []
        for ts in sorted(common_ts):
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            
            if series2[ts] != 0:
                ratio = series1[ts] / series2[ts]
                values.append(FeatureValue(
                    feature_id=spec.feature_id,
                    timestamp=ts,
                    value=ratio,
                ))
        
        return values
    
    def _calc_correlation(
        self,
        spec: FeatureSpec,
        data: Dict[str, Dict[datetime, float]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[FeatureValue]:
        """롤링 상관계수"""
        if len(spec.input_series) < 2:
            return []
        
        s1, s2 = spec.input_series[0], spec.input_series[1]
        series1 = data.get(s1, {})
        series2 = data.get(s2, {})
        
        common_ts = sorted(set(series1.keys()) & set(series2.keys()))
        window = spec.window_days
        
        values = []
        for i, ts in enumerate(common_ts):
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            
            # 윈도우 내 데이터
            window_start = ts - timedelta(days=window)
            pairs = [
                (series1[t], series2[t])
                for t in common_ts[:i+1]
                if t >= window_start
            ]
            
            if len(pairs) >= 3:
                x = [p[0] for p in pairs]
                y = [p[1] for p in pairs]
                
                corr = self._pearson_correlation(x, y)
                if corr is not None:
                    values.append(FeatureValue(
                        feature_id=spec.feature_id,
                        timestamp=ts,
                        value=corr,
                    ))
        
        return values
    
    def _calc_volatility(
        self,
        spec: FeatureSpec,
        data: Dict[str, Dict[datetime, float]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[FeatureValue]:
        """롤링 표준편차 (변동성)"""
        series_id = spec.input_series[0]
        series = data.get(series_id, {})
        
        sorted_items = sorted(series.items())
        window = spec.window_days
        
        # 먼저 수익률 계산
        returns = []
        for i in range(1, len(sorted_items)):
            ts, val = sorted_items[i]
            prev_val = sorted_items[i-1][1]
            if prev_val != 0:
                ret = (val - prev_val) / prev_val
                returns.append((ts, ret))
        
        values = []
        for i, (ts, _) in enumerate(returns):
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            
            # 윈도우 내 수익률
            window_start = ts - timedelta(days=window)
            window_rets = [r for t, r in returns[:i+1] if t >= window_start]
            
            if len(window_rets) >= 2:
                mean = sum(window_rets) / len(window_rets)
                variance = sum((r - mean) ** 2 for r in window_rets) / len(window_rets)
                vol = math.sqrt(variance)
                
                # 연율화 (옵션)
                annualized = vol * math.sqrt(252)
                
                values.append(FeatureValue(
                    feature_id=spec.feature_id,
                    timestamp=ts,
                    value=annualized,
                ))
        
        return values
    
    def _calc_percentile(
        self,
        spec: FeatureSpec,
        data: Dict[str, Dict[datetime, float]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[FeatureValue]:
        """롤링 백분위"""
        series_id = spec.input_series[0]
        series = data.get(series_id, {})
        
        sorted_items = sorted(series.items())
        window = spec.window_days
        
        values = []
        for i, (ts, val) in enumerate(sorted_items):
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            
            # 윈도우 내 값들
            window_start = ts - timedelta(days=window)
            window_vals = sorted([
                v for t, v in sorted_items[:i+1]
                if t >= window_start
            ])
            
            if window_vals:
                # 현재 값의 백분위
                rank = sum(1 for v in window_vals if v <= val)
                percentile = rank / len(window_vals)
                
                values.append(FeatureValue(
                    feature_id=spec.feature_id,
                    timestamp=ts,
                    value=percentile,
                ))
        
        return values
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> Optional[float]:
        """피어슨 상관계수 계산"""
        n = len(x)
        if n < 2:
            return None
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = math.sqrt(sum((v - mean_x) ** 2 for v in x) / n)
        std_y = math.sqrt(sum((v - mean_y) ** 2 for v in y) / n)
        
        if std_x == 0 or std_y == 0:
            return None
        
        return cov / (std_x * std_y)
