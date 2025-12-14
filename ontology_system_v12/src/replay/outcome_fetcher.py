"""
Outcome fetcher: t 이후 구간의 실제 결과를 조회하는 단순 구현.
현재는 방향성(directon)만 계산할 수 있는 간단한 형태로 둔다.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any


@dataclass
class Outcome:
    direction_actual: int  # -1, 0, +1
    magnitude_actual: float = 0.0
    drawdown: float = 0.0
    vol: float = 0.0
    raw_refs: Dict[str, Any] = None


class SimpleOutcomeFetcher:
    """
    사용자 제공 콜백을 감싸는 래퍼.
    """

    def __init__(self, fetch_fn: Callable[[datetime, datetime], Dict[str, float]]):
        self.fetch_fn = fetch_fn

    def fetch(self, as_of: datetime, horizon_days: int = 30) -> Optional[Outcome]:
        target_date = as_of + timedelta(days=horizon_days)
        data = self.fetch_fn(as_of, target_date)
        if not data:
            return None
        # data 예: {"start": 100, "end": 105, "min": 95, "vol": 0.2}
        start = data.get("start")
        end = data.get("end")
        if start is None or end is None:
            return None
        direction = 0
        if end > start:
            direction = 1
        elif end < start:
            direction = -1
        drawdown = 0.0
        if "min" in data:
            drawdown = (data["min"] - start) / max(abs(start), 1e-6)
        return Outcome(
            direction_actual=direction,
            magnitude_actual=(end - start) / max(abs(start), 1e-6),
            drawdown=drawdown,
            vol=data.get("vol", 0.0),
            raw_refs=data,
        )
