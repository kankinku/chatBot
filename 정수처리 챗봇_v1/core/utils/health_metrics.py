"""
헬스/레디니스 및 운영 지표 수집 유틸 (경량)
"""

from typing import Dict, Any
import time

class HealthMetrics:
    def __init__(self):
        self._since = time.time()
        self._route_false_positives = 0
        self._route_false_negatives = 0
        self._alerts_triggered = 0

    def incr_fp(self):
        self._route_false_positives += 1

    def incr_fn(self):
        self._route_false_negatives += 1

    def incr_alert(self):
        self._alerts_triggered += 1

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "uptime_sec": int(now - self._since),
            "routing": {
                "false_positives": self._route_false_positives,
                "false_negatives": self._route_false_negatives,
            },
            "alerts": {
                "triggered": self._alerts_triggered
            }
        }

health_metrics = HealthMetrics()


