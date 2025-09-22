import time
from typing import Callable


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_time_sec: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_time_sec = recovery_time_sec
        self._failures = 0
        self._opened_at = 0.0

    def call(self, func: Callable, *args, **kwargs):
        now = time.time()
        if self._opened_at and now - self._opened_at < self.recovery_time_sec:
            raise RuntimeError("circuit_open")
        try:
            result = func(*args, **kwargs)
            self._failures = 0
            self._opened_at = 0.0
            return result
        except Exception:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = now
            raise


