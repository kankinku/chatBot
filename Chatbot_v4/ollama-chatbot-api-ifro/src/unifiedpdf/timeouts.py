from __future__ import annotations

import concurrent.futures
import os
import threading
from typing import Any, Callable, Optional


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


LLM_TIMEOUT_S = _env_float("LLM_TIMEOUT", 120.0)
RERANK_TIMEOUT_S = _env_float("RERANK_TIMEOUT", 10.0)
SEARCH_TIMEOUT_S = _env_float("SEARCH_TIMEOUT", 5.0)


def run_with_timeout(func: Callable[[], Any], timeout_s: float, default: Optional[Any] = None) -> Any:
    """Windows-safe timeout using ThreadPoolExecutor.
    SIGALRM 금지, 시간 초과 시 기본값 반환.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return default
        except Exception:
            return default

