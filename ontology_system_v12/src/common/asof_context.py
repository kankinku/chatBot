"""
Async-safe context management using contextvars.
Replaces thread-local implementation.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, ContextManager
from contextlib import contextmanager
import contextvars

@dataclass
class AsOfContext:
    mode: str = "LIVE"  # LIVE | REPLAY_SNAPSHOT_ONLY
    as_of: Optional[datetime] = None
    snapshot_id: Optional[str] = None
    request_id: Optional[str] = None

# ContextVar for async safety
_context_var: contextvars.ContextVar[AsOfContext] = contextvars.ContextVar(
    "as_of_context", default=AsOfContext()
)

def get_context() -> AsOfContext:
    return _context_var.get()

def set_context(ctx: AsOfContext):
    _context_var.set(ctx)

def validate_access(requested_time: Optional[datetime]) -> None:
    """
    Guard: REPLAY mode denies access to future data or if as_of is missing.
    """
    ctx = get_context()
    if ctx.mode == "REPLAY_SNAPSHOT_ONLY":
        if ctx.as_of is None:
            raise ValueError("AsOfContext missing in REPLAY mode")
        if requested_time and requested_time > ctx.as_of:
            raise ValueError(f"Future access DENIED in REPLAY mode. Req={requested_time}, AsOf={ctx.as_of}")

@contextmanager
def asof_guard(as_of: datetime, mode: str = "LIVE", snapshot_id: Optional[str] = None):
    token = _context_var.set(AsOfContext(mode=mode, as_of=as_of, snapshot_id=snapshot_id))
    try:
        yield
    finally:
        _context_var.reset(token)
