"""
Decorators for guarding repository access.
"""
import functools
import inspect
from datetime import datetime
from src.common.asof_context import validate_access, get_context

def guard_replay_access(func):
    """
    Decorator that enforces as_of guard logic before execution.
    It inspects arguments for 'as_of' or 'timestamp' and validates them.
    Also ensures global REPLAY mode constraints.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = get_context()
        if ctx.mode == "REPLAY_SNAPSHOT_ONLY":
            # 1. Base check
            if ctx.as_of is None:
                 raise PermissionError("Access denied: REPLAY mode requires active as_of context.")
            
            # 2. Argument check
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            
            # Check known time arguments
            check_time = None
            if 'as_of' in bound.arguments and bound.arguments['as_of']:
                 check_time = bound.arguments['as_of']
            elif 'timestamp' in bound.arguments and bound.arguments['timestamp']:
                 check_time = bound.arguments['timestamp']
            
            if check_time:
                 validate_access(check_time)
            
            # If no time arg is present but we are in REPLAY mode, 
            # generally we might want to check if the method is inherently safe.
            # For now, we rely on 'validate_access' inside if needed, or this decorator 
            # effectively acts as a "Gatekeeper" confirming we are in a valid context state.
            
        return func(*args, **kwargs)
    return wrapper
