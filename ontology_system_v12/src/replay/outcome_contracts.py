from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

@dataclass
class Outcome:
    target_id: str
    as_of: datetime
    horizon_date: datetime
    
    direction_actual: int  # -1, 0, 1
    magnitude_actual: float  # e.g., pct change
    drawdown: float  # max drawdown in period
    
    raw_value_start: float
    raw_value_end: float
    
    status: str = "OK"  # OK, MISSING, ERROR
    error_reason: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
