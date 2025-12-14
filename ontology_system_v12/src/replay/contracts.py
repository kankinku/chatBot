"""
Replay 관련 표준 컨텍스트/DTO.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class ReplayContext:
    """
    리플레이 실행에 필요한 최소 정보.
    """

    as_of: datetime
    snapshot_id: str
    policy_id: str
    seed: int = 0
    mode: Literal["REPLAY_SNAPSHOT_ONLY"] = "REPLAY_SNAPSHOT_ONLY"
