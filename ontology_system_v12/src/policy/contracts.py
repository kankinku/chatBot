"""
Policy contract objects.

이 모듈은 추론 파이프라인이 외부 학습/탐색에서 산출한 정책을
표준 형태로 주입받기 위한 DTO를 정의한다.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
import uuid


def _gen_id() -> str:
    return f"POL_{uuid.uuid4().hex[:12]}"


@dataclass
class PolicyBundle:
    """
    정책 설정 번들.
    reasoning 단계에서 하드코딩된 가중치 대신 이 객체를 주입한다.
    """

    policy_id: str = field(default_factory=_gen_id)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    params: Dict[str, Any] = field(default_factory=dict)

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        중첩 키를 dot-notation으로 조회 (예: fusion.domain_weight).
        """
        if key in self.params:
            return self.params.get(key, default)
        # dot notation
        if "." in key:
            head, *tail = key.split(".")
            current = self.params.get(head, {})
            for part in tail:
                if not isinstance(current, dict):
                    return default
                current = current.get(part, default)
            return current
        return default
