"""
도메인 단위/동의어/예외 패턴 사전 (최소 버전)

숫자-단위-대상 구조화 및 질의 감지를 위한 공용 유틸
"""

from typing import Set, Dict
import re

# 표준 단위 셋 (도메인 우선)
DOMAIN_UNITS: Set[str] = {
    "분", "시간", "초",
    "mg/L", "㎎/L", "mg·l-1", "mg\n/L",
    "m³/h", "㎥/h", "m3/h",
    "%",
}

# 단위 동의어 매핑(정규화 대상 → 표준 단위)
UNIT_SYNONYMS: Dict[str, str] = {
    "min": "분",
    "분간": "분",
    "minute": "분",
    "minutes": "분",
    "hr": "시간",
    "hour": "시간",
    "hours": "시간",
    "퍼센트": "%",
    "percent": "%",
    "%": "%",
    "mg/l": "mg/L",
    "㎎/l": "mg/L",
    "mg·l-1": "mg/L",
    "m3/h": "m³/h",
    "㎥/h": "m³/h",
}

# 수량 후보에서 제외할 패턴(날짜/시간/버전/ID 등)
EXCLUDE_PATTERNS = [
    re.compile(r"\b20\d{2}[-/.]?(0[1-9]|1[0-2])[-/.]?(0[1-9]|[12]\d|3[01])\b"),  # YYYY-MM-DD
    re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b"),  # HH:MM(:SS)
    re.compile(r"\bv?\d+(?:\.\d+){1,3}\b", re.IGNORECASE),  # v1.2.3 버전
    re.compile(r"\bID[-_]?[A-Za-z0-9]+\b", re.IGNORECASE),
]

def normalize_unit(unit: str) -> str:
    u = unit.strip()
    u_lower = u.lower()
    if u in DOMAIN_UNITS:
        return u
    if u_lower in UNIT_SYNONYMS:
        return UNIT_SYNONYMS[u_lower]
    # 특수 줄바꿈/스페이스 정리
    u_norm = u.replace("\n", "").replace(" ", "")
    if u_norm in UNIT_SYNONYMS:
        return UNIT_SYNONYMS[u_norm]
    if u_norm in DOMAIN_UNITS:
        return u_norm
    return u

def is_excluded_numeric_context(text: str) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if pat.search(text):
            return True
    return False


