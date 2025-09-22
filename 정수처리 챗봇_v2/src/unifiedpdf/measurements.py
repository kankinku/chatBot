from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional


UNIT_SYNONYMS = {
    "mg/l": {"ppm"},
    "ppm": {"mg/l"},
    "ug/l": {"ppb"},
    "ppb": {"ug/l"},
    "us/cm": {"µs/cm", "μs/cm"},
    "°c": {"℃"},
    "℃": {"°c"},
    # 정수장 특화 단위 추가
    "ntu": {"탁도", "turbidity"},
    "ph": {"산성도", "알칼리도"},
    "do": {"용존산소", "dissolved oxygen"},
    "bod": {"생물학적산소요구량", "biological oxygen demand"},
    "cod": {"화학적산소요구량", "chemical oxygen demand"},
    "toc": {"총유기탄소", "total organic carbon"},
    "cfu": {"대장균군", "coliform"},
    "m³/d": {"m3/d", "m3/day"},
    "m³/h": {"m3/h", "m3/hour"},
    "l/s": {"liter/s", "liter/sec"},
}

# Unit conversion factors (from -> to)
CONVERSIONS: Dict[Tuple[str, str], float] = {
    ("l/s", "m3/d"): 86.4,
    ("m3/d", "l/s"): 1.0 / 86.4,
    ("mg/l", "ppm"): 1.0,  # water approx
    ("ppm", "mg/l"): 1.0,
    ("ug/l", "ppb"): 1.0,
    ("ppb", "ug/l"): 1.0,
    # 정수장 특화 단위 변환 추가
    ("m³/d", "l/s"): 1.0 / 86.4,
    ("l/s", "m³/d"): 86.4,
    ("m³/h", "l/s"): 1.0 / 3.6,
    ("l/s", "m³/h"): 3.6,
    ("m³/h", "m³/d"): 24.0,
    ("m³/d", "m³/h"): 1.0 / 24.0,
    ("kgf/cm²", "bar"): 0.980665,
    ("bar", "kgf/cm²"): 1.01972,
    ("mpa", "kgf/cm²"): 10.1972,
    ("kgf/cm²", "mpa"): 0.0980665,
}


def normalize_number(text: str) -> str:
    return text.replace(",", "")


def normalize_unit(u: str) -> str:
    ul = u.strip().lower().replace("μ", "µ").replace(" ", "")
    return ul


def normalize_date(date_str: str) -> Optional[str]:
    """날짜 문자열을 표준 형식으로 정규화"""
    if not date_str:
        return None
    
    # 이미 정규화된 형식인지 확인
    if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
        return date_str
    
    # 한국어 날짜 형식 (예: 2025년 2월 17일)
    match = re.match(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # 다른 형식들도 추가 가능
    return date_str


def units_equivalent(a: str, b: str) -> bool:
    a, b = normalize_unit(a), normalize_unit(b)
    if a == b:
        return True
    return b in UNIT_SYNONYMS.get(a, set()) or a in UNIT_SYNONYMS.get(b, set())


def convert_value(x: float, unit_from: str, unit_to: str) -> float | None:
    uf, ut = normalize_unit(unit_from), normalize_unit(unit_to)
    if (uf, ut) in CONVERSIONS:
        return x * CONVERSIONS[(uf, ut)]
    if units_equivalent(uf, ut):
        return x
    return None


def extract_measurements(text: str) -> List[Tuple[str, str]]:
    # Return list of (number, unit)
    pairs: List[Tuple[str, str]] = []
    
    # 기본 수치-단위 패턴
    for m in re.finditer(r"(\d+(?:[\.,]\d+)?)\s*([A-Za-z%°℃µμ/]+)", text):
        num = normalize_number(m.group(1))
        unit = normalize_unit(m.group(2))
        pairs.append((num, unit))
    
    # 상관계수 패턴 (예: 0.72, R²=0.95)
    for m in re.finditer(r"(?:상관계수|correlation|R²|R2|R-squared)\s*[=:]\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE):
        num = normalize_number(m.group(1))
        pairs.append((num, "correlation"))
    
    # 날짜 패턴 (예: 2025년 2월 17일)
    for m in re.finditer(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", text):
        year, month, day = m.group(1), m.group(2), m.group(3)
        formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        pairs.append((formatted_date, "date"))
    
    # 백분율 패턴 (예: 95%, 72.2%)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        num = normalize_number(m.group(1))
        pairs.append((num, "percent"))
    
    return pairs


def build_context_measure_map(contexts: List[str]) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for t in contexts:
        for num, unit in extract_measurements(t):
            m.setdefault(unit, []).append(num)
    return m


def verify_answer_numeric(answer: str, context_map: Dict[str, List[str]], tol_ratio: float = 0.05) -> float:
    ans = extract_measurements(answer)
    if not ans:
        return 1.0
    ok = 0
    for num, unit in ans:
        candidates: List[float] = []
        for u, nums in context_map.items():
            for n in nums:
                try:
                    y = float(n)
                except Exception:
                    continue
                cy = convert_value(y, u, unit)
                if cy is not None:
                    candidates.append(cy)
        if not candidates:
            continue
        try:
            x = float(num)
        except Exception:
            continue
        matched = any((abs(x - y) / abs(y) <= tol_ratio) for y in candidates if y != 0)
        if matched:
            ok += 1
    return ok / len(ans)

