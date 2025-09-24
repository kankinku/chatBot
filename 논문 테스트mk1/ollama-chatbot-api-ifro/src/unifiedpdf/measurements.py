from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class MeasureSpan:
    start: int
    end: int
    raw: str
    value: Optional[float]
    range_: Optional[Tuple[float, float]]
    unit: str
    scale: Optional[str]
    label_hint: Optional[str]
    modifier: Optional[str] = None  # 평균/최대/최소/기준 등


def normalize_number(text: str) -> str:
    return text.replace(",", "").strip()


def normalize_unit(u: str) -> str:
    if not u:
        return ""
    ul = u.strip().replace("μ", "µ").replace(" ", "").lower()
    # ASCII variants
    ul = ul.replace("us/cm", "µs/cm").replace("degc", "°c")
    # Slashes
    ul = ul.replace("mg|l", "mg/l")
    return ul


# Unit ontology (minimal)
UNIT_SYNONYMS: Dict[str, set[str]] = {
    "mg/l": {"ppm"},
    "ppm": {"mg/l"},
    "ug/l": {"ppb"},
    "ppb": {"ug/l"},
    "µs/cm": {"us/cm", "μs/cm"},
    "m3/d": {"m³/d"},
    "m³/d": {"m3/d"},
    "m3/h": {"m³/h"},
    "m³/h": {"m3/h"},
}


# Simple conversions (from -> to)
CONVERSIONS: Dict[Tuple[str, str], float] = {
    ("l/s", "m3/d"): 86.4,
    ("m3/d", "l/s"): 1.0 / 86.4,
    ("mg/l", "ppm"): 1.0,
    ("ppm", "mg/l"): 1.0,
    ("ug/l", "ppb"): 1.0,
    ("ppb", "ug/l"): 1.0,
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


def units_equivalent(a: str, b: str) -> bool:
    a, b = normalize_unit(a), normalize_unit(b)
    if a == b:
        return True
    return b in UNIT_SYNONYMS.get(a, set()) or a in UNIT_SYNONYMS.get(b, set())


def convert_value(x: float, unit_from: str, unit_to: str) -> Optional[float]:
    uf, ut = normalize_unit(unit_from), normalize_unit(unit_to)
    if (uf, ut) in CONVERSIONS:
        return x * CONVERSIONS[(uf, ut)]
    if units_equivalent(uf, ut):
        return x
    # scale conversions (% / ‰ / bp)
    scale_map = {"%": 0.01, "‰": 0.001, "bp": 0.0001}
    if uf in scale_map and ut in scale_map:
        return x * (scale_map[uf] / scale_map[ut])
    return None


NUM_RE = r"\d{1,3}(?:[\s,]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?"
UNIT_RE = r"[A-Za-z°µ/%‰²³\\/]+"
RANGE_SEP = r"[–~\-]"


def _parse_number(s: str) -> Optional[float]:
    try:
        return float(s.replace(" ", "").replace(",", "").replace("\u00a0", ""))
    except Exception:
        return None


def _label_hint(text: str, start: int) -> Optional[str]:
    # 후보 1: 콜론 패턴
    left = text[max(0, start - 80):start]
    m = re.search(r"([\w가-힣A-Za-z/%µ°\s]{1,40})\s*[:：]", left)
    if m:
        return m.group(1).strip()[-40:]
    # 후보 2: 괄호 내 설명
    right = text[start:start + 80]
    m2 = re.search(r"\(([^)]+)\)", right)
    if m2:
        return m2.group(1)[:40]
    # 후보 3: 숫자 직전 명사구(한글/영문 단어 1~2개)
    m3 = re.search(r"([가-힣A-Za-z]{1,20})(?:\s+([가-힣A-Za-z]{1,20}))?\s*$", left)
    if m3:
        parts = [p for p in m3.groups() if p]
        return " ".join(parts)[-40:] if parts else None
    # 후보 4: 직전 라인의 헤더성 토큰 추정
    lb = text.rfind("\n", 0, start)
    if lb != -1:
        lbb = text.rfind("\n", 0, max(0, lb))
        prev_line = text[(lbb + 1 if lbb != -1 else 0):lb]
        prev_line = prev_line.strip()
        if prev_line:
            # 구분자 기준 분할 후 숫자 제거, 마지막 토큰 사용
            cand = prev_line.split("|")[-1].split(":")[-1]
            cand = re.sub(r"\d+(?:[\.,]\d+)?", "", cand).strip()
            if len(cand) >= 2:
                return cand[:40]
    return None


def _modifier_hint(text: str, start: int, end: int) -> Optional[str]:
    window = text[max(0, start - 40):min(len(text), end + 40)]
    keywords = [
        "평균", "최대", "최소", "중앙값", "상한", "하한", "허용", "기준",
        "mean", "avg", "maximum", "minimum", "median", "limit", "threshold",
        "min", "max"
    ]
    for kw in keywords:
        if kw in window:
            return kw
    return None


def extract_measure_spans(text: str) -> List[MeasureSpan]:
    spans: List[MeasureSpan] = []
    # Range like 6.5–8.5 pH
    pat_range = re.compile(fr"(?P<low>({NUM_RE}))\s*{RANGE_SEP}\s*(?P<high>({NUM_RE}))\s*(?P<unit>{UNIT_RE})")
    for m in pat_range.finditer(text):
        low = _parse_number(normalize_number(m.group("low")))
        high = _parse_number(normalize_number(m.group("high")))
        unit = normalize_unit(m.group("unit"))
        scale = unit if unit in {"%", "‰", "bp"} else None
        spans.append(MeasureSpan(
            start=m.start(), end=m.end(), raw=m.group(0), value=None,
            range_=(low, high) if (low is not None and high is not None) else None,
            unit=unit, scale=scale, label_hint=_label_hint(text, m.start()), modifier=_modifier_hint(text, m.start(), m.end())
        ))

    # Single value like 72.2 %, 0.2 NTU
    pat_single = re.compile(fr"(?P<num>({NUM_RE}))\s*(?P<unit>{UNIT_RE})")
    for m in pat_single.finditer(text):
        num = _parse_number(normalize_number(m.group("num")))
        unit_raw = m.group("unit")
        unit = normalize_unit(unit_raw)
        scale = unit if unit in {"%", "‰", "bp"} else None
        spans.append(MeasureSpan(
            start=m.start(), end=m.end(), raw=m.group(0), value=num, range_=None,
            unit=unit, scale=scale, label_hint=_label_hint(text, m.start()), modifier=_modifier_hint(text, m.start(), m.end())
        ))

    return spans


def extract_measurements(text: str) -> List[Tuple[str, str]]:
    # Backward-compat: return (number, unit) pairs from spans
    out: List[Tuple[str, str]] = []
    for sp in extract_measure_spans(text):
        if sp.value is not None and sp.unit:
            out.append((str(sp.value), sp.unit))
        elif sp.range_ is not None and sp.unit:
            out.append((str(sp.range_[0]), sp.unit))
            out.append((str(sp.range_[1]), sp.unit))
    return out


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
        matched = any((abs(x - y) / max(1e-9, abs(y)) <= tol_ratio) for y in candidates if y != 0)
        if matched:
            ok += 1
    return ok / len(ans)
