from __future__ import annotations

import json
import math
import os
from typing import Iterable, List, Tuple

import urllib.request


def _ollama_generate(prompt: str, model_name: str, timeout: int = 20) -> str:
    url = "http://127.0.0.1:11434/api/generate"
    data = {
        "model": model_name, 
        "prompt": prompt, 
        "stream": False,
        "keep_alive": "24h"  # 모델을 메모리에 유지
    }
    req = urllib.request.Request(
        url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return body.get("response", "")
    except Exception:
        return ""


def _format_correction_prompt(text: str, dictionary: str | None = None) -> str:
    # 정수장 도메인 사전을 기본으로 사용
    if dictionary is None:
        from .domain_dictionary import get_domain_dictionary
        dictionary = get_domain_dictionary()
    
    parts = [
        "당신은 정수장 전문 OCR 후교정기입니다. 절대 재작성/요약/삭제를 하지 마세요.",
        "규칙:",
        "1) 줄바꿈/공백/번호/구두점을 유지",
        "2) 숫자/단위/기호 보존(예: mg/L, ppm, ℃, L/s, m³/d, NTU, pH, DO)",
        "3) 확실한 OCR 오류만 교정(rn→m, 0↔O, l↔1, cl→d, mg|L→mg/L 등)",
        "4) 정수장 전문 용어는 사전 기준으로 교정, 불확실하면 원문 그대로",
        "5) 수치 범위 확인: pH(6.5-8.5), 탁도(0-0.5NTU), 잔류염소(0.1-0.4mg/L)",
        "출력: 교정된 본문만 출력",
    ]
    if dictionary:
        parts.append("[정수장 전문 사전]")
        parts.append(dictionary)
    parts.append("[원문]")
    parts.append(text)
    return "\n".join(parts)


def noise_score(t: str) -> float:
    if not t:
        return 0.0
    n = len(t)
    bad = t.count("�")
    # Suspicious bigrams common in OCR mistakes
    sus = ["rn", "cl", "0O", "O0", "l1", "1l"]
    s = sum(t.count(x) for x in sus)
    punct = sum(1 for c in t if not c.isalnum() and not c.isspace())
    score = 0.4 * (bad / n) + 0.4 * (s / max(1, n / 20)) + 0.2 * (punct / n)
    return max(0.0, min(1.0, score))


def select_low_quality_indices(texts: List[str], threshold: float, max_chars: int) -> List[int]:
    scored = [(i, noise_score(t), len(t)) for i, t in enumerate(texts)]
    scored.sort(key=lambda x: x[1], reverse=True)
    out: List[int] = []
    budget = 0
    for i, sc, ln in scored:
        if sc < threshold:
            break
        if budget + ln > max_chars:
            break
        out.append(i)
        budget += ln
    return out


def correct_texts_with_llm(texts: List[str], model_name: str, batch: int = 4, timeout: int = 20, dictionary: str | None = None) -> List[str]:
    out: List[str] = [t for t in texts]
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        for j, t in enumerate(chunk):
            prompt = _format_correction_prompt(t, dictionary=dictionary)
            resp = _ollama_generate(prompt, model_name=model_name, timeout=timeout)
            out[i + j] = resp.strip() or t
    return out


def apply_llm_post_correction(texts: List[str], model_name: str, threshold: float = 0.5, max_chars: int = 10000, batch: int = 4, dictionary: str | None = None) -> List[str]:
    if max_chars <= 0 or not texts:
        return texts
    idx = select_low_quality_indices(texts, threshold=threshold, max_chars=max_chars)
    if not idx:
        return texts
    to_fix = [texts[i] for i in idx]
    fixed = correct_texts_with_llm(to_fix, model_name=model_name, batch=batch, dictionary=dictionary)
    out = [t for t in texts]
    for k, i in enumerate(idx):
        out[i] = fixed[k]
    return out


# Basic rule-based corrections (fast, deterministic)
def apply_basic_corrections(texts: List[str]) -> List[str]:
    repl = [
        ("rn", "m"),
        ("mg|L", "mg/L"),
        ("us/cm", "µS/cm"),
        ("μs/cm", "µS/cm"),
        ("m3/d", "m³/d"),
        ("m3/h", "m³/h"),
        (" deg c", " °C"),
        ("degc", "°C"),
        ("ppm ", "ppm "),
        (" ppb", " ppb"),
    ]
    out: List[str] = []
    for t in texts:
        s = t
        for a, b in repl:
            s = s.replace(a, b)
        # unify micro symbol
        s = s.replace("μ", "µ")
        out.append(s)
    return out
