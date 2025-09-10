from __future__ import annotations

import json
import urllib.request
import logging
import os
import time
from typing import Dict, List, Optional

from .config import PipelineConfig
from .types import RetrievedSpan
from .timeouts import LLM_TIMEOUT_S, run_with_timeout


_LOG_PATH = os.path.join("logs", "llm_errors.log")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
logging.basicConfig(level=logging.ERROR)
_logger = logging.getLogger("unifiedpdf.llm")
if not _logger.handlers:
    _fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    _fh.setLevel(logging.ERROR)
    _logger.addHandler(_fh)


def _format_prompt(question: str, contexts: List[RetrievedSpan], qtype: str = "general") -> str:
    # 유형별 한줄 가이드
    guide = {
        "numeric": "숫자와 단위만 정확히 답변하세요.",
        "definition": "간단한 정의만 답변하세요.",
        "procedural": "핵심 절차만 간단히 답변하세요.",
        "comparative": "차이점만 간단히 답변하세요.",
        "general": "핵심만 1-2문장으로 간단히 답변하세요.",
    }.get(qtype, "핵심만 1-2문장으로 간단히 답변하세요.")

    parts = [
        "당신은 한국어로만 답변하는 문서 QA 시스템입니다.",
        "반드시 지켜야 할 규칙:",
        "1) 오직 한국어로만 답변하세요 (영어, 중국어, 일본어, 이모지, 특수문자 절대 금지)",
        "2) 문서에 있는 내용만 답변하세요",
        "3) 간단하고 정확하게 답변하세요",
        "4) 모르면 '문서에서 해당 정보를 확인할 수 없습니다.'라고 답변하세요",
        "5) 이모지, 특수문자, 외국어 사용 금지",
        "6) 자연스러운 한국어 문장으로만 답변",
        f"7) {guide}",
        "",
        "[문서]",
    ]
    for i, s in enumerate(contexts, start=1):
        parts.append(f"[{i}] {s.chunk.text}")
    parts.append("")
    parts.append(f"[질문] {question}")
    parts.append("[답변]")
    return "\n".join(parts)


def ollama_generate(prompt: str, model_name: str, timeout_s: Optional[int] = None) -> str:
    url = "http://127.0.0.1:11434/api/generate"
    data = {"model": model_name, "prompt": prompt, "stream": False}
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s or LLM_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return body.get("response", "")
    except Exception as e:
        _logger.error("Ollama request failed: %s", e)
        return ""


def generate_answer(question: str, contexts: List[RetrievedSpan], cfg: PipelineConfig, qtype: str = "general") -> str:
    prompt = _format_prompt(question, contexts, qtype=qtype)

    def _call():
        return ollama_generate(prompt, cfg.model_name, timeout_s=LLM_TIMEOUT_S)

    # Retries with backoff
    tries = max(0, int(getattr(cfg, "llm_retries", 0))) + 1
    backoff = int(getattr(cfg, "llm_retry_backoff_ms", 300)) / 1000.0
    text = ""
    for t in range(tries):
        text = run_with_timeout(_call, timeout_s=LLM_TIMEOUT_S + 2, default="")  # small cushion
        if text.strip():
            break
        time.sleep(backoff * (t + 1))
    return text.strip()
