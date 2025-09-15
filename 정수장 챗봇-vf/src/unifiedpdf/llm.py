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
    # 정수장 도메인 특화 가이드
    guide = {
        "numeric": "정확한 숫자와 단위로 답변하세요.",
        "definition": "정의나 개념을 명확히 답변하세요.",
        "procedural": "구체적인 절차나 방법을 단계별로 답변하세요.",
        "comparative": "차이점을 명확히 비교하여 답변하세요.",
        "system_info": "시스템 정보를 정확히 답변하세요 (URL, 계정, 설정값 등).",
        "technical_spec": "기술적 사양을 정확히 답변하세요 (모델명, 성능지표, 설정값 등).",
        "operational": "운영 관련 정보를 구체적으로 답변하세요.",
        "problem": "문제 원인과 해결방법을 구체적으로 답변하세요.",
        "general": "핵심 내용을 정확하고 간결하게 답변하세요.",
    }.get(qtype, "핵심 내용을 정확하고 간결하게 답변하세요.")

    parts = [
        "정수장 시스템 사용자 설명서 QA입니다. 다음 규칙을 반드시 지키세요:",
        "1) 오직 한국어로만 답변 (영어, 중국어, 일본어, 이모지, 특수문자 절대 금지)",
        "2) 문서 내용만 답변 (추측이나 외부 지식 사용 금지)",
        "3) 정확하고 구체적으로 답변",
        "4) 모르면 '문서에서 해당 정보를 확인할 수 없습니다.'",
        "5) 불필요한 설명이나 추가 문장 금지",
        "6) 답변은 자연스러운 한국어 문장으로만 작성",
        "7) 영어 단어나 외국어 사용 절대 금지",
        f"8) {guide}",
        "",
        "정수장 전문 용어:",
        "- AI 플랫폼, 자율운영, 대시보드, SCADA",
        "- 착수, 약품, 혼화응집, 침전, 여과, 소독 공정",
        "- LSTM, GRU, N-beats, XGB, GBR 모델",
        "- R², MAE, MSE, RMSE 성능지표",
        "- KWATER, 관리자계정, 로그인정보",
        "- PMS, EMS, iRDC 시스템",
        "",
        "[문서]",
    ]
    for i, s in enumerate(contexts, start=1):
        parts.append(f"[{i}] {s.chunk.text}")
    parts.append("")
    parts.append(f"[질문] {question}")
    parts.append("위 질문에 대해 문서 내용을 바탕으로 한국어로만 답변하세요. [답변] 같은 형식 없이 바로 답변 내용만 작성하세요. 자연스러운 문장으로 작성하고 불필요한 줄바꿈을 사용하지 마세요.")
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


def _post_process_answer(text: str) -> str:
    """답변 후처리: 언어 검증, 형식 정리, 개행문자 처리"""
    if not text:
        return ""
    
    # 1. [답변] 형식 제거
    text = text.strip()
    if text.startswith("[답변]"):
        text = text[4:].strip()
    
    # 2. 개행문자 정리 (\n을 실제 줄바꿈으로 변환 후 다시 공백으로)
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    
    # 3. 영어/중국어 감지 및 필터링
    import re
    
    # 영어 문장 패턴 감지 (대문자로 시작하는 영어 문장)
    english_pattern = r'[A-Z][a-z]+.*?[.!?]'
    english_matches = re.findall(english_pattern, text)
    
    # 중국어/일본어 문자 감지
    cjk_pattern = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]'
    cjk_matches = re.findall(cjk_pattern, text)
    
    # 영어나 중국어가 많이 포함된 경우 한국어로 재요청
    if len(english_matches) > 2 or len(cjk_matches) > len(text) / 2:
        return "문서에서 해당 정보를 확인할 수 없습니다."
    
    # 4. 불필요한 특수문자 및 이모지 제거
    text = re.sub(r'[^\w\s가-힣.,!?()\-/:]', '', text)
    
    # 5. 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


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
    
    # 답변 후처리 적용
    return _post_process_answer(text)
