"""
SLA 및 LLM 파라미터 SSOT

라우트/모드별 p95 타깃 및 LLM 파라미터(temperature/max_new_tokens 등)를
중앙에서 관리하기 위한 경량 정책 레이어.
"""

from typing import Dict, Any

try:
    from core.config.unified_config import get_config
except Exception:  # pragma: no cover
    def get_config(key: str, default=None):
        import os
        return os.getenv(key, default)


# 라우트 키 표준: pdf_search | greeting | general 등
DEFAULT_SLA_P95_MS = int(get_config("SLA_P95_TARGET_MS", 3000))


def get_route_sla_p95_ms(route: str) -> int:
    """라우트별 p95 타깃 조회(환경변수/설정 오버라이드 지원).

    우선순위 예시(환경변수): SLA_P95_TARGET_MS__PDF_SEARCH=2500
                        SLA_P95_TARGET_MS__GREETING=1500
    없으면 글로벌 기본(DEFAULT_SLA_P95_MS) 사용.
    """
    key = f"SLA_P95_TARGET_MS__{route.upper()}"
    return int(get_config(key, DEFAULT_SLA_P95_MS))


def get_llm_policy(route: str, mode: str = "default") -> Dict[str, Any]:
    """라우트/모드별 LLM 파라미터 정책을 반환.

    환경변수 키 예시:
      LLM_TEMP__PDF_SEARCH=0.6
      LLM_MAX_NEW_TOKENS__PDF_SEARCH=512
      LLM_TEMP__GREETING=0.8
      LLM_MAX_NEW_TOKENS__GREETING=256

    모드별(고비용/저비용) 오버라이드도 지원:
      LLM_TEMP__PDF_SEARCH__HIGH=0.7
      LLM_MAX_NEW_TOKENS__PDF_SEARCH__LOW=256
    """
    # 기본값
    # 보수적 기본값으로 낮춤
    default_temp = float(get_config("LLM_DEFAULT_TEMPERATURE", 0.3))
    default_max_new = int(get_config("LLM_DEFAULT_MAX_NEW_TOKENS", 256))

    route_upper = route.upper()
    mode_upper = (mode or "default").upper()

    # 모드 오버라이드가 우선, 없으면 라우트 오버라이드, 없으면 기본
    temp = get_config(f"LLM_TEMP__{route_upper}__{mode_upper}", None)
    if temp is None:
        temp = get_config(f"LLM_TEMP__{route_upper}", default_temp)
    temp = float(temp)

    max_new = get_config(f"LLM_MAX_NEW_TOKENS__{route_upper}__{mode_upper}", None)
    if max_new is None:
        max_new = get_config(f"LLM_MAX_NEW_TOKENS__{route_upper}", default_max_new)
    max_new = int(max_new)

    top_p = float(get_config("LLM_DEFAULT_TOP_P", 0.8))
    num_beams = int(get_config("LLM_DEFAULT_NUM_BEAMS", 2))
    do_sample = str(get_config("LLM_DEFAULT_DO_SAMPLE", False)).lower() in ("true", "1", "yes")
    early_stopping = str(get_config("LLM_DEFAULT_EARLY_STOP", True)).lower() in ("true", "1", "yes")

    return {
        "temperature": temp,
        "max_new_tokens": max_new,
        "top_p": top_p,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "early_stopping": early_stopping,
    }


def get_rerank_policy(route: str) -> bool:
    """라우트별 재순위 사용 여부(SSOT). 예: RERANK_ENABLED__PDF_SEARCH=true"""
    route_upper = route.upper()
    enabled = str(get_config(f"RERANK_ENABLED__{route_upper}", get_config("ENABLE_RERANKING", True))).lower()
    return enabled in ("true", "1", "yes", "on")


def get_min_accuracy_threshold(route: str) -> float:
    """라우트별 최소 정확도 임계.
    예: MIN_ACCURACY__PDF_SEARCH=0.85 (기본 0.8)
    """
    route_upper = route.upper()
    return float(get_config(f"MIN_ACCURACY__{route_upper}", get_config("GATE_QUALITY_MIN", 0.80)))


