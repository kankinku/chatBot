"""
보안 유틸: 요청 파라미터 스키마 검증, 프롬프트 정화/경계 강제
"""

from typing import Dict, Any, List
import re

class ValidationError(Exception):
    pass


def validate_question_payload(payload: Dict[str, Any], *, max_question_len: int, allowed_fields: List[str]) -> Dict[str, Any]:
    """/ask 요청 바디 스키마/필드 검증.
    - 허용되지 않은 필드 금지
    - question 길이 검사
    - 기본 타입 검사(간단)
    """
    if not isinstance(payload, dict):
        raise ValidationError("payload_must_be_object")

    # 허용되지 않은 필드 차단
    extra = [k for k in payload.keys() if k not in set(allowed_fields)]
    if extra:
        raise ValidationError(f"unexpected_fields:{','.join(extra[:5])}")

    q = (payload.get("question") or "").strip()
    if not q:
        raise ValidationError("question_required")
    if len(q) > max_question_len:
        raise ValidationError("question_too_long")

    # 간단 타입 검사
    if "max_chunks" in payload and not isinstance(payload["max_chunks"], int):
        raise ValidationError("max_chunks_must_be_int")
    if "use_conversation_context" in payload and not isinstance(payload["use_conversation_context"], bool):
        raise ValidationError("use_conversation_context_must_be_bool")

    return payload


_PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+previous\s+instructions",
    r"(?i)disregard\s+above\s+rules",
    r"(?i)system\s*:\s*",
    r"(?i)you\s+are\s+chatgpt",
]


def sanitize_prompt(text: str, *, remove_sensitive_meta: bool = True, additional_patterns: List[str] = None) -> str:
    """프롬프트 인젝션 완화: 위험 패턴 마스킹, 민감 메타 제거.
    간단한 정규식 기반. 필요시 확장 가능.
    """
    if not text:
        return text

    sanitized = text
    # 위험 패턴 마스킹
    patterns = list(_PROMPT_INJECTION_PATTERNS)
    if additional_patterns:
        patterns.extend(additional_patterns)
    for pat in patterns:
        sanitized = re.sub(pat, "[masked]", sanitized)

    # 민감 메타(헤더/토큰/키) 제거(간단)
    if remove_sensitive_meta:
        sanitized = re.sub(r"(?i)(api[-_ ]?key|token|secret)\s*[:=]\s*\S+", "[secret]", sanitized)

    # 과도한 제어문자 제거
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", sanitized)
    return sanitized


