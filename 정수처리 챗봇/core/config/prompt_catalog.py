"""
프롬프트/도메인 SSOT: 템플릿 카탈로그 + 버전/플래그(A/B) 운영
"""

from typing import Dict


_CATALOG: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {
    # domain -> variant(A/B/...) -> version -> templates
    "general": {
        "A": {
            "1": {
                "context": """다음은 참고 문서입니다:\n\n{context}\n\n사용자 질문: {question}\n\n간결하고 명확한 한국어로 답변하세요.""",
                "no_context": """사용자 질문: {question}\n\n간결하고 명확한 한국어로 답변하세요.""",
                "summary": """다음 문서들을 요약하세요(핵심 위주, 3문장):\n\n{context}\n\n자연스러운 한국어로 요약해주세요:""",
            }
        }
    },
    "legal": {
        "A": {
            "1": {
                "context": """다음은 법률/규정 관련 문서입니다:\n\n{context}\n\n사용자 질문: {question}\n\n다음 지침을 따르세요:\n- 법령/조항/근거를 명시\n- 인용은 정확히, 추정/단정 지양\n- 판례/행정해석은 출처 표시\n자연스러운 한국어로 명확하게 답변하세요.""",
                "no_context": """사용자 질문: {question}\n\n법률 도메인 어시스턴트입니다. 관련 법령과 근거 중심으로 답변하되, 모호한 경우 불확실성을 명시하세요.""",
                "summary": """다음 법률 문서들을 근거와 함께 요약하세요(핵심 3문장):\n\n{context}\n\n자연스러운 한국어로 요약해주세요:""",
            }
        }
    },
    "water": {
        "A": {
            "1": {
                "context": """다음은 정수처리/정수장 운영 관련 문서입니다:\n\n{context}\n\n사용자 질문: {question}\n\n다음 지침을 따르세요:\n- 공정/시설/운영 기준을 근거로 답변\n- 안전/법규 준수 사항 우선\n- 측정값/단위 명확화, 과도한 추정 금지\n자연스러운 한국어로 명확하게 답변하세요.""",
                "no_context": """사용자 질문: {question}\n\n정수처리 도메인 어시스턴트입니다. 공정/운영 기준에 근거해 간결히 답변하세요.""",
                "summary": """다음 정수처리 문서들을 핵심 기준과 함께 요약하세요(3문장):\n\n{context}\n\n자연스러운 한국어로 요약해주세요:""",
            }
        }
    },
}


def get_templates(domain: str, variant: str = "A", version: str = "1") -> Dict[str, str]:
    domain_key = (domain or "general").lower()
    if domain_key not in _CATALOG:
        domain_key = "general"
    var_map = _CATALOG[domain_key]
    if variant not in var_map:
        variant = "A"
    ver_map = var_map[variant]
    if version not in ver_map:
        version = next(iter(ver_map.keys()))
    return ver_map[version]


