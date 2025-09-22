from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .config import PipelineConfig


@dataclass
class Analysis:
    qtype: str  # numeric | definition | procedural | comparative | problem | general
    length: int
    key_token_count: int
    rrf_vector_weight: float
    rrf_bm25_weight: float
    threshold_adj: float




def _load_domain(cfg: PipelineConfig) -> Dict[str, List[str]]:
    path = getattr(getattr(cfg, "domain", object()), "domain_dict_path", None)
    if not path:
        return {"units": [], "keywords": [], "procedural": [], "comparative": [], "definition": [], "problem": []}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        norm = {k: [str(x).lower() for x in data.get(k, [])] for k in [
            "units", "keywords", "procedural", "comparative", "definition", "problem"
        ]}
        return norm
    except Exception:
        return {"units": [], "keywords": [], "procedural": [], "comparative": [], "definition": [], "problem": []}


def analyze_question(q: str, cfg: PipelineConfig) -> Analysis:
    ql = q.lower().strip()
    tokens = re.findall(r"[\w\-/\.%°℃]+", ql)
    length = len(tokens)

    dom = _load_domain(cfg)
    units = set(dom.get("units", []))
    domain_kw = dom.get("keywords", [])

    has_number = bool(re.search(r"\d", ql))
    has_unit = any(u in ql for u in units)
    has_domain_kw = any(kw in ql for kw in domain_kw if kw)

    numeric_like = has_number or has_unit or has_domain_kw

    def _re_or(base: str, extras: List[str]) -> str:
        return base if not extras else base[:-1] + "|" + "|".join(map(re.escape, extras)) + ")"

    # 정수장 도메인 특화 질문 유형 분류
    is_definition = bool(re.search(_re_or(r"(정의|무엇|란|의미|개념|설명|목적|기능|특징)", dom.get("definition", [])), q))
    is_procedural = bool(re.search(_re_or(r"(방법|절차|순서|어떻게|운영|조치|설정|접속|로그인)", dom.get("procedural", [])), q))
    is_comparative = bool(re.search(_re_or(r"(비교|vs|더|높|낮|차이|장점|단점|차이점)", dom.get("comparative", [])), ql))
    is_problem = bool(re.search(_re_or(r"(문제|오류|이상|고장|원인|대응|대책|해결|증상)", dom.get("problem", [])), q))
    
    # 정수장 특화 질문 유형 추가
    is_system_info = bool(re.search(r"(시스템|플랫폼|대시보드|로그인|계정|비밀번호|주소|url)", ql))
    is_technical_spec = bool(re.search(r"(모델|알고리즘|성능|지표|입력변수|설정값|고려사항)", ql))
    is_operational = bool(re.search(r"(운영|모드|제어|알람|진단|결함|정보|현황)", ql))

    if numeric_like:
        qtype = "numeric"
    elif is_definition:
        qtype = "definition"
    elif is_procedural:
        qtype = "procedural"
    elif is_comparative:
        qtype = "comparative"
    elif is_problem:
        qtype = "problem"
    elif is_system_info:
        qtype = "system_info"
    elif is_technical_spec:
        qtype = "technical_spec"
    elif is_operational:
        qtype = "operational"
    else:
        qtype = "general"
    
    # 질문 유형에 따른 검색 가중치 조정
    if qtype in ["system_info", "technical_spec"]:
        vw, bw = 0.4, 0.6  # BM25에 더 의존 (정확한 키워드 매칭)
    elif qtype in ["operational", "procedural"]:
        vw, bw = 0.7, 0.3  # 벡터 검색에 더 의존 (의미적 유사성)
    else:
        vw, bw = cfg.rrf.vector_weight, cfg.rrf.bm25_weight

    # 키워드 토큰 수 계산 개선 (정수장 전문 용어 우선)
    domain_tokens = [t for t in tokens if any(kw in t for kw in ["ai", "플랫폼", "공정", "모델", "알고리즘", "설정", "운영", "진단", "결함", "성능", "지표"])]
    key_token_count = len([t for t in tokens if len(t) >= 2]) + len(domain_tokens)
    
    # 질문 유형에 따른 임계값 조정
    if qtype in ["system_info", "technical_spec"]:
        threshold_adj = cfg.thresholds.analyzer_threshold_delta - 0.1  # 더 관대한 필터링
    else:
        threshold_adj = cfg.thresholds.analyzer_threshold_delta

    return Analysis(
        qtype=qtype,
        length=length,
        key_token_count=key_token_count,
        rrf_vector_weight=vw,
        rrf_bm25_weight=bw,
        threshold_adj=threshold_adj,
    )

