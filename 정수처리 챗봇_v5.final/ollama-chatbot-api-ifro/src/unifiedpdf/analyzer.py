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




# 도메인 사전 캐시 (메모리 누수 방지)
_domain_cache = {}
_cache_lock = None

def _load_domain(cfg: PipelineConfig) -> Dict[str, List[str]]:
    global _cache_lock
    if _cache_lock is None:
        import threading
        _cache_lock = threading.Lock()
    
    path = getattr(getattr(cfg, "domain", object()), "domain_dict_path", None)
    if not path:
        return {"units": [], "keywords": [], "procedural": [], "comparative": [], "definition": [], "problem": []}
    
    # 캐시에서 확인
    with _cache_lock:
        if path in _domain_cache:
            return _domain_cache[path]
    
    try:
        # 파일 수정 시간 확인
        path_obj = Path(path)
        if not path_obj.exists():
            return {"units": [], "keywords": [], "procedural": [], "comparative": [], "definition": [], "problem": []}
        
        mtime = path_obj.stat().st_mtime
        cache_key = f"{path}_{mtime}"
        
        with _cache_lock:
            if cache_key in _domain_cache:
                return _domain_cache[cache_key]
            
            data = json.loads(path_obj.read_text(encoding="utf-8"))
            norm = {k: [str(x).lower() for x in data.get(k, [])] for k in [
                "units", "keywords", "procedural", "comparative", "definition", "problem"
            ]}
            _domain_cache[cache_key] = norm
            
            # 캐시 크기 제한 (최대 10개)
            if len(_domain_cache) > 10:
                oldest_key = next(iter(_domain_cache))
                del _domain_cache[oldest_key]
            
            return norm
    except Exception:
        return {"units": [], "keywords": [], "procedural": [], "comparative": [], "definition": [], "problem": []}

def clear_domain_cache():
    """도메인 사전 캐시 정리"""
    global _domain_cache, _cache_lock
    if _cache_lock is None:
        import threading
        _cache_lock = threading.Lock()
    
    with _cache_lock:
        _domain_cache.clear()

def clear_regex_cache():
    """정규식 캐시 정리"""
    global _regex_cache, _regex_cache_lock
    if _regex_cache_lock is None:
        import threading
        _regex_cache_lock = threading.Lock()
    
    with _regex_cache_lock:
        _regex_cache.clear()


# 정규식 패턴 캐시 (메모리 누수 방지)
_regex_cache = {}
_regex_cache_lock = None

def _get_compiled_regex(pattern: str):
    global _regex_cache_lock
    if _regex_cache_lock is None:
        import threading
        _regex_cache_lock = threading.Lock()
    
    with _regex_cache_lock:
        if pattern not in _regex_cache:
            _regex_cache[pattern] = re.compile(pattern)
        return _regex_cache[pattern]

def _re_or(base: str, extras: List[str]) -> str:
    if not extras:
        return base
    return base[:-1] + "|" + "|".join(map(re.escape, extras)) + ")"

def analyze_question(q: str, cfg: PipelineConfig) -> Analysis:
    ql = q.lower().strip()
    
    # 컴파일된 정규식 사용
    token_pattern = _get_compiled_regex(r"[\w\-/\.%°℃]+")
    tokens = token_pattern.findall(ql)
    length = len(tokens)

    dom = _load_domain(cfg)
    units = set(dom.get("units", []))
    domain_kw = dom.get("keywords", [])

    # 숫자 검사용 정규식
    number_pattern = _get_compiled_regex(r"\d")
    has_number = bool(number_pattern.search(ql))
    has_unit = any(u in ql for u in units)
    has_domain_kw = any(kw in ql for kw in domain_kw if kw)

    numeric_like = has_number or has_unit or has_domain_kw

    # 질문 유형별 정규식 패턴 생성 및 컴파일
    definition_pattern = _get_compiled_regex(_re_or(r"(정의|무엇|란|의미|개념|설명|목적|기능|특징)", dom.get("definition", [])))
    procedural_pattern = _get_compiled_regex(_re_or(r"(방법|절차|순서|어떻게|운영|조치|설정|접속|로그인)", dom.get("procedural", [])))
    comparative_pattern = _get_compiled_regex(_re_or(r"(비교|vs|더|높|낮|차이|장점|단점|차이점)", dom.get("comparative", [])))
    problem_pattern = _get_compiled_regex(_re_or(r"(문제|오류|이상|고장|원인|대응|대책|해결|증상)", dom.get("problem", [])))
    
    # 정수장 특화 질문 유형 추가
    system_info_pattern = _get_compiled_regex(r"(시스템|플랫폼|대시보드|로그인|계정|비밀번호|주소|url)")
    technical_spec_pattern = _get_compiled_regex(r"(모델|알고리즘|성능|지표|입력변수|설정값|고려사항)")
    operational_pattern = _get_compiled_regex(r"(운영|모드|제어|알람|진단|결함|정보|현황)")

    # 정수장 도메인 특화 질문 유형 분류
    is_definition = bool(definition_pattern.search(q))
    is_procedural = bool(procedural_pattern.search(q))
    is_comparative = bool(comparative_pattern.search(ql))
    is_problem = bool(problem_pattern.search(q))
    
    # 정수장 특화 질문 유형 추가
    is_system_info = bool(system_info_pattern.search(ql))
    is_technical_spec = bool(technical_spec_pattern.search(ql))
    is_operational = bool(operational_pattern.search(ql))

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

