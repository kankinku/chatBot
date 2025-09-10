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

    is_definition = bool(re.search(_re_or(r"(정의|무엇|란|의미|개념|설명)", dom.get("definition", [])), q))
    is_procedural = bool(re.search(_re_or(r"(방법|절차|순서|어떻게|운영|조치)", dom.get("procedural", [])), q))
    is_comparative = bool(re.search(_re_or(r"(비교|vs|더|높|낮|차이|장점|단점)", dom.get("comparative", [])), ql))
    is_problem = bool(re.search(_re_or(r"(문제|오류|이상|고장|원인|대응|대책)", dom.get("problem", [])), q))

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
    else:
        qtype = "general"
    
    vw, bw = cfg.rrf.vector_weight, cfg.rrf.bm25_weight

    key_token_count = len([t for t in tokens if len(t) >= 2])
    threshold_adj = cfg.thresholds.analyzer_threshold_delta

    return Analysis(
        qtype=qtype,
        length=length,
        key_token_count=key_token_count,
        rrf_vector_weight=vw,
        rrf_bm25_weight=bw,
        threshold_adj=threshold_adj,
    )

