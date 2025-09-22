from __future__ import annotations

from typing import List, Tuple

from .analyzer import Analysis
from .config import PipelineConfig
from .merger import dedup_spans
from .types import RetrievedSpan


def choose_k(analysis: Analysis, cfg: PipelineConfig) -> int:
    if analysis.qtype in ("numeric", "comparative"):
        base = cfg.context.k_numeric
    elif analysis.qtype == "definition":
        base = max(cfg.context.k_definition_min, min(cfg.context.k_definition_max, analysis.length // 3 + 4))
    else:
        base = cfg.context.k_default
    # scale by length and key tokens
    bonus = (analysis.length // 10) + (analysis.key_token_count // 6)
    scaled = max(cfg.context.k_min, min(cfg.context.k_max, base + bonus))
    return int(scaled)


def add_neighbors(spans: List[RetrievedSpan], cfg: PipelineConfig) -> List[RetrievedSpan]:
    # Build indices for O(1) neighbor lookup
    by_para = {}
    by_page = {}
    for s in spans:
        ch = s.chunk
        by_para.setdefault((ch.doc_id, ch.extra.get("section"), ch.extra.get("paragraph_id")), []).append(s)
        by_page.setdefault((ch.doc_id, ch.page), []).append(s)

    out: List[RetrievedSpan] = []
    added = set()
    for s in spans:
        ch = s.chunk
        key = (ch.doc_id, ch.page, ch.start_offset)
        if key not in added:
            out.append(s)
            added.add(key)
        neighbor_added = False
        para_key = (ch.doc_id, ch.extra.get("section"), ch.extra.get("paragraph_id"))
        for o in by_para.get(para_key, [])[:2]:  # at most one neighbor
            if o is s:
                continue
            nkey = (o.chunk.doc_id, o.chunk.page, o.chunk.start_offset)
            if nkey not in added:
                out.append(o)
                added.add(nkey)
                neighbor_added = True
                break
        if (not neighbor_added) and cfg.context.allow_neighbor_from_adjacent_page and ch.page is not None:
            for p in (ch.page - 1, ch.page + 1):
                for o in by_page.get((ch.doc_id, p), [])[:1]:
                    nkey = (o.chunk.doc_id, o.chunk.page, o.chunk.start_offset)
                    if nkey not in added:
                        out.append(o)
                        added.add(nkey)
                        neighbor_added = True
                        break
                if neighbor_added:
                    break
    return dedup_spans(out, approx=True)
