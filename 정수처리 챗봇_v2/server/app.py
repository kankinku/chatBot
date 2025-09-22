try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

import json
from pathlib import Path
from typing import List

from unifiedpdf.config import PipelineConfig
from unifiedpdf.facade import UnifiedPDFPipeline
from unifiedpdf.types import Chunk


def _load_corpus(path: str) -> List[Chunk]:
    p = Path(path)
    chunks: List[Chunk] = []
    if not p.exists():
        return chunks
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            chunks.append(
                Chunk(
                    doc_id=obj.get("doc_id", obj.get("filename", "doc")),
                    filename=obj.get("filename", "doc"),
                    page=obj.get("page"),
                    start_offset=int(obj.get("start", 0)),
                    length=int(obj.get("length", len(obj.get("text", "")))),
                    text=obj.get("text", ""),
                    extra=obj.get("extra", {}),
                )
            )
    return chunks


if FASTAPI_AVAILABLE:
    app = FastAPI()
    cfg = PipelineConfig()
    corpus_path = str(Path("data/corpus_v1.jsonl"))
    pipe = UnifiedPDFPipeline(_load_corpus(corpus_path), cfg)
    _warmed = False
    # Simple in-memory aggregator
    AGG = {"requests_total": 0, "no_answer_total": 0}

    class AskRequest(BaseModel):
        question: str
        mode: str = "accuracy"
        k: str = "auto"
    class BatchRequest(BaseModel):
        items: list
        mode: str = "accuracy"

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "warmed": _warmed}

    @app.post("/api/ask")
    def api_ask(req: AskRequest):
        res = pipe.ask(req.question, mode=req.mode)
        AGG["requests_total"] += 1
        AGG["no_answer_total"] += int(res.metrics.get("no_answer", 0))
        return {
            "answer": res.text,
            "confidence": res.confidence,
            "sources": [
                {
                    "filename": s.chunk.filename,
                    "page": s.chunk.page,
                    "start": s.chunk.start_offset,
                    "length": s.chunk.length,
                    "calibrated_conf": s.calibrated_conf,
                }
                for s in res.sources
            ],
            "metrics": res.metrics,
            "fallback_used": res.fallback_used,
        }

    @app.post("/api/qa/batch")
    def api_batch(req: BatchRequest):
        out = []
        for it in req.items:
            q = it.get("question", "")
            res = pipe.ask(q, mode=req.mode)
            AGG["requests_total"] += 1
            AGG["no_answer_total"] += int(res.metrics.get("no_answer", 0))
            out.append({
                "id": it.get("id"),
                "question": q,
                "answer": res.text,
                "confidence": res.confidence,
                "metrics": res.metrics,
                "fallback_used": res.fallback_used,
            })
        return {"results": out, "config_hash": cfg.config_hash()}

    @app.get("/metrics")
    def metrics():
        # Prometheus text exposition format (very small set)
        lines = []
        lines.append(f"unifiedpdf_requests_total {AGG['requests_total']}")
        lines.append(f"unifiedpdf_no_answer_total {AGG['no_answer_total']}")
        lines.append(f"unifiedpdf_config_info{{config_hash=\"{cfg.config_hash()}\"}} 1")
        return "\n".join(lines)

    @app.on_event("startup")
    def _warm_start():
        nonlocal _warmed
        try:
            _ = pipe.ask("웜업", mode="accuracy")
            _warmed = True
        except Exception:
            _warmed = False
else:
    app = None  # Placeholder; FastAPI not installed
