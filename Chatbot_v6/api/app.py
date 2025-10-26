"""
FastAPI Application - API 서버

RESTful API 엔드포인트를 제공합니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import time

from config.pipeline_config import PipelineConfig
from config.environment import get_env_config
from modules.core.logger import setup_logging, get_logger
from modules.core.exceptions import ChatbotException
from modules.pipeline.rag_pipeline import RAGPipeline
from modules.core.types import Chunk

# 로깅 설정
env_config = get_env_config()
setup_logging(
    log_dir=env_config.log_dir,
    log_level=env_config.log_level,
    log_format=env_config.log_format,
)

logger = get_logger(__name__)

# FastAPI 앱
app = FastAPI(
    title="Chatbot v6 API",
    description="정수처리 챗봇 API (4가지 원칙 준수)",
    version="6.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 파이프라인 (초기화는 startup에서)
pipeline: Optional[RAGPipeline] = None


# Request/Response 모델
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 50


class Source(BaseModel):
    text: str
    score: float
    rank: int
    filename: str
    page: Optional[int] = None


class AnswerResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Source]
    metrics: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global pipeline
    
    logger.info("API server starting up")
    
    try:
        # 설정 로드
        config_path = project_root / "config" / "default.yaml"
        pipeline_config = PipelineConfig.from_file(config_path)
        
        # 더미 청크로 파이프라인 초기화 (실제로는 DB/파일에서 로드)
        # TODO: 실제 문서 로드 로직 구현
        dummy_chunks = [
            Chunk(
                doc_id="demo",
                filename="demo.pdf",
                page=1,
                start_offset=0,
                length=100,
                text="고산 정수장 AI플랫폼 URL은 waio-portal-vip:10011입니다.",
            ),
            Chunk(
                doc_id="demo",
                filename="demo.pdf",
                page=1,
                start_offset=100,
                length=100,
                text="로그인 계정은 관리자(admin)입니다.",
            ),
        ]
        
        pipeline = RAGPipeline(
            chunks=dummy_chunks,
            pipeline_config=pipeline_config,
        )
        
        logger.info("API server started successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    logger.info("API server shutting down")


@app.get("/healthz")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "service": "chatbot-v6",
        "version": "6.0.0",
    }


@app.get("/status")
async def get_status():
    """AI 서비스 상태 확인"""
    if not pipeline:
        return {
            "status": "initializing",
            "ai_available": False,
            "model_loaded": False,
            "total_pdfs": 0,
            "total_chunks": 0,
        }
    
    # 문서 통계
    total_chunks = len(pipeline.chunks) if hasattr(pipeline, 'chunks') else 0
    unique_files = len(set(c.filename for c in pipeline.chunks)) if hasattr(pipeline, 'chunks') else 0
    
    return {
        "status": "ok",
        "ai_available": True,
        "model_loaded": True,
        "total_pdfs": unique_files,
        "total_chunks": total_chunks,
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: Request, request: QuestionRequest):
    """
    질문에 대한 답변
    
    Args:
        request: 질문 요청
        
    Returns:
        답변 응답
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start_time = time.time()
    session_id = req.headers.get('X-Session-ID', str(uuid.uuid4()))
    
    try:
        logger.info(f"Received question", 
                   question=request.question,
                   session_id=session_id)
        
        # 파이프라인 실행
        answer = pipeline.ask(
            question=request.question,
            top_k=request.top_k,
        )
        
        processing_time = time.time() - start_time
        
        # 응답 변환
        sources = [
            Source(
                text=span.chunk.text[:200],
                score=span.score,
                rank=span.rank,
                filename=span.chunk.filename,
                page=span.chunk.page,
            )
            for span in answer.sources[:5]
        ]
        
        response = AnswerResponse(
            answer=answer.text,
            confidence=answer.confidence,
            sources=sources,
            metrics={
                **answer.metrics,
                "processing_time": processing_time,
                "session_id": session_id,
            },
        )
        
        logger.info("Question answered successfully",
                   confidence=answer.confidence,
                   processing_time=processing_time,
                   session_id=session_id)
        
        return response
    
    except ChatbotException as e:
        logger.error(f"Chatbot error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": e.to_dict(),
                "message": str(e),
            }
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Chatbot v6 API",
        "version": "6.0.0",
        "description": "정수처리 챗봇 API (4가지 원칙 준수)",
        "endpoints": {
            "health": "/healthz",
            "ask": "/ask (POST)",
            "docs": "/docs",
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=env_config.debug,
    )

