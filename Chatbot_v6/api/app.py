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

from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import time

from config.pipeline_config import PipelineConfig
from config.environment import get_env_config
from config.model_config import LLMModelConfig
from modules.core.logger import setup_logging, get_logger
from modules.core.exceptions import ChatbotException
from modules.pipeline.rag_pipeline import RAGPipeline
from modules.core.types import Chunk
from modules.analysis.question_analyzer import QuestionAnalyzer
from modules.generation.ollama_manager import OllamaManager
from modules.document.loader import DocumentLoader

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
question_analyzer: Optional[QuestionAnalyzer] = None

# 인사말 응답 리스트
GREETING_RESPONSES = [
    "안녕하세요! 👋 정수처리 AI 챗봇입니다. 무엇을 도와드릴까요?",
    "안녕하세요! 😊 정수처리 관련 질문이나 문서 검색을 도와드리겠습니다.",
    "안녕하세요! 🌊 정수장 운영, 수질 관리, 시스템 정보 등 무엇이든 물어보세요!",
    "반갑습니다! 🤖 AI 기반으로 정수처리 문서에서 답변을 찾아드립니다.",
    "안녕하세요! 💧 정수처리 기술, 공정 정보, 매뉴얼 검색 등을 도와드립니다.",
]

# API 라우터 생성 (/api 프리픽스용)
api_router = APIRouter(prefix="/api")


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
    global pipeline, question_analyzer
    
    logger.info("API server starting up")
    
    try:
        # Ollama 서버 연결 및 모델 자동 설치
        logger.info("Checking Ollama server and model availability...")
        ollama_url = f"http://{LLMModelConfig().host}:{LLMModelConfig().port}"
        ollama_manager = OllamaManager(base_url=ollama_url)
        
        # Ollama 서버가 준비될 때까지 대기 (최대 60초)
        max_wait_time = 60
        wait_interval = 2
        waited = 0
        
        while waited < max_wait_time:
            if ollama_manager.check_ollama_running():
                logger.info("Ollama server is running")
                break
            logger.info(f"Waiting for Ollama server... ({waited}/{max_wait_time}s)")
            time.sleep(wait_interval)
            waited += wait_interval
        else:
            logger.warning("Ollama server is not responding. LLM features may not work.")
        
        # 모델 자동 설치
        if ollama_manager.check_ollama_running():
            model_config = LLMModelConfig()
            model_name = model_config.model_name
            logger.info(f"Ensuring model availability: {model_name}")
            
            if ollama_manager.ensure_model_available(model_name):
                logger.info(f"Model {model_name} is ready")
            else:
                logger.warning(f"Failed to ensure model {model_name}. LLM features may not work.")
        
        # 설정 로드
        config_path = project_root / "config" / "default.yaml"
        pipeline_config = PipelineConfig.from_file(config_path)
        
        # 도메인 사전 경로 설정
        domain_dict_path = project_root / "data" / "domain_dictionary.json"
        
        # QuestionAnalyzer 초기화
        question_analyzer = QuestionAnalyzer(
            domain_dict_path=str(domain_dict_path) if domain_dict_path.exists() else None
        )
        
        # PDF 문서 자동 로드 및 임베딩
        data_dir = project_root / "data"
        chunks = []
        
        try:
            logger.info(f"Loading documents from: {data_dir}")
            doc_loader = DocumentLoader(str(data_dir))
            
            # PDF 파일 확인
            pdf_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.PDF"))
            
            if pdf_files:
                logger.info(f"Found {len(pdf_files)} PDF file(s): {[f.name for f in pdf_files]}")
                try:
                    chunks = doc_loader.load_from_directory(use_cache=True)
                    logger.info(f"Loaded {len(chunks)} chunks from {len(pdf_files)} document(s)")
                except Exception as e:
                    logger.error(f"Failed to load documents: {e}", exc_info=True)
                    logger.warning("Falling back to dummy chunks")
                    chunks = []
            else:
                logger.warning(f"No PDF files found in {data_dir}")
        
        except Exception as e:
            logger.error(f"Document loading error: {e}", exc_info=True)
            logger.warning("Falling back to dummy chunks")
            chunks = []
        
        # 문서가 없으면 더미 청크 사용
        if not chunks:
            logger.warning("No chunks loaded, using dummy chunks")
            chunks = [
                Chunk(
                    doc_id="demo",
                    filename="demo.pdf",
                    page=1,
                    start_offset=0,
                    length=100,
                    text="문서가 업로드되지 않았습니다. PDF 파일을 data/ 디렉토리에 추가해주세요.",
                ),
            ]
        
        # 파이프라인 초기화 (자동 임베딩 포함)
        logger.info("Initializing RAG pipeline with embedding...")
        pipeline = RAGPipeline(
            chunks=chunks,
            pipeline_config=pipeline_config,
        )
        
        logger.info(f"API server started successfully with {len(chunks)} chunks")
    
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


# 기존 엔드포인트 (하위 호환성을 위해 유지)
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
        logger.info(
            f"[QUESTION] Received question",
            extra={
                "question": request.question,
                "session_id": session_id,
                "top_k": request.top_k,
                "timestamp": time.time()
            }
        )
        
        # 인사말 체크
        if question_analyzer and question_analyzer.is_greeting(request.question):
            logger.info(
                f"[GREETING] Detected, returning preset response",
                extra={
                    "question": request.question,
                    "session_id": session_id
                }
            )
            
            import random
            greeting_response = random.choice(GREETING_RESPONSES)
            processing_time = time.time() - start_time
            
            # 인사말 응답 생성
            from modules.core.types import Answer
            
            answer = Answer(
                text=greeting_response,
                confidence=1.0,
                sources=[],
                metrics={
                    "total_time_ms": int(processing_time * 1000),
                    "is_greeting": True,
                    "llm_used": False,
                }
            )
        else:
            # 일반 질문 - 파이프라인 실행
            answer = pipeline.ask(
                question=request.question,
                top_k=request.top_k,
            )
            
            processing_time = time.time() - start_time
        
        # 응답 변환 (인사말은 sources가 없을 수 있음)
        sources = [
            Source(
                text=span.chunk.text[:200],
                score=span.score,
                rank=span.rank,
                filename=span.chunk.filename,
                page=span.chunk.page,
            )
            for span in answer.sources[:5]
        ] if answer.sources else []
        
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
        
        logger.info(
            f"[ANSWER] Generated successfully",
            extra={
                "question": request.question,
                "answer": answer.text[:200] + "..." if len(answer.text) > 200 else answer.text,
                "confidence": answer.confidence,
                "processing_time": processing_time,
                "session_id": session_id,
                "sources_count": len(answer.sources),
                "metrics": answer.metrics
            }
        )
        
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


# /api 프리픽스를 사용하는 엔드포인트들 (프록시 서버 호환)
@api_router.post("/ask", response_model=AnswerResponse)
async def ask_question_api(req: Request, request: QuestionRequest):
    """질문에 대한 답변 (API 프리픽스 버전)"""
    # 기존 함수와 동일한 로직 재사용
    return await ask_question(req, request)


@api_router.get("/healthz")
async def health_check_api():
    """헬스 체크 (API 프리픽스 버전)"""
    return await health_check()


@api_router.get("/status")
async def get_status_api():
    """AI 서비스 상태 확인 (API 프리픽스 버전)"""
    return await get_status()


# API 라우터를 메인 앱에 등록
app.include_router(api_router)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Chatbot v6 API",
        "version": "6.0.0",
        "description": "정수처리 챗봇 API (4가지 원칙 준수)",
        "endpoints": {
            "health": "/healthz or /api/healthz",
            "ask": "/ask (POST) or /api/ask (POST)",
            "status": "/status or /api/status",
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

