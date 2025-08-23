"""
TypeScript/Django 연동을 위한 FastAPI 엔드포인트

이 모듈은 PDF QA 시스템의 모든 기능을 REST API로 제공하여
TypeScript 프론트엔드와 Django 백엔드에서 쉽게 사용할 수 있도록 합니다.
"""

import os
import uuid
import tempfile
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# 핵심 모듈들 임포트
from core.pdf_processor import PDFProcessor, TextChunk
from core.vector_store import HybridVectorStore, VectorStoreInterface
from core.question_analyzer import QuestionAnalyzer, AnalyzedQuestion, ConversationItem
from core.answer_generator import AnswerGenerator, Answer, ModelType, GenerationConfig
from core.evaluator import PDFQAEvaluator, SystemEvaluation

logger = logging.getLogger(__name__)

# Pydantic 모델들 (API 스키마 정의)
class QuestionRequest(BaseModel):
    """질문 요청 모델"""
    question: str = Field(..., description="사용자 질문")
    pdf_id: str = Field(..., description="PDF 문서 식별자")
    use_conversation_context: bool = Field(True, description="이전 대화 컨텍스트 사용 여부")
    max_chunks: int = Field(5, description="검색할 최대 청크 수")
    
class ConversationHistoryItem(BaseModel):
    """대화 기록 항목 모델"""
    question: str
    answer: str
    timestamp: str
    confidence_score: float = 0.0

class QuestionResponse(BaseModel):
    """질문 응답 모델"""
    answer: str = Field(..., description="생성된 답변")
    confidence_score: float = Field(..., description="답변 신뢰도 (0-1)")
    used_chunks: List[str] = Field(..., description="사용된 문서 청크 ID들")
    generation_time: float = Field(..., description="답변 생성 시간 (초)")
    question_type: str = Field(..., description="질문 유형")
    model_name: str = Field(..., description="사용된 모델 이름")
    
class PDFUploadResponse(BaseModel):
    """PDF 업로드 응답 모델"""
    pdf_id: str = Field(..., description="생성된 PDF 식별자")
    filename: str = Field(..., description="업로드된 파일명")
    total_pages: int = Field(..., description="총 페이지 수")
    total_chunks: int = Field(..., description="생성된 청크 수")
    processing_time: float = Field(..., description="처리 시간 (초)")
    
class EvaluationRequest(BaseModel):
    """평가 요청 모델"""
    questions: List[str]
    generated_answers: List[str]
    reference_answers: List[str]
    
class ModelConfigRequest(BaseModel):
    """모델 설정 요청 모델"""
    model_type: str = Field("ollama", description="모델 타입 (ollama/huggingface/llama_cpp)")
    model_name: str = Field("llama2:7b", description="모델 이름")
    max_length: int = Field(512, description="최대 생성 길이")
    temperature: float = Field(0.7, description="생성 온도")
    top_p: float = Field(0.9, description="Top-p 샘플링")

class SystemStatusResponse(BaseModel):
    """시스템 상태 응답 모델"""
    status: str = Field(..., description="시스템 상태")
    model_loaded: bool = Field(..., description="모델 로드 상태")
    total_pdfs: int = Field(..., description="등록된 PDF 수")
    total_chunks: int = Field(..., description="총 청크 수")
    memory_usage: Dict[str, Any] = Field(..., description="메모리 사용량")

# 전역 객체들 (싱글톤 패턴)
pdf_processor: Optional[PDFProcessor] = None
vector_store: Optional[VectorStoreInterface] = None
question_analyzer: Optional[QuestionAnalyzer] = None
answer_generator: Optional[AnswerGenerator] = None
evaluator: Optional[PDFQAEvaluator] = None

# PDF 메타데이터 저장소 (실제로는 데이터베이스 사용 권장)
pdf_metadata: Dict[str, Dict] = {}

# FastAPI 앱 초기화
app = FastAPI(
    title="PDF QA System API",
    description="로컬 LLM을 사용하는 PDF 기반 질문 답변 시스템 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정 (TypeScript 프론트엔드 지원)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # React/Vue 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 의존성 함수들
def get_pdf_processor() -> PDFProcessor:
    """PDF 처리기 의존성"""
    global pdf_processor
    if pdf_processor is None:
        pdf_processor = PDFProcessor()
    return pdf_processor

def get_vector_store() -> VectorStoreInterface:
    """벡터 저장소 의존성"""
    global vector_store
    if vector_store is None:
        vector_store = HybridVectorStore()
    return vector_store

def get_question_analyzer() -> QuestionAnalyzer:
    """질문 분석기 의존성"""
    global question_analyzer
    if question_analyzer is None:
        question_analyzer = QuestionAnalyzer()
    return question_analyzer

def get_answer_generator() -> AnswerGenerator:
    """답변 생성기 의존성"""
    global answer_generator
    if answer_generator is None:
        answer_generator = AnswerGenerator()
        if not answer_generator.load_model():
            raise HTTPException(status_code=500, detail="답변 생성 모델 로드 실패")
    return answer_generator

def get_evaluator() -> PDFQAEvaluator:
    """평가기 의존성"""
    global evaluator
    if evaluator is None:
        evaluator = PDFQAEvaluator()
    return evaluator

# API 엔드포인트들

@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "PDF QA System API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    vector_store: VectorStoreInterface = Depends(get_vector_store),
    answer_generator: AnswerGenerator = Depends(get_answer_generator)
):
    """시스템 상태 조회"""
    import psutil
    import gc
    
    # 메모리 사용량 조회
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return SystemStatusResponse(
        status="running",
        model_loaded=answer_generator.llm.is_loaded,
        total_pdfs=len(pdf_metadata),
        total_chunks=sum(meta.get("total_chunks", 0) for meta in pdf_metadata.values()),
        memory_usage={
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
            "cpu_percent": psutil.cpu_percent()
        }
    )

@app.post("/upload_pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    vector_store: VectorStoreInterface = Depends(get_vector_store)
):
    """PDF 파일 업로드 및 처리"""
    import time
    
    start_time = time.time()
    
    # 파일 검증
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    # 임시 파일로 저장
    pdf_id = str(uuid.uuid4())
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # PDF 처리
        chunks, metadata = pdf_processor.process_pdf(temp_path)
        
        # 벡터 저장소에 추가
        vector_store.add_chunks(chunks)
        
        # 메타데이터 저장
        pdf_metadata[pdf_id] = {
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "total_pages": metadata.get("pages", 0),
            "total_chunks": len(chunks),
            "extraction_method": metadata.get("extraction_method", [])
        }
        
        # 백그라운드에서 벡터 저장소 저장
        background_tasks.add_task(vector_store.save)
        
        # 임시 파일 삭제
        os.unlink(temp_path)
        
        processing_time = time.time() - start_time
        
        logger.info(f"PDF 업로드 완료: {file.filename} ({len(chunks)}개 청크)")
        
        return PDFUploadResponse(
            pdf_id=pdf_id,
            filename=file.filename,
            total_pages=metadata.get("pages", 0),
            total_chunks=len(chunks),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"PDF 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류 발생: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    vector_store: VectorStoreInterface = Depends(get_vector_store),
    question_analyzer: QuestionAnalyzer = Depends(get_question_analyzer),
    answer_generator: AnswerGenerator = Depends(get_answer_generator)
):
    """질문에 대한 답변 생성"""
    
    # PDF 존재 확인
    if request.pdf_id not in pdf_metadata:
        raise HTTPException(status_code=404, detail="PDF를 찾을 수 없습니다.")
    
    try:
        # 1. 질문 분석
        analyzed_question = question_analyzer.analyze_question(
            request.question,
            use_conversation_context=request.use_conversation_context
        )
        
        # 2. 관련 문서 검색
        query_embedding = analyzed_question.embedding
        relevant_chunks = vector_store.search(
            query_embedding,
            top_k=request.max_chunks
        )
        
        if not relevant_chunks:
            raise HTTPException(status_code=404, detail="관련 문서를 찾을 수 없습니다.")
        
        # 3. 이전 대화 기록 가져오기
        conversation_history = None
        if request.use_conversation_context:
            conversation_history = question_analyzer.get_conversation_context(max_items=3)
        
        # 4. 답변 생성
        answer = answer_generator.generate_answer(
            analyzed_question,
            relevant_chunks,
            conversation_history
        )
        
        # 5. 대화 기록에 추가
        question_analyzer.add_conversation_item(
            request.question,
            answer.content,
            answer.used_chunks,
            answer.confidence_score
        )
        
        logger.info(f"질문 처리 완료: {analyzed_question.question_type.value}")
        
        return QuestionResponse(
            answer=answer.content,
            confidence_score=answer.confidence_score,
            used_chunks=answer.used_chunks,
            generation_time=answer.generation_time,
            question_type=analyzed_question.question_type.value,
            model_name=answer.model_name
        )
        
    except Exception as e:
        logger.error(f"질문 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"질문 처리 중 오류 발생: {str(e)}")

@app.get("/conversation_history")
async def get_conversation_history(
    pdf_id: str,
    max_items: int = 10,
    question_analyzer: QuestionAnalyzer = Depends(get_question_analyzer)
):
    """대화 기록 조회"""
    if pdf_id not in pdf_metadata:
        raise HTTPException(status_code=404, detail="PDF를 찾을 수 없습니다.")
    
    history = question_analyzer.get_conversation_context(max_items=max_items)
    return {"conversation_history": history}

@app.delete("/conversation_history")
async def clear_conversation_history(
    question_analyzer: QuestionAnalyzer = Depends(get_question_analyzer)
):
    """대화 기록 초기화"""
    question_analyzer.conversation_history.clear()
    return {"message": "대화 기록이 초기화되었습니다."}

@app.post("/configure_model")
async def configure_model(request: ModelConfigRequest):
    """모델 설정 변경"""
    global answer_generator
    
    try:
        # 기존 모델 언로드
        if answer_generator:
            answer_generator.unload_model()
        
        # 새 설정으로 모델 초기화
        config = GenerationConfig(
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        model_type = ModelType(request.model_type)
        answer_generator = AnswerGenerator(
            model_type=model_type,
            model_name=request.model_name,
            generation_config=config
        )
        
        # 모델 로드
        if not answer_generator.load_model():
            raise HTTPException(status_code=500, detail="새 모델 로드 실패")
        
        return {"message": f"모델 설정이 변경되었습니다: {request.model_name}"}
        
    except Exception as e:
        logger.error(f"모델 설정 변경 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모델 설정 변경 실패: {str(e)}")

@app.post("/evaluate")
async def evaluate_system(
    request: EvaluationRequest,
    evaluator: PDFQAEvaluator = Depends(get_evaluator)
):
    """시스템 성능 평가"""
    try:
        # 간단한 평가 실행 (실제로는 더 복잡한 평가 필요)
        semantic_similarities = []
        
        for gen_answer, ref_answer in zip(request.generated_answers, request.reference_answers):
            similarity = evaluator._calculate_semantic_similarity(gen_answer, ref_answer)
            semantic_similarities.append(similarity)
        
        avg_similarity = sum(semantic_similarities) / len(semantic_similarities)
        
        return {
            "evaluation_results": {
                "average_semantic_similarity": avg_similarity,
                "individual_similarities": semantic_similarities,
                "total_questions": len(request.questions)
            }
        }
        
    except Exception as e:
        logger.error(f"평가 실패: {e}")
        raise HTTPException(status_code=500, detail=f"평가 중 오류 발생: {str(e)}")

@app.get("/pdfs")
async def list_pdfs():
    """등록된 PDF 목록 조회"""
    pdf_list = []
    for pdf_id, metadata in pdf_metadata.items():
        pdf_list.append({
            "pdf_id": pdf_id,
            "filename": metadata["filename"],
            "upload_time": metadata["upload_time"],
            "total_pages": metadata["total_pages"],
            "total_chunks": metadata["total_chunks"]
        })
    
    return {"pdfs": pdf_list}

@app.delete("/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """PDF 삭제"""
    if pdf_id not in pdf_metadata:
        raise HTTPException(status_code=404, detail="PDF를 찾을 수 없습니다.")
    
    # 메타데이터에서 제거
    del pdf_metadata[pdf_id]
    
    # 실제로는 벡터 저장소에서도 해당 청크들을 제거해야 함
    # 현재 구현에서는 전체 벡터 저장소 재구성이 필요
    
    return {"message": f"PDF {pdf_id}가 삭제되었습니다."}

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Django 연동을 위한 추가 엔드포인트들

@app.post("/django/ask")
async def django_ask_question(
    question: str,
    pdf_id: str,
    conversation_history: Optional[List[ConversationHistoryItem]] = None,
    vector_store: VectorStoreInterface = Depends(get_vector_store),
    question_analyzer: QuestionAnalyzer = Depends(get_question_analyzer),
    answer_generator: AnswerGenerator = Depends(get_answer_generator)
):
    """Django에서 호출하기 쉬운 질문 엔드포인트"""
    
    # 대화 기록 복원 (필요한 경우)
    if conversation_history:
        question_analyzer.conversation_history.clear()
        for item in conversation_history:
            question_analyzer.add_conversation_item(
                item.question,
                item.answer,
                [],  # 청크 정보는 없음
                item.confidence_score
            )
    
    # 기본 ask 엔드포인트 로직 재사용
    request = QuestionRequest(
        question=question,
        pdf_id=pdf_id,
        use_conversation_context=bool(conversation_history)
    )
    
    return await ask_question(request, vector_store, question_analyzer, answer_generator)

# 에러 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"글로벌 예외 발생: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "내부 서버 오류가 발생했습니다."}
    )

# 서버 실행 함수
def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """
    FastAPI 서버 실행
    
    Args:
        host: 서버 호스트
        port: 서버 포트
        debug: 디버그 모드
    """
    uvicorn.run(
        "api.endpoints:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )

if __name__ == "__main__":
    # 개발 서버 실행
    run_server(debug=True)
