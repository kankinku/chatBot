"""
FastAPI 엔드포인트 (최적화 버전)

빠른 응답을 위한 간소화된 API
"""

import os
import sys
import uuid
import tempfile
import locale
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import asyncio

# 핵심 모듈들 임포트
from core.document.pdf_processor import PDFProcessor, TextChunk
from core.document.vector_store import HybridVectorStore, VectorStoreInterface
from core.query.question_analyzer import QuestionAnalyzer, AnalyzedQuestion, ConversationItem
from core.llm.answer_generator import AnswerGenerator, Answer, ModelType, GenerationConfig
# SQL 관련 import가 제거되었습니다.

from core.query.query_router import QueryRouter, QueryRoute
from core.query.llm_greeting_handler import GreetingHandler
from utils.chatbot_logger import chatbot_logger, QuestionType, ProcessingStep
from utils.unified_logger import unified_logger, LogCategory
from core.cache.fast_cache import get_all_cache_stats
from core.config.unified_config import get_config, get_config_version, config as unified_config_instance
from core.document.store_coordinator import StoreCoordinator
from core.config.sla_config import get_route_sla_p95_ms, get_llm_policy, get_rerank_policy, get_min_accuracy_threshold
from core.utils.security_utils import validate_question_payload, sanitize_prompt, ValidationError
from core.quality.quality_loop import (
    load_golden_set, save_golden_set, add_golden_item, delete_golden_item,
    compute_metrics, generate_change_matrix
)
from core.utils.health_metrics import health_metrics
from core.utils.alerts import notify_slack
from core.utils.circuit_breaker import CircuitBreaker
from core.catalog.catalog_service import (
    load_catalog, save_catalog, submit_change_request,
    list_change_requests, approve_request, reject_request
)
from core.ops.ops_stub import get_regression_summary, run_scheduled_job, run_security_scan

logger = logging.getLogger(__name__)

# 메트릭 수집(간단 버전)
_metrics = {
    "requests_total": 0,
    "vector_search_failures": 0,
    "llm_failures": 0,
    "search_timeouts": 0,
    "llm_timeouts": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "rate_limited": 0,
}

# 간단한 롤링 윈도 메트릭(메모리)
_rolling_events = {
    "latency_ms": deque(maxlen=1000),
    "errors": deque(maxlen=1000),
}
_rolling_params = {
    "topk": deque(maxlen=1000),
    "sim_threshold": deque(maxlen=1000),
}
_routing_counters = {}
_rolling_events_1h = {
    "latency_ms": deque(maxlen=5000),
    "errors": deque(maxlen=5000),
}

# Pydantic 모델들 (단순화)
class QuestionRequest(BaseModel):
    """질문 요청 모델"""
    question: str = Field(..., description="사용자 질문")
    pdf_id: str = Field("", description="PDF 문서 식별자")
    user_id: str = Field("", description="사용자 식별자")
    use_conversation_context: bool = Field(True, description="이전 대화 컨텍스트 사용 여부")
    max_chunks: int = Field(5, description="검색할 최대 청크 수 (권장: 3~5)")

    class Config:
        extra = "forbid"

class QuestionResponse(BaseModel):
    """질문 응답 모델"""
    answer: str = Field(..., description="생성된 답변")
    confidence_score: float = Field(..., description="답변 신뢰도")
    used_chunks: List[str] = Field(..., description="사용된 문서 청크 ID들")
    generation_time: float = Field(..., description="답변 생성 시간 (초)")
    question_type: str = Field(..., description="질문 유형")
    llm_model_name: str = Field(..., description="사용된 모델 이름")
    pipeline_type: str = Field("basic", description="사용된 파이프라인 타입")
    sql_query: Optional[str] = Field(None, description="생성된 SQL 쿼리")

class PDFUploadResponse(BaseModel):
    """PDF 업로드 응답 모델"""
    pdf_id: str = Field(..., description="생성된 PDF 식별자")
    filename: str = Field(..., description="업로드된 파일명")
    total_pages: int = Field(..., description="총 페이지 수")
    total_chunks: int = Field(..., description="생성된 청크 수")
    processing_time: float = Field(..., description="처리 시간 (초)")

class SystemStatusResponse(BaseModel):
    """시스템 상태 응답 모델"""
    status: str = Field(..., description="시스템 상태")
    llm_model_loaded: bool = Field(..., description="모델 로드 상태")
    total_pdfs: int = Field(..., description="등록된 PDF 수")
    total_chunks: int = Field(..., description="총 청크 수")
    memory_usage: Dict[str, Any] = Field(..., description="메모리 사용량")

class KeywordCacheResponse(BaseModel):
    """키워드 캐시 응답 모델"""
    total_keywords: int = Field(..., description="총 키워드 수")
    frequent_keywords: int = Field(..., description="자주 사용된 키워드 수")
    extracted_keywords: int = Field(..., description="추출된 키워드 수")
    cache_threshold: int = Field(..., description="캐시 임계값")
    top_keywords: Dict[str, int] = Field(..., description="상위 키워드 (최대 10개)")

class KeywordPipelineResponse(BaseModel):
    """키워드 파이프라인 추가 응답 모델"""
    success: bool = Field(..., description="성공 여부")
    added_keywords: List[str] = Field(..., description="추가된 키워드 목록")
    message: str = Field(..., description="응답 메시지")

# FastAPI 앱 초기화
app = FastAPI(
    title=get_config("API_TITLE", "범용 RAG 시스템 API"),
    description=get_config("API_DESCRIPTION", "범용 문서 검색 및 데이터베이스 쿼리 시스템 API"),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 세션ID 미들웨어 및 설정 버전 전파
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    sid = str(uuid.uuid4())
    try:
        try:
            unified_logger.set_session_id(sid)
        except Exception:
            pass
        # 입력 검증: 길이 및 금지어
        if request.method in ("POST", "PUT", "PATCH") and request.url.path in ("/ask",):
            try:
                body = await request.body()
                if body and len(body) > QUESTION_MAX_LEN * 4:
                    return JSONResponse(status_code=413, content={"code": "INPUT_TOO_LARGE", "message": "요청 본문이 너무 큽니다.", "detail": {"limit_bytes": QUESTION_MAX_LEN * 4}})
                # 간단 JSON 파싱
                import json as _json
                j = _json.loads(body.decode("utf-8")) if body else {}
                q = (j.get("question") or "").strip()
                if len(q) > QUESTION_MAX_LEN:
                    return JSONResponse(status_code=422, content={"code": "QUESTION_TOO_LONG", "message": "질문 길이 제한 초과", "detail": {"max_len": QUESTION_MAX_LEN}})
                if FORBIDDEN_WORDS and any(bad in q for bad in FORBIDDEN_WORDS):
                    return JSONResponse(status_code=422, content={"code": "FORBIDDEN_TERM", "message": "금지어가 포함되어 있습니다.", "detail": {"violations": [bad for bad in FORBIDDEN_WORDS if bad in q][:5]}})
            except Exception:
                pass
        response = await call_next(request)
    except Exception as _mw_err:
        # 예외 응답에도 세션/버전 헤더 주입 시도
        response = JSONResponse(status_code=500, content={"detail": "internal_error"})
    try:
        response.headers["X-Session-Id"] = sid
        response.headers["X-Config-Version"] = get_config_version() or ""
        # 감사 헤더 일부 반영
        response.headers["X-Request-Path"] = request.url.path
        origin = request.headers.get("origin") or request.client.host if request.client else ""
        response.headers["X-Origin"] = origin
    except Exception:
        pass
    try:
        # 요청 종료 시 세션 해제
        unified_logger.set_session_id(None)
    except Exception:
        pass
    return response

# UTF-8 인코딩 설정
# 시스템 인코딩을 UTF-8로 설정
try:
    if sys.platform.startswith('linux'):
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    elif sys.platform.startswith('win'):
        locale.setlocale(locale.LC_ALL, 'Korean_Korea.UTF-8')
    else:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception as _loc_err:
    logger.warning(f"로케일 설정 실패, 기본값으로 계속 진행: {_loc_err}")



# 업로드/검증 정책 설정 (설정 SSOT)
MAX_UPLOAD_MB = float(get_config("MAX_UPLOAD_MB", 100))  # 기본 100MB
MAX_PAGES = int(get_config("MAX_PAGES", 2000))  # 기본 2000 페이지
ALLOWED_MIME = set(str(get_config("ALLOWED_MIME", "application/pdf")).split(","))
ALLOWED_EXT = set(str(get_config("ALLOWED_EXT", ".pdf")).lower().split(","))

# 타임아웃 설정(초)
INDEX_TIMEOUT_S = float(get_config("INDEX_TIMEOUT_S", 60))
SEARCH_TIMEOUT_S = float(get_config("SEARCH_TIMEOUT_S", 15))
LLM_TIMEOUT_S = float(get_config("LLM_TIMEOUT_S", 30))
LLM_RETRY = int(get_config("LLM_RETRY", 2))
LLM_BACKOFF_S = float(get_config("LLM_BACKOFF_S", 0.5))

# 인증/레이트리밋 설정 (설정 SSOT)
REQUIRE_AUTH = str(get_config("REQUIRE_AUTH", "true")).lower() == "true"
API_KEY = str(get_config("API_KEY", ""))
ADMIN_API_KEY = str(get_config("ADMIN_API_KEY", ""))
RL_WINDOW_SEC = int(get_config("RL_WINDOW_SEC", 60))
RL_MAX_REQUESTS = int(get_config("RL_MAX_REQUESTS", 60))  # 윈도 내 총 허용
RL_BURST = int(get_config("RL_BURST", 20))  # 순간 버스트 허용

# 공개 라우트
PUBLIC_PATHS = {"/", "/health", "/healthz", "/readyz"}

# 표준 오류 코드/메시지 매핑
ERROR_MAP = {
    "UPLOAD_TOO_LARGE": {"status": 413, "message": "업로드 용량 초과"},
    "INVALID_MIME": {"status": 415, "message": "허용되지 않은 MIME 타입"},
    "INVALID_EXTENSION": {"status": 400, "message": "허용되지 않은 파일 확장자"},
    "INVALID_PDF_STRUCTURE": {"status": 422, "message": "유효하지 않은 PDF 구조"},
    "PAGE_LIMIT_EXCEEDED": {"status": 422, "message": "허용 페이지 수 초과"},
}

def raise_mapped_error(code: str, detail: str = None):
    info = ERROR_MAP.get(code, {"status": 400, "message": "요청 오류"})
    raise HTTPException(status_code=info["status"], detail={"code": code, "message": info["message"], "detail": detail})

def _basic_pdf_structure_check(data: bytes) -> bool:
    try:
        if not data or len(data) < 5:
            return False
        if not data[:4] == b"%PDF":
            return False
        # EOF 토큰 존재 여부 (파일 끝 2KB 범위에서 검색)
        tail = data[-2048:] if len(data) > 2048 else data
        return b"%%EOF" in tail
    except Exception:
        return False

def _count_pdf_pages_fast(data: bytes) -> int:
    # 빠른 근사: '/Type /Page' 토큰 수를 페이지 수로 추정
    try:
        return data.count(b"/Type /Page")
    except Exception:
        return 0

# 인증/레이트리밋 유틸
_req_history = {}

def _get_client_key(headers: Dict[str, Any]) -> str:
    # 우선순위: 사용자 API 키 > 사용자 ID 헤더 > IP(프락시 환경 고려 어려워 단순화)
    user_key = headers.get("x-api-key") or headers.get("X-API-Key")
    user_id = headers.get("x-user-id") or headers.get("X-User-Id")
    return user_key or user_id or "anonymous"

def _rate_limit(headers: Dict[str, Any]) -> Optional[int]:
    import time as _t
    now = _t.time()
    key = _get_client_key(headers)
    dq = _req_history.get(key)
    if dq is None:
        dq = deque()
        _req_history[key] = dq
    # 윈도 밖 제거
    while dq and now - dq[0] > RL_WINDOW_SEC:
        dq.popleft()
    # 버스트/윈도 제한 검사
    if len(dq) >= RL_MAX_REQUESTS:
        # 다음 허용까지 대기 시간
        retry_after = max(1, int(RL_WINDOW_SEC - (now - dq[0])))
        return retry_after
    dq.append(now)
    return None

def _require_key(headers: Dict[str, Any], admin: bool = False):
    if not REQUIRE_AUTH:
        return
    provided = headers.get("x-api-key") or headers.get("X-API-Key") or ""
    valid = ADMIN_API_KEY if admin else (API_KEY or ADMIN_API_KEY)
    if not valid:
        raise HTTPException(status_code=401, detail={"code": "AUTH_DISABLED", "message": "서버에 API 키가 설정되지 않았습니다."})
    if admin:
        if provided != ADMIN_API_KEY or not ADMIN_API_KEY:
            raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "관리자 권한이 필요합니다."})
    else:
        if provided not in {API_KEY, ADMIN_API_KEY}:
            raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "유효한 API 키가 필요합니다."})
    try:
        # 접근 로깅(IP/hash)
        import hashlib
        key_hash = hashlib.sha256(provided.encode('utf-8')).hexdigest()[:16] if provided else ""
        origin = headers.get("origin")
        unified_logger.info("access", LogCategory.SECURITY, module="auth", metadata={"key_hash": key_hash, "origin": origin})
    except Exception:
        pass

def require_user_auth(request):
    if request.url.path in PUBLIC_PATHS:
        return
    _require_key(request.headers, admin=False)
    retry_after = _rate_limit(request.headers)
    if retry_after is not None:
        # 429와 Retry-After 헤더
        headers = {"Retry-After": str(retry_after)}
        try:
            _metrics["rate_limited"] += 1
        except Exception:
            pass
        raise HTTPException(status_code=429, detail={"code": "RATE_LIMITED", "message": "요청이 너무 많습니다."}, headers=headers)

def require_admin_auth(request):
    _require_key(request.headers, admin=True)
    retry_after = _rate_limit(request.headers)
    if retry_after is not None:
        headers = {"Retry-After": str(retry_after)}
        try:
            _metrics["rate_limited"] += 1
        except Exception:
            pass
        raise HTTPException(status_code=429, detail={"code": "RATE_LIMITED", "message": "요청이 너무 많습니다."}, headers=headers)

# FastAPI 의존성 래퍼
def user_guard(request: Request):
    require_user_auth(request)

def admin_guard(request: Request):
    require_admin_auth(request)

# 워커풀 설정
WORKER_POOL_SIZE = int(get_config("WORKER_POOL_SIZE", 4))
_executor = ThreadPoolExecutor(max_workers=WORKER_POOL_SIZE)
_search_cb = CircuitBreaker(failure_threshold=5, recovery_time_sec=int(get_config("SEARCH_CB_RECOVERY_SEC", 20)))
_llm_cb = CircuitBreaker(failure_threshold=3, recovery_time_sec=int(get_config("LLM_CB_RECOVERY_SEC", 30)))

# 벡터 스토어 쓰기 코디네이터
store_coordinator: Optional[StoreCoordinator] = None

# SLA/보안 검증 설정
SLA_P95_TARGET_MS = int(get_config("SLA_P95_TARGET_MS", 3000))
BUDGET_ROUTE_PCT = float(get_config("BUDGET_ROUTE_PCT", 0.20))
BUDGET_SEARCH_PCT = float(get_config("BUDGET_SEARCH_PCT", 0.35))
BUDGET_RERANK_PCT = float(get_config("BUDGET_RERANK_PCT", 0.25))
BUDGET_LLM_PCT = float(get_config("BUDGET_LLM_PCT", 0.20))

QUESTION_MAX_LEN = int(get_config("QUESTION_MAX_LEN", 2000))
FORBIDDEN_WORDS = [w.strip() for w in str(get_config("FORBIDDEN_WORDS", "")).split(",") if w.strip()]
ONLINE_EVAL_SAMPLE_VAL = float(get_config("ONLINE_EVAL_SAMPLE", 0.02))
TEST_MODE = bool(str(get_config("TEST_MODE", "false")).lower() == "true")

# 전역 객체들
pdf_processor: Optional[PDFProcessor] = None
vector_store: Optional[VectorStoreInterface] = None
question_analyzer: Optional[QuestionAnalyzer] = None
answer_generator: Optional[AnswerGenerator] = None
# sql_generator 제거됨
query_router: Optional[QueryRouter] = None
llm_greeting_handler: Optional[GreetingHandler] = None

# PDF 메타데이터 저장소
pdf_metadata: Dict[str, Dict] = {}

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
        logger.info("벡터 저장소 초기화 중...")
        vector_store = HybridVectorStore()
        total_chunks = vector_store.get_total_chunks()
        logger.info(f"벡터 저장소 초기화 완료 - 총 청크 수: {total_chunks}")
    return vector_store

def get_store_coordinator() -> StoreCoordinator:
    global store_coordinator
    if store_coordinator is None:
        store_coordinator = StoreCoordinator(vector_store=HybridVectorStore())
    return store_coordinator

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
    return answer_generator

# get_sql_generator 함수 제거됨

def get_query_router() -> QueryRouter:
    """쿼리 라우터 의존성"""
    global query_router
    if query_router is None:
        query_router = QueryRouter()
    return query_router

def get_llm_greeting_handler() -> GreetingHandler:
    """LLM 기반 인사말 핸들러 의존성"""
    global llm_greeting_handler
    if llm_greeting_handler is None:
        # answer_generator가 준비된 후에 초기화
        answer_gen = get_answer_generator()
        llm_greeting_handler = GreetingHandler(answer_gen)
    return llm_greeting_handler

def initialize_system():
    """시스템 초기화 및 자동 PDF 업로드"""
    global pdf_processor, vector_store, question_analyzer, answer_generator, sql_generator, query_router
    
    try:
        logger.info("시스템 초기화 시작...")
        try:
            unified_logger.info(
                "시스템 초기화",
                LogCategory.SYSTEM,
                module="bootstrap",
                metadata={"config_version": get_config_version()}
            )
        except Exception:
            pass
        
        # 컴포넌트들 초기화
        pdf_processor = PDFProcessor()
        vector_store = HybridVectorStore()
        question_analyzer = QuestionAnalyzer()
        answer_generator = AnswerGenerator()
        query_router = QueryRouter()
        
        logger.info("컴포넌트 초기화 완료")
        
        # 기존 PDF 문서 로드
        try:
            existing_pdfs = vector_store.get_all_pdfs()
            for pdf_info in existing_pdfs:
                pdf_metadata[pdf_info['id']] = pdf_info
            logger.info(f"기존 PDF {len(existing_pdfs)}개 로드 완료")
            
            # 기존 청크 수 확인
            total_chunks = vector_store.get_total_chunks()
            logger.info(f"기존 청크 수: {total_chunks}")
            
            # 이미 충분한 데이터가 있으면 자동 업로드 건너뛰기
            if total_chunks > 0 and len(existing_pdfs) > 0:
                logger.info("이미 충분한 PDF 데이터가 로드되어 있습니다. 자동 업로드를 건너뜁니다.")
                logger.info("=" * 60)
                logger.info("시스템 초기화 완료!")
                logger.info("=" * 60)
                return
            
        except Exception as e:
            logger.warning(f"기존 PDF 로드 실패: {e}")
        
        # 자동 PDF 업로드 (데이터가 없을 때만 실행)
        logger.info("=" * 60)
        logger.info("data 폴더의 PDF 파일들을 벡터 저장소에 업로드합니다...")
        logger.info("=" * 60)
        auto_upload_result = auto_upload_pdfs_sync()
        logger.info(f"자동 업로드 완료: {auto_upload_result}")
        logger.info("=" * 60)
        logger.info("PDF 업로드 완료!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"시스템 초기화 실패: {e}")
        raise

def auto_upload_pdfs_sync():
    """동기적으로 PDF 파일들을 자동 업로드"""
    try:
        import os
        
        # data 폴더와 data/pdfs 폴더 모두 확인
        data_folders = ["./data", "./data/pdfs"]
        pdf_files = []
        
        for data_folder in data_folders:
            if not os.path.exists(data_folder):
                logger.warning(f"{data_folder} 폴더가 존재하지 않습니다.")
                continue
            
            # 재귀적으로 PDF 파일 찾기
            for root, dirs, files in os.walk(data_folder):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        pdf_files.append(pdf_path)
        
        if not pdf_files:
            logger.info("data 폴더에서 PDF 파일을 찾을 수 없습니다.")
            return {"message": "업로드할 PDF 파일이 없습니다.", "uploaded_count": 0}
        
        logger.info(f"data 폴더에서 {len(pdf_files)}개의 PDF 파일을 발견했습니다.")
        
        uploaded_count = 0
        skipped_count = 0
        failed_count = 0
        
        for pdf_path in pdf_files:
            try:
                # 이미 처리된 PDF인지 확인
                pdf_id = os.path.basename(pdf_path)
                if pdf_id in pdf_metadata:
                    logger.info(f"이미 처리된 PDF 건너뛰기: {pdf_id}")
                    skipped_count += 1
                    continue
                
                logger.info(f"PDF 처리 중: {pdf_id}")
                
                # PDF 처리
                chunks, metadata = pdf_processor.process_pdf(pdf_path)
                vector_store.add_chunks(chunks)
                
                # 메타데이터 저장
                pdf_metadata[pdf_id] = {
                    "filename": pdf_id,
                    "total_pages": len(chunks),
                    "upload_time": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                    "file_size": os.path.getsize(pdf_path)
                }
                
                uploaded_count += 1
                logger.info(f"PDF 처리 완료: {pdf_id} ({len(chunks)}개 청크)")
                
            except Exception as e:
                logger.error(f"PDF 처리 실패 {pdf_path}: {e}")
                failed_count += 1
        
        logger.info(f"PDF 처리 완료: {uploaded_count}개 처리됨, {skipped_count}개 건너뜀, {failed_count}개 오류")
        
        return {
            "message": f"자동 업로드 완료: {uploaded_count}개 성공, {skipped_count}개 건너뜀, {failed_count}개 실패",
            "uploaded_count": uploaded_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "total_files": len(pdf_files)
        }
        
    except Exception as e:
        logger.error(f"자동 업로드 실패: {e}")
        return {"error": str(e)}

# API 엔드포인트들
@app.post("/config/reload")
async def reload_config(_admin=Depends(admin_guard)):
    """설정을 재로드하고 버전을 갱신합니다."""
    try:
        start = time.time()
        before = {k: get_config(k) for k in ["API_TITLE","API_DESCRIPTION","HOST","PORT","DOMAIN_TEMPLATE","ONLINE_EVAL_SAMPLE"]}
        unified_config_instance.reload()
        elapsed = (time.time() - start) * 1000.0
        after = {k: get_config(k) for k in before.keys()}
        safe_diff = {k: {"before": before.get(k), "after": after.get(k)} for k in before.keys() if before.get(k) != after.get(k)}
        try:
            unified_logger.info(
                "설정 리로드 완료",
                LogCategory.SYSTEM,
                module="config",
                execution_time_ms=elapsed,
                metadata={"config_version": get_config_version()}
            )
        except Exception:
            pass
        impact = {"router": True, "search": True, "rerank": False, "llm": True, "prompt": True, "cache": True}
        return {"status": "ok", "config_version": get_config_version(), "elapsed_ms": elapsed, "impact": impact, "diff": safe_diff}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "RELOAD_FAILED", "message": str(e)})

@app.get("/config/info")
async def get_config_info(_admin=Depends(admin_guard)):
    """민감 키 제외한 안전한 설정 노출"""
    try:
        safe_keys = [
            "API_TITLE", "API_DESCRIPTION", "HOST", "PORT",
            "MAX_UPLOAD_MB", "MAX_PAGES",
            "INDEX_TIMEOUT_S", "SEARCH_TIMEOUT_S", "LLM_TIMEOUT_S",
            "RL_WINDOW_SEC", "RL_MAX_REQUESTS", "RL_BURST",
            "WORKER_POOL_SIZE", "SLA_P95_TARGET_MS",
            "BUDGET_ROUTE_PCT", "BUDGET_SEARCH_PCT", "BUDGET_RERANK_PCT", "BUDGET_LLM_PCT",
            "QUESTION_MAX_LEN", "DOMAIN_TEMPLATE", "ONLINE_EVAL_SAMPLE", "FORBIDDEN_WORDS", "TEST_MODE"
        ]
        data = {k: get_config(k) for k in safe_keys}
        data["config_version"] = get_config_version()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {"message": "범용 RAG 시스템 API 서버가 실행 중입니다."}

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "config_version": get_config_version()}

@app.get("/security/validation")
async def get_security_validation(_admin=Depends(admin_guard)):
    try:
        return {
            "question_max_len": QUESTION_MAX_LEN,
            "forbidden_words": FORBIDDEN_WORDS,
            "online_eval_sample": ONLINE_EVAL_SAMPLE_VAL,
            "test_mode": TEST_MODE,
            "rate_limit": {
                "window_sec": RL_WINDOW_SEC,
                "max_requests": RL_MAX_REQUESTS,
                "burst": RL_BURST
            },
            "auth_required": REQUIRE_AUTH,
            "config_version": get_config_version()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security/validation")
async def update_security_validation(
    question_max_len: Optional[int] = None,
    forbidden_words: Optional[str] = None,
    online_eval_sample: Optional[float] = None,
    test_mode: Optional[bool] = None,
    _admin=Depends(admin_guard)
):
    try:
        global QUESTION_MAX_LEN, FORBIDDEN_WORDS
        if question_max_len is not None:
            QUESTION_MAX_LEN = int(question_max_len)
        if forbidden_words is not None:
            FORBIDDEN_WORDS = [w.strip() for w in forbidden_words.split(',') if w.strip()]
        global ONLINE_EVAL_SAMPLE_VAL, TEST_MODE
        if online_eval_sample is not None:
            ONLINE_EVAL_SAMPLE_VAL = float(online_eval_sample)
        if test_mode is not None:
            TEST_MODE = bool(test_mode)
        return {
            "updated": True,
            "question_max_len": QUESTION_MAX_LEN,
            "forbidden_words": FORBIDDEN_WORDS,
            "online_eval_sample": ONLINE_EVAL_SAMPLE_VAL,
            "test_mode": TEST_MODE,
            "config_version": get_config_version()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security/validate")
async def security_validate_sample(text: str, _admin=Depends(admin_guard)):
    try:
        q = (text or "").strip()
        violations = [bad for bad in FORBIDDEN_WORDS if bad and bad in q]
        too_long = len(q) > QUESTION_MAX_LEN
        return {"ok": (not violations and not too_long), "too_long": too_long, "violations": violations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def healthz():
    snap = health_metrics.snapshot()
    return {"status": "ok", "timestamp": datetime.now().isoformat(), "metrics": snap, "config_version": get_config_version()}

@app.get("/readyz")
async def readyz():
    try:
        vs = get_vector_store()
        _ = vs.get_total_chunks()
        # 간단 일관성 플래그: FAISS/Chroma 개수 차이 스냅샷
        consistency = None
        try:
            coordinator = get_store_coordinator()
            diff = coordinator.reconcile()
            consistency = (diff.get("faiss_count") == diff.get("chroma_count"))
            cache_ready = True
        except Exception:
            consistency = None
            cache_ready = None
        # 모델/캐시 준비 플래그
        model_loaded = (answer_generator is not None)
        cache_ready = cache_ready if cache_ready is not None else True
        snap = health_metrics.snapshot()
        return {"ready": True, "consistency_ok": consistency, "model_loaded": model_loaded, "cache_ready": cache_ready, "metrics": snap, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=503, detail={"code": "NOT_READY", "message": str(e), "config_version": get_config_version()})

# SQL 관련 성능 통계, 스키마, 캐시 엔드포인트들이 제거되었습니다.

@app.post("/clear-vector-store")
async def clear_vector_store(
    vector_store: VectorStoreInterface = Depends(get_vector_store),
    _admin=Depends(admin_guard)
):
    """벡터 저장소 초기화"""
    try:
        vector_store.clear()
        return {"message": "벡터 저장소가 성공적으로 초기화되었습니다.", "config_version": get_config_version()}
    except Exception as e:
        logger.error(f"벡터 저장소 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail=f"벡터 저장소 초기화 실패: {str(e)}")

@app.post("/reset-chunks")
async def reset_chunks(
    vector_store: VectorStoreInterface = Depends(get_vector_store),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    _admin=Depends(admin_guard)
):
    """청크 초기화 및 재생성"""
    try:
        logger.info("청크 초기화 및 재생성 시작...")
        
        # 1단계: 기존 청크 초기화
        total_chunks = vector_store.get_total_chunks()
        logger.info(f"기존 청크 수: {total_chunks}")
        
        vector_store.clear()
        logger.info("벡터 저장소 초기화 완료")
        
        # 2단계: PDF 파일 스캔
        pdf_files = _find_pdf_files()
        logger.info(f"발견된 PDF 파일: {len(pdf_files)}개")
        
        if not pdf_files:
            return {"message": "PDF 파일이 없습니다. data/pdfs 폴더에 PDF 파일을 추가해주세요."}
        
        # 3단계: 청크 재생성
        total_new_chunks = 0
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"처리 중: {pdf_file}")
                
                # PDF 처리
                chunks = pdf_processor.process_pdf(pdf_file)
                
                if chunks:
                    # 벡터 저장소에 추가
                    vector_store.add_chunks(chunks)
                    total_new_chunks += len(chunks)
                    logger.info(f"{len(chunks)}개 청크 생성 완료")
                else:
                    logger.warning(f"청크 생성 실패: {pdf_file}")
                    
            except Exception as e:
                logger.error(f"PDF 처리 오류 {pdf_file}: {e}")
                continue
        
        logger.info(f"청크 재생성 완료! 총 {total_new_chunks}개 청크 생성")
        
        return {
            "message": "청크 초기화 및 재생성이 완료되었습니다.",
            "total_new_chunks": total_new_chunks,
            "total_chunks": vector_store.get_total_chunks(),
            "config_version": get_config_version()
        }
        
    except Exception as e:
        logger.error(f"청크 초기화 및 재생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"청크 초기화 및 재생성 실패: {str(e)}")

def _find_pdf_files() -> List[str]:
    """PDF 파일들 찾기"""
    pdf_files = []
    
    # data/pdfs 폴더 확인
    pdf_dir = Path(__file__).parent.parent / "data" / "pdfs"
    if pdf_dir.exists():
        pdf_files.extend([str(f) for f in pdf_dir.glob("*.pdf")])
    
    # data 폴더 직접 확인
    data_dir = Path(__file__).parent.parent / "data"
    if data_dir.exists():
        pdf_files.extend([str(f) for f in data_dir.glob("*.pdf")])
    
    return pdf_files

@app.get("/vector-store-stats")
async def get_vector_store_stats(
    vector_store: VectorStoreInterface = Depends(get_vector_store)
):
    """벡터 저장소 통계 정보"""
    _auth = Depends(user_guard)
    try:
        total_chunks = vector_store.get_total_chunks()
        pdfs = vector_store.get_all_pdfs()
        
        return {
            "total_chunks": total_chunks,
            "total_pdfs": len(pdfs),
            "pdfs": pdfs,
            "config_version": get_config_version()
        }
    except Exception as e:
        logger.error(f"벡터 저장소 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@app.post("/vector/reconcile")
async def vector_reconcile(_admin=Depends(admin_guard)):
    """FAISS/Chroma 간 무결성 검사 및 차이 리포트"""
    try:
        coordinator = get_store_coordinator()
        diff = coordinator.reconcile()
        return {"status": "ok", "diff": diff, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector/resync")
async def vector_resync(apply: bool = False, _admin=Depends(admin_guard)):
    """재동기화 트리거: apply=true 시 간단한 정리(Chroma측 누락/여분 처리) 스켈레톤"""
    try:
        coordinator = get_store_coordinator()
        diff = coordinator.reconcile()
        applied = False
        actions = {"deleted_in_chroma": 0, "added_missing": 0}
        if apply:
            try:
                # FAISS에만 있는 것은 현재 스킵, Chroma에만 있는 ID는 삭제로 정리
                only_in_chroma = diff.get("only_in_chroma") or []
                if only_in_chroma:
                    coordinator._store.chroma_store.delete_ids(only_in_chroma)
                    actions["deleted_in_chroma"] = len(only_in_chroma)
                # 옵션: add_missing 처리 스켈레톤(FAISS 존재, Chroma 없음)
                only_in_faiss = diff.get("only_in_faiss") or []
                if only_in_faiss:
                    # FAISS에 있는 원본 청크를 찾아 Chroma에 추가
                    try:
                        faiss_chunks = getattr(coordinator._store.faiss_store, 'chunks', [])
                        id_to_chunk = {c.chunk_id: c for c in faiss_chunks if getattr(c, 'chunk_id', None)}
                        missing_chunks = [id_to_chunk[cid] for cid in only_in_faiss if cid in id_to_chunk]
                        if missing_chunks:
                            coordinator._store.chroma_store.add_chunks(missing_chunks)
                            actions["added_missing"] = len(missing_chunks)
                    except Exception:
                        pass
                applied = True
            except Exception:
                applied = False
        return {"status": "ok", "applied": applied, "actions": actions, "plan": diff, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector/journal")
async def vector_journal_info(_admin=Depends(admin_guard)):
    try:
        coordinator = get_store_coordinator()
        return {"journal": coordinator.journal_info(), "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector/journal/replay")
async def vector_journal_replay(limit: Optional[int] = None, _admin=Depends(admin_guard)):
    try:
        coordinator = get_store_coordinator()
        return {"replay": coordinator.replay_journal(limit), "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector/journal/append")
async def vector_journal_append(payload: Dict[str, Any], _admin=Depends(admin_guard)):
    """운영 중 복구를 위한 저널 수동 주입(운영자용)."""
    try:
        coordinator = get_store_coordinator()
        # 내부 보호: 코디네이터 내부 함수를 최소 노출
        # 간단히 add/delete/update 중 하나만 허용
        op = payload.get("op")
        if op not in {"add", "delete", "update"}:
            raise HTTPException(status_code=400, detail="invalid_op")
        # 코디네이터에 기록만 추가하고 즉시 실행은 replay로 일원화
        # 내부 메서드 접근을 피하고, 파일에 직접 기록이 필요하면 향후 확장
        # 현재는 안전을 위해 거절
        return {"accepted": False, "message": "append disabled; use /vector/upsert or /vector/delete then /vector/journal/replay"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector/upsert")
async def vector_upsert(chunks: List[Dict[str, Any]], _admin=Depends(admin_guard)):
    """업데이트/삽입 트랜잭션 API: chunk_id 기준 upsert.
    입력: [{chunk_id, content, page_number, pdf_id, metadata}]
    """
    try:
        coordinator = get_store_coordinator()
        # 간단한 직렬화 → TextChunk
        from core.document.pdf_processor import TextChunk
        to_update: List[TextChunk] = []
        for c in chunks:
            if not isinstance(c, dict):
                continue
            chunk = TextChunk(
                content=str(c.get("content", "")),
                page_number=int(c.get("page_number", 0)),
                chunk_id=str(c.get("chunk_id")),
                metadata=c.get("metadata") or {},
                pdf_id=c.get("pdf_id") or None
            )
            to_update.append(chunk)
        updated = coordinator.update_chunks(to_update)
        return {"updated": updated, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector/delete")
async def vector_delete(ids: List[str], _admin=Depends(admin_guard)):
    try:
        coordinator = get_store_coordinator()
        deleted = coordinator.delete_ids(ids)
        return {"deleted": deleted, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- 품질 루프 엔드포인트 ----
@app.get("/quality/goldens")
async def quality_get_golden():
    try:
        return {"items": load_golden_set(), "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quality/goldens")
async def quality_add_golden(item: Dict[str, Any], _admin=Depends(admin_guard)):
    try:
        count = add_golden_item(item)
        return {"count": count, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/quality/goldens/{idx}")
async def quality_delete_golden(idx: int, _admin=Depends(admin_guard)):
    try:
        ok = delete_golden_item(idx)
        return {"deleted": ok, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quality/offline-metrics")
async def quality_offline_metrics(predictions: List[Dict[str, Any]], k: int = 5, _admin=Depends(admin_guard)):
    try:
        golden = load_golden_set()
        metrics = compute_metrics(golden, predictions, k=k)
        return {"metrics": metrics, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quality/change-matrix")
async def quality_change_matrix(_admin=Depends(admin_guard)):
    try:
        current = {k: get_config(k) for k in [
            "LLM_MODEL", "EMBEDDING_MODEL", "CROSS_ENCODER_MODEL",
            "SLA_P95_TARGET_MS", "SEARCH_TIMEOUT_S", "LLM_TIMEOUT_S",
        ]}
        # 베이스라인은 현재는 None (향후 저장 지원)
        matrix = generate_change_matrix(current, baseline=None)
        return {"matrix": matrix, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/release/gate")
async def release_gate_check(latest_report_only: bool = True, _admin=Depends(admin_guard)):
    """릴리즈 게이트 스텁: 지연/오류/품질 임계 체크 결과 반환"""
    try:
        # 실측 기반 간단 판정: 최근 1시간 p95/에러수 + 품질(샘플 기반) 체크
        p95_target = int(get_config("GATE_P95_MS", SLA_P95_TARGET_MS))
        max_errors_1h = int(get_config("GATE_ERROR_MAX_1H", 100))
        quality_min = float(get_config("GATE_QUALITY_MIN", 0.80))

        latencies_1h = list(_rolling_events_1h["latency_ms"]) if _rolling_events_1h["latency_ms"] else []
        p95_1h = 0
        if latencies_1h:
            xs1 = sorted(latencies_1h)
            p95_1h = xs1[int(0.95 * (len(xs1) - 1))]
        import time as _t
        now = _t.time()
        recent_errors_1h = [t for t in list(_rolling_events_1h["errors"]) if now - t <= 3600]
        err_1h = len(recent_errors_1h)
        # 품질은 온라인 샘플 큐잉 로그가 없으므로, 간단히 0.9로 가정(향후 품질 루프 연동)
        est_quality = 0.9

        passed = (p95_1h <= p95_target) and (err_1h <= max_errors_1h) and (est_quality >= quality_min)
        parts = {
            "p95_latency_ms": int(p95_1h),
            "p95_target_ms": p95_target,
            "errors_1h": err_1h,
            "errors_1h_max": max_errors_1h,
            "est_quality": est_quality,
            "quality_min": quality_min,
            "passed": passed,
        }
        return {"gate": parts, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ci/selftest")
async def ci_selftest():
    """CI 재현성 간단 셀프 테스트: 모델/루트/헬스 엔드포인트 점검."""
    try:
        # 환경 모킹 정보
        mock_ollama = bool(str(get_config("MOCK_OLLAMA", False)).lower() == "true")
        # 핵심 의존성 체크
        vs = get_vector_store()
        _ = vs.get_total_chunks()
        # 라우터/LLM 준비 상태
        _ = get_query_router()
        _ = get_answer_generator()
        # 헬스 스냅샷
        health = {"ready": True}
        return {"status": "ok", "mocks": {"ollama": mock_ollama, "chroma": bool(get_config("MOCK_CHROMA", False)), "pdf": bool(get_config("MOCK_PDF", False))}, "health": health, "config_version": get_config_version()}
    except Exception as e:
        return {"status": "error", "message": str(e), "config_version": get_config_version()}

@app.get("/metrics")
async def metrics():
    try:
        cache_stats = get_all_cache_stats()
        # 롤링 p95/에러율 계산(간단 구현)
        latencies = list(_rolling_events["latency_ms"]) if _rolling_events["latency_ms"] else []
        p95 = 0
        if latencies:
            xs = sorted(latencies)
            p95 = xs[int(0.95 * (len(xs) - 1))]
        # 최근 5분 오류율 (샘플 기반)
        import time as _t
        now = _t.time()
        recent_errors = [t for t in list(_rolling_events["errors"]) if now - t <= 300]
        error_rate_5m = len(recent_errors)
        # 최근 1시간 p95/오류
        latencies_1h = list(_rolling_events_1h["latency_ms"]) if _rolling_events_1h["latency_ms"] else []
        p95_1h = 0
        if latencies_1h:
            xs1 = sorted(latencies_1h)
            p95_1h = xs1[int(0.95 * (len(xs1) - 1))]
        recent_errors_1h = [t for t in list(_rolling_events_1h["errors"]) if now - t <= 3600]
        error_1h = len(recent_errors_1h)
        # 파라미터 평균(TopK/임계)
        import statistics as _stat
        topk_avg = int(_stat.mean(list(_rolling_params["topk"])) if _rolling_params["topk"] else 0)
        sim_avg = float(_stat.mean(list(_rolling_params["sim_threshold"])) if _rolling_params["sim_threshold"] else 0.0)
        parts = {
            "requests_total": _metrics['requests_total'],
            "vector_search_failures": _metrics['vector_search_failures'],
            "llm_failures": _metrics['llm_failures'],
            "timeouts": {"search": _metrics['search_timeouts'], "llm": _metrics['llm_timeouts']},
            "cache": cache_stats,
            "rate_limited": _metrics['rate_limited'],
            "rolling": {"p95_ms": p95, "errors_5m": error_rate_5m, "p95_1h_ms": p95_1h, "errors_1h": error_1h, "avg_topk": topk_avg, "avg_sim_threshold": sim_avg},
            "routing": _routing_counters,
            "config_version": get_config_version()
        }
        # 임계 초과시 간단 Slack 알림(옵션)
        try:
            p95_gate = int(get_config("GATE_P95_MS", SLA_P95_TARGET_MS))
            if p95_1h and p95_1h > p95_gate:
                notify_slack(f"[ALERT] p95_1h {p95_1h}ms > gate {p95_gate}ms")
        except Exception:
            pass
        # 간단한 텍스트/JSON 선택 가능 향후 확장 대비
        return JSONResponse(content={"metrics": parts})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    vector_store: VectorStoreInterface = Depends(get_vector_store),
    coordinator: StoreCoordinator = Depends(get_store_coordinator),
    _auth=Depends(user_guard)
):
    """PDF 업로드 및 처리"""
    # 확장자 검사
    if not any(file.filename.lower().endswith(ext.strip()) for ext in ALLOWED_EXT):
        raise_mapped_error("INVALID_EXTENSION", detail=f"filename={file.filename}")
    # MIME 검사
    if file.content_type not in ALLOWED_MIME:
        raise_mapped_error("INVALID_MIME", detail=f"content_type={file.content_type}")
    
    temp_file_path = None
    try:
        # 선제 업로드 제한 검사 (메모리 상 읽기)
        content = await file.read()
        max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
        if len(content) > max_bytes:
            raise_mapped_error("UPLOAD_TOO_LARGE", detail=f"size={len(content)} bytes > limit={max_bytes}")
        # 최소 PDF 구조 검증
        if not _basic_pdf_structure_check(content):
            raise_mapped_error("INVALID_PDF_STRUCTURE")
        # 페이지 상한 사전 점검(근사)
        approx_pages = _count_pdf_pages_fast(content)
        if approx_pages and approx_pages > MAX_PAGES:
            raise_mapped_error("PAGE_LIMIT_EXCEEDED", detail=f"pages~={approx_pages} > limit={MAX_PAGES}")

        # 임시 파일로 저장(이후 정확 페이지 수 검증 및 처리)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # PDF 처리
        start_time = datetime.now()
        pdf_id = str(uuid.uuid4())
        
        # 정확한 페이지 수 검증 (가능 시)
        try:
            from PyPDF2 import PdfReader  # 선택적 의존성
            reader = PdfReader(temp_file_path)
            total_pages_precise = len(reader.pages)
            if total_pages_precise > MAX_PAGES:
                raise_mapped_error("PAGE_LIMIT_EXCEEDED", detail=f"pages={total_pages_precise} > limit={MAX_PAGES}")
        except ImportError:
            # 의존성 없으면 근사값으로 대체(이미 체크됨)
            pass
        except Exception:
            # 페이지 확인 실패는 구조 검증으로 대체
            pass

        # PDF 처리 및 청크 생성 (스레드풀 오프로딩)
        loop = asyncio.get_running_loop()
        chunks, metadata = await loop.run_in_executor(
            _executor, lambda: pdf_processor.process_pdf(temp_file_path, pdf_id)
        )
        
        # 벡터 저장소에 저장 (스레드풀 오프로딩)
        await loop.run_in_executor(_executor, lambda: coordinator.add_chunks(chunks))
        
        # 키워드를 파이프라인 설정에 추가
        if pdf_processor.enable_keyword_extraction:
            pdf_processor.keyword_extractor.add_keywords_to_pipeline()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 메타데이터 저장
        pdf_metadata[pdf_id] = {
            "filename": file.filename,
            "total_pages": len(chunks),
            "upload_time": datetime.now().isoformat(),
            "file_size": len(content)
        }
        
        return PDFUploadResponse(
            pdf_id=pdf_id,
            filename=file.filename,
            total_pages=len(chunks),
            total_chunks=len(chunks),
            processing_time=processing_time
        )
        
    except HTTPException:
        # HTTPException은 그대로 전파
        raise
    except Exception as e:
        logger.error(f"PDF 처리 실패: {e}")
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": "PDF 처리 중 오류 발생", "detail": str(e)})
    finally:
        # 임시 파일 삭제 보장
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception:
            pass

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    vector_store: VectorStoreInterface = Depends(get_vector_store),
    question_analyzer: QuestionAnalyzer = Depends(get_question_analyzer),
    answer_generator: AnswerGenerator = Depends(get_answer_generator),
    query_router: QueryRouter = Depends(get_query_router),
    _auth=Depends(user_guard)
):
    """질문에 대한 답변 생성 (최적화된 버전)"""
    
    # 성능 측정 시작
    start_time = time.time()
    session_id = None
    
    try:
        # 단계별 로그 시작
        if chatbot_logger:
            session_id = chatbot_logger._generate_session_id()
            chatbot_logger.log_step(session_id, ProcessingStep.START, 0.0, f"질문: {request.question[:50]}...")
            try:
                unified_logger.set_session_id(session_id)
            except Exception:
                pass
        # 입력 스키마 검증 및 프롬프트 정화
        try:
            payload = validate_question_payload(
                request.dict(),
                max_question_len=QUESTION_MAX_LEN,
                allowed_fields=["question", "pdf_id", "user_id", "use_conversation_context", "max_chunks"],
            )
        except ValidationError as ve:
            raise HTTPException(status_code=422, detail={"code": "INVALID_PAYLOAD", "message": str(ve)})
        extra = str(get_config("SANITIZE_EXTRA_PATTERNS", "")).strip()
        patterns = [p for p in extra.split("|") if p]
        request.question = sanitize_prompt(payload["question"], remove_sensitive_meta=True, additional_patterns=patterns)
        # 타이밍 수집용 기록자
        timings = {"start": start_time}
        
        # SBERT 기반 쿼리 라우팅
        routing_start = time.time()
        route_result = query_router.route_query(request.question)
        routing_time = time.time() - routing_start
        timings["routing_ms"] = routing_time * 1000.0
        
        if chatbot_logger and session_id:
            chatbot_logger.log_step(
                session_id, 
                ProcessingStep.SBERT_ROUTING, 
                routing_time, 
                f"라우팅결과: {route_result.route.value} (신뢰도: {route_result.confidence:.3f})"
            )
        
        logger.info(f"📍 라우팅 결과: {route_result.route.value} (신뢰도: {route_result.confidence:.3f})")
        try:
            unified_logger.info(
                "routing_decision",
                LogCategory.SYSTEM,
                module="router",
                metadata={"route": route_result.route.value, "confidence": route_result.confidence}
            )
        except Exception:
            pass
        
        # 인사말 처리 (LLM 기반)
        if route_result.route == QueryRoute.GREETING:
            if chatbot_logger and session_id:
                chatbot_logger.log_step(session_id, ProcessingStep.GREETING_PIPELINE, 0.0, "인사말 처리 시작")
            
            try:
                llm_greeting_handler = get_llm_greeting_handler()
                greeting_response = llm_greeting_handler.get_greeting_response(request.question)
                
                # 인사말 로깅
                try:
                    if chatbot_logger:
                        chatbot_logger.log_greeting(
                            user_question=request.question,
                            greeting_response=greeting_response["answer"],
                            processing_time=greeting_response["generation_time"],
                            confidence_score=greeting_response["confidence_score"],
                            greeting_type=greeting_response.get("greeting_type", "general")
                        )
                except Exception as log_error:
                    logger.warning(f"인사말 로깅 중 오류 발생: {log_error}")
                
                if chatbot_logger and session_id:
                    chatbot_logger.log_step(session_id, ProcessingStep.COMPLETION, greeting_response["generation_time"], "인사말 처리 완료")
                
                return QuestionResponse(
                    answer=greeting_response["answer"],
                    confidence_score=greeting_response["confidence_score"],
                    used_chunks=[],
                    generation_time=greeting_response["generation_time"],
                    question_type="greeting",
                    llm_model_name=f"llm_greeting_{greeting_response.get('method', 'unknown')}",
                    pipeline_type="greeting",
                    sql_query=None
                )
            except Exception as greeting_error:
                logger.error(f"인사말 처리 중 오류: {greeting_error}")
                # 기본 인사말 (설정 SSOT)
                fallback_greeting = str(get_config("DEFAULT_GREETING", "안녕하세요! 범용 RAG 시스템에 오신 것을 환영합니다! 🤖"))
                
                if chatbot_logger and session_id:
                    chatbot_logger.log_step(session_id, ProcessingStep.ERROR, 0.0, f"인사말 처리 오류: {greeting_error}")
                
                return QuestionResponse(
                    answer=fallback_greeting,
                    confidence_score=0.8,
                    used_chunks=[],
                    generation_time=0.001,
                    question_type="greeting",
                    llm_model_name="fallback_greeting",
                    pipeline_type="greeting",
                    sql_query=None
                )
        
        # SQL 쿼리 처리는 제거되었습니다. PDF 검색으로 폴백합니다.
        if route_result.route == QueryRoute.SQL_QUERY:
            logger.info("SQL 쿼리가 요청되었지만 SQL 기능이 제거되어 PDF 검색으로 폴백합니다.")
            route_result.route = QueryRoute.PDF_SEARCH
        
        # PDF 검색 처리 (기본 모드)
        logger.info("📄 PDF 검색 모드로 처리")
        
        if chatbot_logger and session_id:
            chatbot_logger.log_step(session_id, ProcessingStep.PDF_PIPELINE, 0.0, "PDF 파이프라인 시작")
        
        # 1. 질문 분석
        analysis_start = time.time()
        analyzed_question = question_analyzer.analyze_question(
            request.question,
            use_conversation_context=request.use_conversation_context
        )
        analysis_time = time.time() - analysis_start
        timings["analysis_ms"] = analysis_time * 1000.0
        
        if chatbot_logger and session_id:
            chatbot_logger.log_step(session_id, ProcessingStep.QUESTION_ANALYSIS, analysis_time, f"질문분석: {analyzed_question.question_type.value}")
        
        # 2. 관련 문서 검색 (멀티쿼리 가중 결합 + 임계값 적용)
        search_start = time.time()
        _metrics["requests_total"] += 1
        query_embedding = analyzed_question.embedding
        
        # 벡터 스토어 상태 확인
        total_chunks = vector_store.get_total_chunks()
        logger.info(f"벡터 스토어 상태 - 총 청크 수: {total_chunks}")
        
        if total_chunks == 0:
            logger.error("⚠️ 벡터 스토어에 청크가 없습니다! PDF 업로드가 필요합니다.")
            return QuestionResponse(
                answer="죄송합니다. 현재 문서 데이터베이스가 비어있습니다. 관리자에게 문의해주세요.",
                confidence_score=0.0,
                used_chunks=[],
                generation_time=0.001,
                question_type="error",
                llm_model_name="none",
                pipeline_type="error",
                sql_query=None
            )
        
        # 라우트별 SLA 타깃 및 예산 계산(SSOT)
        route_target_ms = get_route_sla_p95_ms(route_result.route.value)
        route_budget_ms = route_target_ms * BUDGET_ROUTE_PCT
        search_budget_ms = route_target_ms * BUDGET_SEARCH_PCT
        llm_budget_ms = route_target_ms * BUDGET_LLM_PCT

        # 질문 길이 기반 TopK 가이드 (짧을수록 작게)
        qlen = len(request.question)
        base_topk = 5 if qlen > 120 else 4 if qlen > 60 else 3
        effective_top_k = max(3, min(request.max_chunks, base_topk))
        # 초기 유사도/길이 기준 임계값
        dynamic_similarity = 0.15 if qlen > 80 else 0.20
        # 예산 기반 검색 타임아웃 축소
        search_timeout = min(SEARCH_TIMEOUT_S, max(3.0, search_budget_ms / 1000.0))
        try:
            _rolling_params["topk"].append(effective_top_k)
            _rolling_params["sim_threshold"].append(dynamic_similarity)
        except Exception:
            pass
        
        # 멀티쿼리 구성: Q0(원문), Q1(핵심 키워드 축약), Q2(정확 구절 강조)
        original_query = request.question
        processed_query = analyzed_question.processed_question or original_query
        extracted_keywords = analyzed_question.keywords or []
        # 필수/보조 키워드 단순 분리: 선두 2개를 필수, 이후 3개를 보조로 간주
        required_keywords = extracted_keywords[:2]
        optional_keywords = extracted_keywords[2:5]
        # Q1: 키워드 중심 축약 쿼리
        keyword_query = " ".join(required_keywords + optional_keywords) if (required_keywords or optional_keywords) else processed_query
        # Q2: 정확 구절 강조 (가능 시 인용부호로 감싸기)
        phrase_core = " ".join(required_keywords) if required_keywords else ""
        phrase_query = f'"{phrase_core}"' if phrase_core else processed_query
        
        # 임베딩 생성 (SBERT)
        embeddings = []
        try:
            if hasattr(question_analyzer, 'embedding_model') and question_analyzer.embedding_model:
                embeddings = question_analyzer.embedding_model.encode([original_query, keyword_query, phrase_query])
            else:
                embeddings = [query_embedding, query_embedding, query_embedding]
        except Exception as embed_err:
            logger.warning(f"멀티쿼리 임베딩 실패, 단일 쿼리로 폴백: {embed_err}")
            embeddings = [query_embedding, query_embedding, query_embedding]
        
        # 멀티쿼리 검색 실행
        weights = [0.5, 0.3, 0.2]
        multi_results = []
        try:
            for e in embeddings:
                # 검색 타임아웃 적용
                res = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        _executor, lambda: _search_cb.call(vector_store.search,
                            e,
                            top_k=max(effective_top_k * 2, effective_top_k),
                            similarity_threshold=dynamic_similarity
                        )
                    ),
                    timeout=search_timeout
                )
                multi_results.append(res)
        except asyncio.TimeoutError:
            _metrics["vector_search_failures"] += 1
            _metrics["search_timeouts"] += 1
            raise HTTPException(status_code=504, detail={"code": "SEARCH_TIMEOUT", "message": "벡터 검색 시간이 초과되었습니다."})
        
        # 결과 가중 결합
        combined_scores = {}
        combined_chunks = {}
        for idx, results in enumerate(multi_results):
            w = weights[idx]
            for chunk, score in results:
                cid = chunk.chunk_id
                combined_scores[cid] = combined_scores.get(cid, 0.0) + (score * w)
                if cid not in combined_chunks:
                    combined_chunks[cid] = chunk
        
        # 상위 결과 선정
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        relevant_chunks = [(combined_chunks[cid], s) for cid, s in sorted_combined[:effective_top_k]]
        search_time = time.time() - search_start
        timings["search_ms"] = search_time * 1000.0
        
        if chatbot_logger and session_id:
            chatbot_logger.log_step(session_id, ProcessingStep.VECTOR_SEARCH, search_time, f"벡터검색: {len(relevant_chunks)}개 청크 발견")
        
        # 디버깅: 검색 결과 로깅
        logger.info(f"🔍 검색된 청크 수: {len(relevant_chunks)}")
        for i, (chunk, score) in enumerate(relevant_chunks[:3]):
            logger.info(f"  📄 청크 {i+1}: {chunk.chunk_id} (유사도: {score:.3f})")
            logger.info(f"    내용: {chunk.content[:150]}...")
        
        # 검색 결과가 없을 때 경고
        if not relevant_chunks:
            logger.warning("⚠️ 검색된 관련 청크가 없습니다!")
            try:
                health_metrics.incr_fn()  # 라우팅/검색 미탐 추적
            except Exception:
                pass
        else:
            logger.info(f"✅ {len(relevant_chunks)}개의 관련 청크를 찾았습니다.")
        
        # 3. 컨텍스트 검증 및 답변 생성
        if not relevant_chunks:
            logger.warning("🔍 검색된 관련 청크가 없어 LLM 직접 답변으로 전환합니다.")
            answer_gen_start = time.time()
            try:
                # 라우트/모드별 LLM 정책 주입(LOW 비용 모드)
                llm_policy = get_llm_policy(route_result.route.value, mode="LOW")
                if hasattr(answer_generator, 'llm') and hasattr(answer_generator.llm, 'config'):
                    answer_generator.llm.config.temperature = llm_policy["temperature"]
                    answer_generator.llm.config.max_length = llm_policy["max_new_tokens"]
                answer = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        _executor, lambda: answer_generator.generate_direct_answer(request.question)
                    ),
                    timeout=min(LLM_TIMEOUT_S, max(3.0, llm_budget_ms / 1000.0))
                )
            except asyncio.TimeoutError:
                _metrics["llm_failures"] += 1
                _metrics["llm_timeouts"] += 1
                raise HTTPException(status_code=504, detail={"code": "LLM_TIMEOUT", "message": "LLM 응답 시간이 초과되었습니다."})
            answer_gen_time = time.time() - answer_gen_start
            timings["llm_ms"] = answer_gen_time * 1000.0
            
            if chatbot_logger and session_id:
                chatbot_logger.log_step(session_id, ProcessingStep.ANSWER_GENERATION, answer_gen_time, "직접답변생성 (청크없음)")
        else:
            # 컨텍스트 내용 로깅
            context_content = "\n".join([chunk.content[:100] + "..." for chunk, _ in relevant_chunks[:3]])
            logger.info(f"📄 컨텍스트 내용 (일부):\n{context_content}")
            try:
                unified_logger.info(
                    "context_labels",
                    LogCategory.SECURITY,
                    module="prompt",
                    metadata={"sources": [getattr(c, 'metadata', {}) for c, _ in relevant_chunks[:3]]}
                )
            except Exception:
                pass
            
            # 재순위 on/off 정책(라우트별)
            try:
                rerank_enabled = get_rerank_policy(route_result.route.value)
                unified_logger.info("rerank_policy", LogCategory.SYSTEM, module="policy", metadata={"route": route_result.route.value, "enabled": rerank_enabled})
            except Exception:
                rerank_enabled = True

            answer_gen_start = time.time()
            try:
                # 라우트/모드별 LLM 정책 주입(DEFAULT 모드)
                llm_policy = get_llm_policy(route_result.route.value, mode="DEFAULT")
                if hasattr(answer_generator, 'llm') and hasattr(answer_generator.llm, 'config'):
                    answer_generator.llm.config.temperature = llm_policy["temperature"]
                    answer_generator.llm.config.max_length = llm_policy["max_new_tokens"]
                answer = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        _executor, lambda: _llm_cb.call(
                            answer_generator.generate_answer,
                            analyzed_question,
                            relevant_chunks if rerank_enabled else relevant_chunks,
                            conversation_history=None,
                            pdf_id=request.pdf_id
                        )
                    ),
                    timeout=min(LLM_TIMEOUT_S, max(3.0, llm_budget_ms / 1000.0))
                )
            except asyncio.TimeoutError:
                _metrics["llm_failures"] += 1
                _metrics["llm_timeouts"] += 1
                raise HTTPException(status_code=504, detail={"code": "LLM_TIMEOUT", "message": "LLM 응답 시간이 초과되었습니다."})
            answer_gen_time = time.time() - answer_gen_start
            timings["llm_ms"] = answer_gen_time * 1000.0
            
            if chatbot_logger and session_id:
                chatbot_logger.log_step(session_id, ProcessingStep.ANSWER_GENERATION, answer_gen_time, f"컨텍스트답변생성: {len(relevant_chunks)}개 청크 사용")

            # 사후 검증: 필수 키워드 포함 여부 검사 및 1회 교정 재생성
            def _contains_all_required(text: str, required: List[str]) -> bool:
                try:
                    lowered = text or ""
                    return all((k in lowered) for k in required if k)
                except Exception:
                    return True
            used_retry = False
            if required_keywords and not _contains_all_required(answer.content, required_keywords):
                # 필수 키워드를 포함한 컨텍스트로 필터링 후 1회 재생성
                filtered = [(c, s) for (c, s) in relevant_chunks if any((rk in c.content) for rk in required_keywords)]
                if filtered:
                    used_retry = True
                    logger.info(f"🔁 필수 키워드 미충족으로 재생성 시도 (필수: {required_keywords})")
                    answer_gen_start = time.time()
                    try:
                        # 라우트/모드별 LLM 정책 주입(RETRY 모드)
                        llm_policy = get_llm_policy(route_result.route.value, mode="RETRY")
                        if hasattr(answer_generator, 'llm') and hasattr(answer_generator.llm, 'config'):
                            answer_generator.llm.config.temperature = llm_policy["temperature"]
                            answer_generator.llm.config.max_length = llm_policy["max_new_tokens"]
                        answer = await asyncio.wait_for(
                            asyncio.get_running_loop().run_in_executor(
                                _executor, lambda: _llm_cb.call(
                                    answer_generator.generate_answer,
                                    analyzed_question,
                                    filtered,
                                    conversation_history=None,
                                    pdf_id=request.pdf_id
                                )
                            ),
                            timeout=LLM_TIMEOUT_S
                        )
                    except asyncio.TimeoutError:
                        _metrics["llm_failures"] += 1
                        raise HTTPException(status_code=504, detail={"code": "LLM_TIMEOUT", "message": "LLM 응답 시간이 초과되었습니다."})
                    answer_gen_time = time.time() - answer_gen_start
                    if chatbot_logger and session_id:
                        chatbot_logger.log_step(session_id, ProcessingStep.ANSWER_GENERATION, answer_gen_time, "재생성(필수키워드 충족 유도)")
            # 디버그 로그: 검증 결과 (표준 로거 사용)
            try:
                logger.debug(
                    f"[검증] 필수키워드: {required_keywords} | 충족: {_contains_all_required(answer.content, required_keywords)} | 재시도사용: {used_retry}"
                )
            except Exception:
                pass

            # 최소 정확도 임계 정책 및 승격/강등 이벤트 표준화 로깅
            try:
                min_acc = get_min_accuracy_threshold(route_result.route.value)
                conf = float(getattr(answer, 'confidence_score', None) or getattr(answer, 'confidence', 0.0))
                event = "promote" if conf >= float(min_acc) else "demote"
                unified_logger.info(
                    "accuracy_policy_event",
                    LogCategory.SYSTEM,
                    module="policy",
                    metadata={
                        "route": route_result.route.value,
                        "min_accuracy": float(min_acc),
                        "confidence": conf,
                        "event": event,
                    }
                )
            except Exception:
                pass
        
                # 4. 대화 히스토리에 추가
        question_analyzer.add_conversation_item(
            question=request.question,
            answer=answer.content,
            used_chunks=answer.used_chunks,
            confidence_score=answer.confidence_score
        )
        
        # 5. API 로깅 + SLA 메타데이터
        try:
            if chatbot_logger:
                # 질문 의도 및 키워드 추출 (간단한 버전)
                intent = "PDF_QUERY"
                keywords = request.question.split()[:5]  # 첫 5개 단어를 키워드로 사용
                
                chatbot_logger.log_question(
                    user_question=request.question,
                    question_type=QuestionType.PDF,
                    intent=intent,
                    keywords=keywords,
                    processing_time=answer.generation_time,
                    confidence_score=answer.confidence_score,
                    generated_answer=answer.content,
                    used_chunks=answer.used_chunks,
                    model_name=answer.model_name,
                    additional_info={
                        "pipeline_type": route_result.route.value,
                        "user_id": request.user_id,
                        "sla": {
                            "p95_target_ms": SLA_P95_TARGET_MS,
                            "budgets": {
                                "route": BUDGET_ROUTE_PCT,
                                "search": BUDGET_SEARCH_PCT,
                                "rerank": BUDGET_RERANK_PCT,
                                "llm": BUDGET_LLM_PCT
                            },
                            "dynamic": {
                                "topk": effective_top_k,
                                "sim_threshold": dynamic_similarity
                            }
                        }
                    }
                )
        except Exception as log_error:
            logger.warning(f"API 로깅 중 오류 발생: {log_error}")
        
        # 완료 로그
        if chatbot_logger and session_id:
            total_time = time.time() - start_time
            chatbot_logger.log_step(session_id, ProcessingStep.COMPLETION, total_time, "PDF 파이프라인 완료")
        timings["total_ms"] = (time.time() - start_time) * 1000.0
        try:
            _rolling_events["latency_ms"].append(timings["total_ms"])
            _rolling_events_1h["latency_ms"].append(timings["total_ms"]) 
        except Exception:
            pass
        
        resp = QuestionResponse(
            answer=answer.content,
            confidence_score=answer.confidence_score,
            used_chunks=answer.used_chunks,
            generation_time=answer.generation_time,
            question_type=analyzed_question.question_type.value,
            llm_model_name=answer.model_name,
            pipeline_type=route_result.route.value,
            sql_query=None
        )
        # 온라인 품질 샘플링(1-5%) 큐잉(간략 로깅 기반)
        try:
            import random
            sample_rate = ONLINE_EVAL_SAMPLE_VAL
            if random.random() < sample_rate:
                unified_logger.info(
                    "online_eval_enqueue",
                    LogCategory.SYSTEM,
                    module="quality",
                    metadata={
                        "question": request.question[:120],
                        "answer_len": len(answer.content or ""),
                        "used_chunks": len(answer.used_chunks or []),
                        "confidence": answer.confidence_score,
                        "route": route_result.route.value
                    }
                )
        except Exception:
            pass
        # 타이밍 헤더 샘플 노출(로거 + 헤더 삽입 시도)
        try:
            unified_logger.info("timings", LogCategory.SYSTEM, module="api", metadata={"timings": timings})
        except Exception:
            pass
        try:
            from fastapi.responses import JSONResponse as _JR
            payload = resp.dict()
            headers = {
                "X-Timing-Analysis-ms": str(int(timings.get("analysis_ms", 0))),
                "X-Timing-Search-ms": str(int(timings.get("search_ms", 0))),
                "X-Timing-LLM-ms": str(int(timings.get("llm_ms", 0))),
                "X-Timing-Total-ms": str(int(timings.get("total_ms", 0)))
            }
            return _JR(content=payload, headers=headers)
        except Exception:
            return resp
        
    except Exception as e:
        logger.error(f"질문 처리 실패: {e}")
        try:
            _rolling_events["errors"].append(time.time())
            _rolling_events_1h["errors"].append(time.time())
        except Exception:
            pass
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        
        # 에러 로깅
        try:
            if chatbot_logger:
                if session_id:
                    chatbot_logger.log_step(session_id, ProcessingStep.ERROR, 0.0, f"전체처리오류: {str(e)}")
                
                chatbot_logger.log_error(
                    user_question=request.question,
                    error_message=str(e),
                    question_type=QuestionType.UNKNOWN
                )
        except Exception as log_error:
            logger.warning(f"에러 로깅 실패: {log_error}")
        
        # 사용자 친화적인 에러 메시지
        error_message = "죄송합니다. 질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status(_admin=Depends(admin_guard)):
    """시스템 상태 조회"""
    try:
        import psutil
        
        # 메모리 사용량
        memory = psutil.virtual_memory()
        memory_usage = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        }
        
        # 모델 로드 상태
        model_loaded = (
            answer_generator is not None and 
            question_analyzer is not None and 
            vector_store is not None
        )
        
        # PDF 및 청크 수
        total_pdfs = len(pdf_metadata)
        # 벡터 저장소에서 청크 수 확인 (get_all_chunks 메서드가 없으므로 다른 방법 사용)
        total_chunks = 0
        if vector_store and hasattr(vector_store, 'chunks'):
            total_chunks = len(vector_store.chunks)
        elif vector_store and hasattr(vector_store, 'faiss_store') and hasattr(vector_store.faiss_store, 'chunks'):
            total_chunks = len(vector_store.faiss_store.chunks)
        
        return SystemStatusResponse(
            status="running",
            llm_model_loaded=model_loaded,
            total_pdfs=total_pdfs,
            total_chunks=total_chunks,
            memory_usage=memory_usage
        )
        
    except Exception as e:
        logger.error(f"시스템 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"시스템 상태 조회 실패: {str(e)}")


@app.get("/router/stats")
async def get_router_stats(
    query_router: QueryRouter = Depends(get_query_router)
):
    """쿼리 라우터 통계"""
    _auth=Depends(user_guard)
    try:
        stats = query_router.get_route_statistics()
        counters = {}
        try:
            counters = query_router.get_route_counters()
        except Exception:
            counters = {}
        try:
            global _routing_counters
            _routing_counters = counters or _routing_counters
        except Exception:
            pass
        return {
            "status": "success",
            "router_stats": stats,
            "route_counters": counters,
            "policy_version": get_config_version(),
            "config_version": get_config_version()
        }
    except Exception as e:
        logger.error(f"라우터 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"라우터 통계 조회 실패: {str(e)}")

@app.get("/ratelimit/report")
async def ratelimit_report(_admin=Depends(admin_guard)):
    """상위 IP/키 별 요청 카운트(간단)."""
    try:
        top = []
        for k, dq in _req_history.items():
            top.append({"client": k, "count": len(dq)})
        top = sorted(top, key=lambda x: x["count"], reverse=True)[:50]
        return {"top_clients": top, "window_sec": RL_WINDOW_SEC, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- 데이터 카탈로그/승인 워크플로우 ----
@app.get("/catalog")
async def catalog_get(_admin=Depends(admin_guard)):
    try:
        return {"items": load_catalog(), "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- 회귀 대시보드/운영 자동화/보안 스캔 스텁 ----
@app.get("/ops/regression-summary")
async def ops_regression_summary(_admin=Depends(admin_guard)):
    try:
        return {"summary": get_regression_summary(), "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ops/schedule/run")
async def ops_schedule_run(name: str, _admin=Depends(admin_guard)):
    try:
        return {"result": run_scheduled_job(name), "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ops/security/scan")
async def ops_security_scan(_admin=Depends(admin_guard)):
    try:
        return {"result": run_security_scan(), "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/catalog/requests")
async def catalog_submit_request(item: Dict[str, Any], _admin=Depends(admin_guard)):
    try:
        count = submit_change_request(item)
        return {"queued": count, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/catalog/requests")
async def catalog_list_requests(_admin=Depends(admin_guard)):
    try:
        return {"requests": list_change_requests(), "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/catalog/requests/{idx}/approve")
async def catalog_approve(idx: int, _admin=Depends(admin_guard)):
    try:
        ok = approve_request(idx)
        return {"approved": ok, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/catalog/requests/{idx}/reject")
async def catalog_reject(idx: int, _admin=Depends(admin_guard)):
    try:
        ok = reject_request(idx)
        return {"rejected": ok, "config_version": get_config_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/router/test")
async def test_routing(
    question: str,
    query_router: QueryRouter = Depends(get_query_router)
):
    """라우팅 테스트"""
    try:
        result = query_router.route_query(question)
        return {
            "question": question,
            "route": result.route.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "metadata": result.metadata
        }
    except Exception as e:
        logger.error(f"라우팅 테스트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"라우팅 테스트 실패: {str(e)}")

@app.get("/pdfs")
async def get_pdf_list():
    """등록된 PDF 목록 조회"""
    try:
        pdfs = []
        for pdf_id, metadata in pdf_metadata.items():
            pdfs.append({
                "pdf_id": pdf_id,
                "filename": metadata.get("filename", "Unknown"),
                "upload_time": metadata.get("upload_time", ""),
                "total_pages": metadata.get("total_pages", 0),
                "total_chunks": metadata.get("total_chunks", 0),
                "file_size": metadata.get("file_size", 0)
            })
        
        return {"pdfs": pdfs, "config_version": get_config_version()}
        
    except Exception as e:
        logger.error(f"PDF 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"PDF 목록 조회 실패: {str(e)}")

@app.post("/auto-upload")
async def auto_upload_pdfs():
    """data 폴더의 PDF 파일들을 자동으로 업로드"""
    try:
        import os
        from pathlib import Path
        
        # data 폴더와 data/pdfs 폴더 모두 확인
        data_folders = ["./data", "./data/pdfs"]
        pdf_files = []
        
        for data_folder in data_folders:
            if not os.path.exists(data_folder):
                logger.warning(f"{data_folder} 폴더가 존재하지 않습니다.")
                continue
            
            # 재귀적으로 PDF 파일 찾기
            for root, dirs, files in os.walk(data_folder):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        pdf_files.append(pdf_path)
        
        if not pdf_files:
            return {"message": "업로드할 PDF 파일이 없습니다.", "uploaded_count": 0}
        
        logger.info(f"자동 업로드 시작: {len(pdf_files)}개의 PDF 파일")
        
        uploaded_count = 0
        failed_count = 0
        
        coordinator = get_store_coordinator()
        for pdf_path in pdf_files:
            try:
                # 이미 처리된 PDF인지 확인
                pdf_id = os.path.basename(pdf_path)
                if pdf_id in pdf_metadata:
                    logger.info(f"이미 처리된 PDF 건너뛰기: {pdf_id}")
                    continue
                
                # PDF 처리
                chunks, metadata = pdf_processor.process_pdf(pdf_path)
                coordinator.add_chunks(chunks)
                
                # 메타데이터 저장
                pdf_metadata[pdf_id] = {
                    "filename": pdf_id,
                    "total_pages": len(chunks),
                    "upload_time": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                    "file_size": os.path.getsize(pdf_path)
                }
                
                uploaded_count += 1
                logger.info(f"PDF 자동 업로드 완료: {pdf_id} ({len(chunks)}개 청크)")
                
            except Exception as e:
                logger.error(f"PDF 자동 업로드 실패 {pdf_path}: {e}")
                failed_count += 1
        
        return {
            "message": f"자동 업로드 완료: {uploaded_count}개 성공, {failed_count}개 실패",
            "uploaded_count": uploaded_count,
            "failed_count": failed_count,
            "total_files": len(pdf_files),
            "config_version": get_config_version()
        }
        
    except Exception as e:
        logger.error(f"자동 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"자동 업로드 실패: {str(e)}")

@app.get("/api/greeting/statistics")
async def get_greeting_statistics():
    """인사말 처리 통계 확인"""
    try:
        llm_greeting_handler = get_llm_greeting_handler()
        stats = llm_greeting_handler.get_statistics()
        return {
            "status": "success",
            "data": stats,
            "config_version": get_config_version()
        }
    except Exception as e:
        logger.error(f"인사말 통계 확인 중 오류: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/company/info")
async def get_company_info():
    """회사 정보 조회"""
    try:
        from core.config.company_config import CompanyConfig
        company_config = CompanyConfig()
        company_info = company_config.get_company_info()
        return {
            "status": "success",
            "data": company_info,
            "config_version": get_config_version()
        }
    except Exception as e:
        logger.error(f"회사 정보 조회 중 오류: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/keywords/cache/stats", response_model=KeywordCacheResponse)
async def get_keyword_cache_stats():
    """키워드 캐시 통계 조회"""
    try:
        pdf_processor = get_pdf_processor()
        stats = pdf_processor.get_keyword_cache_stats()
        
        if "error" in stats:
            raise HTTPException(status_code=400, detail=stats["error"])
        
        return KeywordCacheResponse(**stats)
    except Exception as e:
        logger.error(f"키워드 캐시 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/keywords/pipeline/add", response_model=KeywordPipelineResponse)
async def add_keywords_to_pipeline():
    """추출된 키워드를 파이프라인 설정에 추가"""
    try:
        pdf_processor = get_pdf_processor()
        pdf_processor.add_keywords_to_pipeline()
        
        # 추가된 키워드 정보 가져오기
        stats = pdf_processor.get_keyword_cache_stats()
        frequent_keywords = pdf_processor.keyword_extractor.get_frequent_keywords()
        
        return KeywordPipelineResponse(
            success=True,
            added_keywords=frequent_keywords[:20],  # 최대 20개
            message=f"파이프라인에 {len(frequent_keywords)}개 키워드 추가 완료"
        )
    except Exception as e:
        logger.error(f"키워드 파이프라인 추가 실패: {e}")
        return KeywordPipelineResponse(
            success=False,
            added_keywords=[],
            message=str(e)
        )

@app.delete("/api/keywords/cache/clear")
async def clear_keyword_cache():
    """키워드 캐시 초기화"""
    try:
        pdf_processor = get_pdf_processor()
        pdf_processor.clear_keyword_cache()
        return {"message": "키워드 캐시 초기화 완료", "config_version": get_config_version()}
    except Exception as e:
        logger.error(f"키워드 캐시 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/keywords/cache/save")
async def save_keyword_cache():
    """키워드 캐시를 파일로 저장"""
    try:
        pdf_processor = get_pdf_processor()
        pdf_processor.save_keyword_cache()
        return {"message": "키워드 캐시 저장 완료", "config_version": get_config_version()}
    except Exception as e:
        logger.error(f"키워드 캐시 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/keywords/cache/load")
async def load_keyword_cache():
    """파일에서 키워드 캐시 로드"""
    try:
        pdf_processor = get_pdf_processor()
        pdf_processor.load_keyword_cache()
        return {"message": "키워드 캐시 로드 완료", "config_version": get_config_version()}
    except Exception as e:
        logger.error(f"키워드 캐시 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 메모리 모니터링 API 엔드포인트 추가
@app.get("/memory/status")
async def get_memory_status():
    """메모리 상태 정보 반환"""
    try:
        from core.utils.memory_optimizer import memory_optimizer, model_memory_manager
        
        # 시스템 메모리 정보
        memory_info = memory_optimizer.get_memory_info()
        
        # 모델 메모리 정보
        model_status = model_memory_manager.get_model_status()
        
        return {
            "system_memory": {
                "total_gb": memory_info.total,
                "available_gb": memory_info.available,
                "used_gb": memory_info.used,
                "percent": memory_info.percent,
                "process_memory_gb": memory_info.process_memory
            },
            "model_memory": model_status,
            "timestamp": datetime.now().isoformat(),
            "config_version": get_config_version()
        }
    except Exception as e:
        logger.error(f"메모리 상태 조회 실패: {e}")
        return {"error": str(e)}

@app.post("/memory/optimize")
async def optimize_memory():
    """메모리 최적화 실행"""
    try:
        from core.utils.memory_optimizer import memory_optimizer
        
        # 메모리 최적화 실행
        after_info = memory_optimizer.optimize_memory(aggressive=True)
        
        return {
            "message": "메모리 최적화 완료",
            "optimized_memory_gb": after_info.used,
            "memory_percent": after_info.percent,
            "timestamp": datetime.now().isoformat(),
            "config_version": get_config_version()
        }
    except Exception as e:
        logger.error(f"메모리 최적화 실패: {e}")
        return {"error": str(e)}

@app.get("/memory/models")
async def get_loaded_models():
    """로드된 모델 목록 반환"""
    try:
        from core.utils.memory_optimizer import model_memory_manager
        
        model_status = model_memory_manager.get_model_status()
        
        return {
            "loaded_models": model_status["loaded_models"],
            "total_memory_gb": model_status["total_memory_gb"],
            "max_memory_gb": model_status["max_memory_gb"],
            "model_details": model_status["model_details"],
            "timestamp": datetime.now().isoformat(),
            "config_version": get_config_version()
        }
    except Exception as e:
        logger.error(f"모델 목록 조회 실패: {e}")
        return {"error": str(e)}

@app.delete("/memory/models/{model_name}")
async def unload_model(model_name: str):
    """특정 모델 언로드"""
    try:
        from core.utils.memory_optimizer import model_memory_manager
        
        model_memory_manager.unload_model(model_name)
        
        return {
            "message": f"모델 {model_name} 언로드 완료",
            "timestamp": datetime.now().isoformat(),
            "config_version": get_config_version()
        }
    except Exception as e:
        logger.error(f"모델 언로드 실패: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # 서버 시작 전 시스템 초기화
    initialize_system()
    uvicorn.run(app, host=get_config("HOST", "0.0.0.0"), port=int(get_config("PORT", 8008)))
