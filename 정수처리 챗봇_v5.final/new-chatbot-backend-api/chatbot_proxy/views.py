"""
챗봇 프록시 API 뷰
프론트엔드의 챗봇 요청을 챗봇 서버로 전달하는 프록시 역할을 수행합니다.
"""

import httpx
import logging
import asyncio
import uuid
import time
from typing import Dict, Any, Optional, List
from ninja_extra import Router
from ninja import Schema
from ninja.errors import HttpError
from django.conf import settings
from django.utils import timezone
from django.http import HttpRequest
import os

# 모델 import 추가
from .models import Conversation, ChatMessage, ChatLog, ChatMetrics

logger = logging.getLogger(__name__)

# 챗봇 서버 URL 설정
CHATBOT_BASE_URL = os.getenv('CHATBOT_URL', 'http://localhost:8000')

# 대화 기록 관리 헬퍼 함수들
def get_or_create_conversation(session_id: str, user_ip: str = None) -> Conversation:
    """대화 세션을 가져오거나 새로 생성"""
    conversation, created = Conversation.objects.get_or_create(
        session_id=session_id,
        defaults={
            'user_ip': user_ip,
            'is_active': True
        }
    )
    if not created:
        conversation.updated_at = timezone.now()
        conversation.save()
    return conversation

def save_chat_message(conversation: Conversation, message_type: str, content: str, 
                     confidence: float = None, sources: List[Dict] = None, 
                     processing_time: float = None) -> ChatMessage:
    """챗봇 메시지를 데이터베이스에 저장"""
    return ChatMessage.objects.create(
        conversation=conversation,
        message_type=message_type,
        content=content,
        confidence=confidence,
        sources=sources,
        processing_time=processing_time
    )

def save_chat_log(level: str, message: str, module: str = None, 
                 user_ip: str = None, session_id: str = None):
    """챗봇 로그를 데이터베이스에 저장"""
    ChatLog.objects.create(
        level=level,
        message=message,
        module=module,
        user_ip=user_ip,
        session_id=session_id
    )

def update_chat_metrics(session_id: str, success: bool, response_time: float, confidence: float = None):
    """챗봇 메트릭 업데이트"""
    metrics, created = ChatMetrics.objects.get_or_create(
        session_id=session_id,
        defaults={
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'average_confidence': 0
        }
    )
    
    metrics.total_requests += 1
    if success:
        metrics.successful_requests += 1
    else:
        metrics.failed_requests += 1
    
    # 평균 응답 시간 업데이트
    if metrics.total_requests == 1:
        metrics.average_response_time = response_time
    else:
        metrics.average_response_time = (
            (metrics.average_response_time * (metrics.total_requests - 1) + response_time) 
            / metrics.total_requests
        )
    
    # 평균 신뢰도 업데이트 (confidence가 있는 경우만)
    if confidence is not None:
        if metrics.total_requests == 1:
            metrics.average_confidence = confidence
        else:
            metrics.average_confidence = (
                (metrics.average_confidence * (metrics.total_requests - 1) + confidence) 
                / metrics.total_requests
            )
    
    metrics.save()

router = Router()

# Pydantic 스키마 정의
class ChatMessageRequest(Schema):
    message: str

class ChatMessageResponse(Schema):
    success: bool
    response: str
    timestamp: str

class AIQuestionRequest(Schema):
    question: str
    mode: str = 'accuracy'
    k: str = 'auto'

class AIQuestionResponse(Schema):
    answer: str
    confidence: float
    sources: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    fallback_used: Optional[bool] = None

class ChatServerStatus(Schema):
    status: str
    ai_available: bool
    model_loaded: bool
    total_pdfs: int
    total_chunks: int
    memory_usage: Optional[Dict[str, Any]] = None

class BatchQuestionRequest(Schema):
    items: List[Dict[str, Any]]
    mode: str = 'accuracy'

class BatchQuestionResponse(Schema):
    results: List[Dict[str, Any]]
    config_hash: str

# 비동기 HTTP 클라이언트 헬퍼 함수 (최적화된 버전)
async def make_chatbot_request(method: str, endpoint: str, data: Dict[str, Any] = None, timeout: int = 60) -> Dict[str, Any]:
    """챗봇 서버에 HTTP 요청을 보내는 헬퍼 함수"""
    url = f"{CHATBOT_BASE_URL}{endpoint}"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method.upper() == 'GET':
                response = await client.get(url)
            elif method.upper() == 'POST':
                response = await client.post(url, json=data)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")
            
            response.raise_for_status()
            return response.json()
            
    except httpx.TimeoutException:
        # 타임아웃은 중요한 오류이므로 로깅 유지
        logger.error(f"챗봇 서버 요청 타임아웃: {url}")
        raise HttpError(504, "챗봇 서버 응답 시간이 초과되었습니다.")
    except httpx.ConnectError:
        # 연결 오류는 중요한 오류이므로 로깅 유지
        logger.error(f"챗봇 서버 연결 실패: {url}")
        raise HttpError(503, "챗봇 서버에 연결할 수 없습니다.")
    except httpx.HTTPStatusError as e:
        # HTTP 오류는 중요한 오류이므로 로깅 유지
        logger.error(f"챗봇 서버 HTTP 오류: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 404:
            raise HttpError(404, "요청한 리소스를 찾을 수 없습니다.")
        elif e.response.status_code == 500:
            raise HttpError(500, "챗봇 서버 내부 오류가 발생했습니다.")
        else:
            raise HttpError(502, f"챗봇 서버 오류: {e.response.status_code}")
    except Exception as e:
        # 일반적인 예외는 디버그 레벨로 로깅
        logger.debug(f"챗봇 서버 요청 중 예외 발생: {str(e)}")
        raise HttpError(500, "챗봇 서버 요청 처리 중 오류가 발생했습니다.")

# 동기 래퍼 함수
def sync_make_chatbot_request(method: str, endpoint: str, data: Dict[str, Any] = None, timeout: int = 60) -> Dict[str, Any]:
    """비동기 챗봇 요청을 동기적으로 실행하는 래퍼"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(make_chatbot_request(method, endpoint, data, timeout))

# API 엔드포인트들

@router.post("/chat", response=ChatMessageResponse)
def proxy_simple_chat(request, data: ChatMessageRequest):
    """간단한 챗봇 메시지 프록시"""
    try:
        response_data = sync_make_chatbot_request(
            method='POST',
            endpoint='/api/ask',
            data={'question': data.message, 'mode': 'accuracy', 'k': 'auto'},
            timeout=120  # AI 처리 시간을 고려하여 타임아웃 증가
        )
        
        return ChatMessageResponse(
            success=True,
            response=response_data.get('answer', ''),
            timestamp=str(response_data.get('timestamp', ''))
        )
        
    except HttpError:
        raise
    except Exception as e:
        logger.error(f"간단한 챗봇 프록시 오류: {str(e)}")
        raise HttpError(500, "챗봇 메시지 처리 중 오류가 발생했습니다.")

@router.post("/ask", response=AIQuestionResponse)
def proxy_ai_question(request: HttpRequest, data: AIQuestionRequest):
    """AI 질문 답변 프록시 (대화 기록 저장 포함)"""
    start_time = time.time()
    session_id = request.headers.get('X-Session-ID', str(uuid.uuid4()))
    user_ip = request.META.get('REMOTE_ADDR')
    
    try:
        # 대화 세션 가져오기 또는 생성
        conversation = get_or_create_conversation(session_id, user_ip)
        
        # 사용자 질문 저장
        save_chat_message(
            conversation=conversation,
            message_type='user',
            content=data.question
        )
        
        # 로그 저장
        save_chat_log(
            level='INFO',
            message=f"사용자 질문 수신: {data.question[:100]}...",
            module='proxy_ai_question',
            user_ip=user_ip,
            session_id=session_id
        )
        
        request_data = {
            'question': data.question,
            'mode': data.mode,
            'k': data.k
        }
        
        response_data = sync_make_chatbot_request(
            method='POST',
            endpoint='/api/ask',
            data=request_data,
            timeout=120  # AI 처리 시간을 고려하여 더 긴 타임아웃
        )
        
        processing_time = time.time() - start_time
        
        # fallback_used 필드 처리 - 문자열 'none'을 False로 변환
        fallback_used = response_data.get('fallback_used', False)
        if isinstance(fallback_used, str):
            fallback_used = fallback_used.lower() not in ['none', 'false', '0', '']
        elif fallback_used is None:
            fallback_used = False
        
        answer = response_data.get('answer', '')
        confidence = response_data.get('confidence', 0.0)
        sources = response_data.get('sources', [])
        
        # 챗봇 답변 저장
        save_chat_message(
            conversation=conversation,
            message_type='bot',
            content=answer,
            confidence=confidence,
            sources=sources,
            processing_time=processing_time
        )
        
        # 메트릭 업데이트
        update_chat_metrics(session_id, True, processing_time, confidence)
        
        # 성공 로그 저장
        save_chat_log(
            level='INFO',
            message=f"AI 답변 생성 완료: 신뢰도 {confidence:.2f}, 처리시간 {processing_time:.2f}초",
            module='proxy_ai_question',
            user_ip=user_ip,
            session_id=session_id
        )
            
        return AIQuestionResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            metrics=response_data.get('metrics', {}),
            fallback_used=fallback_used
        )
        
    except HttpError:
        # 실패 메트릭 업데이트
        processing_time = time.time() - start_time
        update_chat_metrics(session_id, False, processing_time)
        
        # 실패 로그 저장
        save_chat_log(
            level='ERROR',
            message=f"AI 질문 처리 실패: {data.question[:100]}...",
            module='proxy_ai_question',
            user_ip=user_ip,
            session_id=session_id
        )
        raise
    except Exception as e:
        # 실패 메트릭 업데이트
        processing_time = time.time() - start_time
        update_chat_metrics(session_id, False, processing_time)
        
        logger.error(f"AI 질문 프록시 오류: {str(e)}")
        save_chat_log(
            level='ERROR',
            message=f"AI 질문 프록시 예외: {str(e)}",
            module='proxy_ai_question',
            user_ip=user_ip,
            session_id=session_id
        )
        raise HttpError(500, "AI 질문 처리 중 오류가 발생했습니다.")

@router.post("/batch", response=BatchQuestionResponse)
def proxy_batch_questions(request, data: BatchQuestionRequest):
    """배치 질문 답변 프록시"""
    try:
        request_data = {
            'items': data.items,
            'mode': data.mode
        }
        
        response_data = sync_make_chatbot_request(
            method='POST',
            endpoint='/api/qa/batch',
            data=request_data,
            timeout=300  # 배치 처리 시간을 고려하여 더 긴 타임아웃
        )
        
        return BatchQuestionResponse(
            results=response_data.get('results', []),
            config_hash=response_data.get('config_hash', '')
        )
        
    except HttpError:
        raise
    except Exception as e:
        logger.error(f"배치 질문 프록시 오류: {str(e)}")
        raise HttpError(500, "배치 질문 처리 중 오류가 발생했습니다.")

@router.get("/status", response=ChatServerStatus)
def proxy_chatbot_status(request):
    """챗봇 서버 상태 조회 프록시"""
    try:
        response_data = sync_make_chatbot_request(
            method='GET',
            endpoint='/status',
            timeout=10
        )
        
        return ChatServerStatus(
            status=response_data.get('status', 'unknown'),
            ai_available=response_data.get('ai_available', False),
            model_loaded=response_data.get('model_loaded', False),
            total_pdfs=response_data.get('total_pdfs', 0),
            total_chunks=response_data.get('total_chunks', 0),
            memory_usage=response_data.get('memory_usage', None)
        )
        
    except HttpError:
        raise
    except Exception as e:
        # 상태 조회 실패는 디버그 레벨로 로깅
        logger.debug(f"챗봇 상태 프록시 오류: {str(e)}")
        raise HttpError(500, "챗봇 상태 조회 중 오류가 발생했습니다.")

@router.get("/health")
def proxy_health_check(request):
    """챗봇 서버 헬스 체크 프록시 - 새로운 FastAPI 서버에 맞게 수정"""
    try:
        response_data = sync_make_chatbot_request(
            method='GET',
            endpoint='/healthz',
            timeout=5
        )
        
        return response_data
        
    except HttpError:
        raise
    except Exception as e:
        # 헬스 체크 실패는 디버그 레벨로 로깅
        logger.debug(f"헬스 체크 프록시 오류: {str(e)}")
        raise HttpError(500, "헬스 체크 중 오류가 발생했습니다.")

@router.get("/metrics")
def proxy_metrics(request):
    """챗봇 서버 메트릭 조회 프록시"""
    try:
        url = f"{CHATBOT_BASE_URL}/metrics"
        
        import httpx
        with httpx.Client(timeout=10) as client:
            response = client.get(url)
            response.raise_for_status()
            
            # 메트릭은 텍스트 형식으로 반환되므로 텍스트로 처리
            return {"metrics": response.text}
        
    except httpx.TimeoutException:
        logger.error(f"챗봇 서버 메트릭 요청 타임아웃: {url}")
        raise HttpError(504, "챗봇 서버 메트릭 응답 시간이 초과되었습니다.")
    except httpx.ConnectError:
        logger.error(f"챗봇 서버 메트릭 연결 실패: {url}")
        raise HttpError(503, "챗봇 서버에 연결할 수 없습니다.")
    except httpx.HTTPStatusError as e:
        logger.error(f"챗봇 서버 메트릭 HTTP 오류: {e.response.status_code} - {e.response.text}")
        raise HttpError(502, f"챗봇 서버 메트릭 오류: {e.response.status_code}")
    except Exception as e:
        logger.debug(f"메트릭 프록시 오류: {str(e)}")
        raise HttpError(500, "메트릭 조회 중 오류가 발생했습니다.")

# 대화 기록 관리 API 엔드포인트들

class ConversationResponse(Schema):
    session_id: str
    user_ip: Optional[str]
    created_at: str
    updated_at: str
    is_active: bool
    message_count: int

class ChatMessageResponse(Schema):
    id: int
    message_type: str
    content: str
    confidence: Optional[float]
    sources: Optional[List[Dict[str, Any]]]
    processing_time: Optional[float]
    created_at: str

class ConversationDetailResponse(Schema):
    conversation: ConversationResponse
    messages: List[ChatMessageResponse]

@router.get("/conversations", response=List[ConversationResponse])
def get_conversations(request: HttpRequest):
    """모든 대화 세션 목록 조회"""
    try:
        conversations = Conversation.objects.all()[:50]  # 최근 50개만
        result = []
        
        for conv in conversations:
            message_count = conv.messages.count()
            result.append(ConversationResponse(
                session_id=conv.session_id,
                user_ip=conv.user_ip,
                created_at=conv.created_at.isoformat(),
                updated_at=conv.updated_at.isoformat(),
                is_active=conv.is_active,
                message_count=message_count
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"대화 목록 조회 오류: {str(e)}")
        raise HttpError(500, "대화 목록 조회 중 오류가 발생했습니다.")

@router.get("/conversations/{session_id}", response=ConversationDetailResponse)
def get_conversation_detail(request: HttpRequest, session_id: str):
    """특정 대화 세션의 상세 정보 조회"""
    try:
        conversation = Conversation.objects.get(session_id=session_id)
        messages = conversation.messages.all()
        
        conversation_data = ConversationResponse(
            session_id=conversation.session_id,
            user_ip=conversation.user_ip,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            is_active=conversation.is_active,
            message_count=messages.count()
        )
        
        messages_data = []
        for msg in messages:
            messages_data.append(ChatMessageResponse(
                id=msg.id,
                message_type=msg.message_type,
                content=msg.content,
                confidence=msg.confidence,
                sources=msg.sources,
                processing_time=msg.processing_time,
                created_at=msg.created_at.isoformat()
            ))
        
        return ConversationDetailResponse(
            conversation=conversation_data,
            messages=messages_data
        )
        
    except Conversation.DoesNotExist:
        raise HttpError(404, "대화 세션을 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"대화 상세 조회 오류: {str(e)}")
        raise HttpError(500, "대화 상세 조회 중 오류가 발생했습니다.")

@router.delete("/conversations/{session_id}")
def delete_conversation(request: HttpRequest, session_id: str):
    """특정 대화 세션 삭제"""
    try:
        conversation = Conversation.objects.get(session_id=session_id)
        conversation.delete()
        
        save_chat_log(
            level='INFO',
            message=f"대화 세션 삭제: {session_id}",
            module='delete_conversation',
            user_ip=request.META.get('REMOTE_ADDR'),
            session_id=session_id
        )
        
        return {"success": True, "message": "대화 세션이 삭제되었습니다."}
        
    except Conversation.DoesNotExist:
        raise HttpError(404, "대화 세션을 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"대화 삭제 오류: {str(e)}")
        raise HttpError(500, "대화 삭제 중 오류가 발생했습니다.")

@router.get("/logs", response=List[Dict[str, Any]])
def get_chat_logs(request: HttpRequest, level: str = None, limit: int = 100):
    """챗봇 로그 조회"""
    try:
        logs_query = ChatLog.objects.all()
        
        if level:
            logs_query = logs_query.filter(level=level.upper())
        
        logs = logs_query[:limit]
        
        result = []
        for log in logs:
            result.append({
                'id': log.id,
                'level': log.level,
                'message': log.message,
                'module': log.module,
                'user_ip': log.user_ip,
                'session_id': log.session_id,
                'created_at': log.created_at.isoformat()
            })
        
        return result
        
    except Exception as e:
        logger.error(f"로그 조회 오류: {str(e)}")
        raise HttpError(500, "로그 조회 중 오류가 발생했습니다.")

@router.get("/metrics", response=List[Dict[str, Any]])
def get_chat_metrics(request: HttpRequest):
    """챗봇 메트릭 조회"""
    try:
        metrics = ChatMetrics.objects.all()[:20]  # 최근 20개
        
        result = []
        for metric in metrics:
            result.append({
                'session_id': metric.session_id,
                'total_requests': metric.total_requests,
                'successful_requests': metric.successful_requests,
                'failed_requests': metric.failed_requests,
                'success_rate': (
                    metric.successful_requests / metric.total_requests * 100 
                    if metric.total_requests > 0 else 0
                ),
                'average_response_time': metric.average_response_time,
                'average_confidence': metric.average_confidence,
                'created_at': metric.created_at.isoformat()
            })
        
        return result
        
    except Exception as e:
        logger.error(f"메트릭 조회 오류: {str(e)}")
        raise HttpError(500, "메트릭 조회 중 오류가 발생했습니다.")
