"""
Django 연동 클라이언트

Django 백엔드에서 PDF QA API를 쉽게 사용할 수 있도록 하는
클라이언트 라이브러리와 유틸리티 함수들을 제공합니다.
"""

import requests
import json
import logging
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import os

# Django 프로젝트에서 사용할 수 있도록 설정
try:
    from django.conf import settings
    from django.core.files.uploadedfile import UploadedFile
    from django.http import JsonResponse
    from django.views.decorators.http import require_http_methods
    from django.views.decorators.csrf import csrf_exempt
    from django.utils.decorators import method_decorator
    from django.views import View
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PDFQAConfig:
    """PDF QA API 설정"""
    api_base_url: str = "http://localhost:8000"
    timeout: int = 30
    max_retries: int = 3
    
class PDFQAError(Exception):
    """PDF QA API 에러"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

class PDFQAClient:
    """
    Django에서 사용하는 PDF QA API 클라이언트
    
    사용 예시:
    ```python
    # settings.py
    PDFQA_CONFIG = {
        'API_BASE_URL': 'http://localhost:8000',
        'TIMEOUT': 30
    }
    
    # views.py
    from .pdf_qa_client import PDFQAClient
    
    def ask_question_view(request):
        client = PDFQAClient()
        
        result = client.ask_question(
            question=request.POST['question'],
            pdf_id=request.POST['pdf_id']
        )
        
        return JsonResponse(result)
    ```
    """
    
    def __init__(self, config: Optional[PDFQAConfig] = None):
        """
        클라이언트 초기화
        
        Args:
            config: API 설정 (None인 경우 Django settings에서 로드)
        """
        if config:
            self.config = config
        else:
            # Django settings에서 설정 로드
            if DJANGO_AVAILABLE and hasattr(settings, 'PDFQA_CONFIG'):
                pdfqa_settings = settings.PDFQA_CONFIG
                self.config = PDFQAConfig(
                    api_base_url=pdfqa_settings.get('API_BASE_URL', 'http://localhost:8000'),
                    timeout=pdfqa_settings.get('TIMEOUT', 30),
                    max_retries=pdfqa_settings.get('MAX_RETRIES', 3)
                )
            else:
                self.config = PDFQAConfig()
        
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
        
        logger.info(f"PDFQAClient 초기화: {self.config.api_base_url}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        API 요청 실행
        
        Args:
            method: HTTP 메서드
            endpoint: API 엔드포인트
            **kwargs: requests 라이브러리에 전달할 인자들
            
        Returns:
            API 응답 데이터
            
        Raises:
            PDFQAError: API 요청 실패 시
        """
        url = f"{self.config.api_base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API 요청 실패 (시도 {attempt + 1}/{self.config.max_retries}): {e}")
                
                if attempt == self.config.max_retries - 1:
                    # 마지막 시도에서도 실패
                    error_data = None
                    status_code = None
                    
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        try:
                            error_data = e.response.json()
                        except:
                            error_data = {"detail": e.response.text}
                    
                    raise PDFQAError(
                        f"API 요청 실패: {str(e)}",
                        status_code=status_code,
                        response_data=error_data
                    )
    
    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        return self._make_request('GET', '/status')
    
    def health_check(self) -> Dict:
        """헬스 체크"""
        return self._make_request('GET', '/health')
    
    def upload_pdf(self, file_path: str, filename: Optional[str] = None) -> Dict:
        """
        PDF 파일 업로드
        
        Args:
            file_path: 업로드할 PDF 파일 경로
            filename: 파일명 (None인 경우 file_path에서 추출)
            
        Returns:
            업로드 결과
        """
        if not os.path.exists(file_path):
            raise PDFQAError(f"파일을 찾을 수 없습니다: {file_path}")
        
        if not filename:
            filename = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'application/pdf')}
            return self._make_request('POST', '/upload_pdf', files=files)
    
    def upload_pdf_from_django_file(self, uploaded_file: 'UploadedFile') -> Dict:
        """
        Django UploadedFile 객체에서 PDF 업로드
        
        Args:
            uploaded_file: Django의 UploadedFile 객체
            
        Returns:
            업로드 결과
        """
        if not DJANGO_AVAILABLE:
            raise PDFQAError("Django가 설치되지 않았습니다.")
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        try:
            return self.upload_pdf(temp_path, uploaded_file.name)
        finally:
            # 임시 파일 삭제
            os.unlink(temp_path)
    
    def ask_question(self, 
                    question: str, 
                    pdf_id: str,
                    use_conversation_context: bool = True,
                    max_chunks: int = 5) -> Dict:
        """
        질문하기
        
        Args:
            question: 사용자 질문
            pdf_id: PDF 식별자
            use_conversation_context: 이전 대화 컨텍스트 사용 여부
            max_chunks: 검색할 최대 청크 수
            
        Returns:
            답변 결과
        """
        data = {
            "question": question,
            "pdf_id": pdf_id,
            "use_conversation_context": use_conversation_context,
            "max_chunks": max_chunks
        }
        
        return self._make_request('POST', '/ask', json=data)
    
    def django_ask_question(self,
                          question: str,
                          pdf_id: str,
                          conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Django 특화 질문 엔드포인트 호출
        
        Args:
            question: 질문
            pdf_id: PDF ID
            conversation_history: 대화 기록
            
        Returns:
            답변 결과
        """
        params = {
            "question": question,
            "pdf_id": pdf_id
        }
        
        if conversation_history:
            params["conversation_history"] = conversation_history
        
        return self._make_request('POST', '/django/ask', json=params)
    
    def get_conversation_history(self, pdf_id: str, max_items: int = 10) -> Dict:
        """대화 기록 조회"""
        params = {
            "pdf_id": pdf_id,
            "max_items": max_items
        }
        
        return self._make_request('GET', '/conversation_history', params=params)
    
    def clear_conversation_history(self) -> Dict:
        """대화 기록 초기화"""
        return self._make_request('DELETE', '/conversation_history')
    
    def list_pdfs(self) -> Dict:
        """등록된 PDF 목록 조회"""
        return self._make_request('GET', '/pdfs')
    
    def delete_pdf(self, pdf_id: str) -> Dict:
        """PDF 삭제"""
        return self._make_request('DELETE', f'/pdfs/{pdf_id}')
    
    def configure_model(self, 
                       model_type: str,
                       model_name: str,
                       max_length: int = 512,
                       temperature: float = 0.7,
                       top_p: float = 0.9) -> Dict:
        """모델 설정 변경"""
        data = {
            "model_type": model_type,
            "model_name": model_name,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p
        }
        
        return self._make_request('POST', '/configure_model', json=data)
    
    def evaluate_system(self,
                       questions: List[str],
                       generated_answers: List[str],
                       reference_answers: List[str]) -> Dict:
        """시스템 성능 평가"""
        data = {
            "questions": questions,
            "generated_answers": generated_answers,
            "reference_answers": reference_answers
        }
        
        return self._make_request('POST', '/evaluate', json=data)

# Django 뷰 클래스들 (Django가 설치된 경우에만 사용 가능)

if DJANGO_AVAILABLE:
    
    @method_decorator(csrf_exempt, name='dispatch')
    class PDFQAAPIView(View):
        """
        Django 뷰 클래스 기반의 PDF QA API 래퍼
        
        사용 예시:
        ```python
        # urls.py
        from django.urls import path
        from .views import PDFQAAPIView
        
        urlpatterns = [
            path('api/pdfqa/', PDFQAAPIView.as_view(), name='pdfqa_api'),
        ]
        
        # 클라이언트에서 사용:
        # POST /api/pdfqa/ 
        # {
        #   "action": "ask_question",
        #   "question": "질문 내용",
        #   "pdf_id": "pdf_식별자"
        # }
        ```
        """
        
        def __init__(self):
            super().__init__()
            self.client = PDFQAClient()
        
        def post(self, request):
            """POST 요청 처리"""
            try:
                data = json.loads(request.body)
                action = data.get('action')
                
                if action == 'upload_pdf':
                    return self._handle_upload_pdf(request, data)
                elif action == 'ask_question':
                    return self._handle_ask_question(data)
                elif action == 'get_conversation_history':
                    return self._handle_get_conversation_history(data)
                elif action == 'clear_conversation_history':
                    return self._handle_clear_conversation_history()
                elif action == 'list_pdfs':
                    return self._handle_list_pdfs()
                elif action == 'delete_pdf':
                    return self._handle_delete_pdf(data)
                elif action == 'configure_model':
                    return self._handle_configure_model(data)
                else:
                    return JsonResponse({'error': f'알 수 없는 액션: {action}'}, status=400)
                    
            except json.JSONDecodeError:
                return JsonResponse({'error': '잘못된 JSON 형식'}, status=400)
            except PDFQAError as e:
                return JsonResponse({
                    'error': str(e),
                    'status_code': e.status_code,
                    'details': e.response_data
                }, status=e.status_code or 500)
            except Exception as e:
                logger.error(f"PDF QA API 요청 처리 중 오류: {e}")
                return JsonResponse({'error': '내부 서버 오류'}, status=500)
        
        def get(self, request):
            """GET 요청 처리"""
            try:
                action = request.GET.get('action')
                
                if action == 'status':
                    result = self.client.get_system_status()
                    return JsonResponse(result)
                elif action == 'health':
                    result = self.client.health_check()
                    return JsonResponse(result)
                elif action == 'list_pdfs':
                    result = self.client.list_pdfs()
                    return JsonResponse(result)
                else:
                    return JsonResponse({'error': f'알 수 없는 액션: {action}'}, status=400)
                    
            except PDFQAError as e:
                return JsonResponse({
                    'error': str(e),
                    'status_code': e.status_code,
                    'details': e.response_data
                }, status=e.status_code or 500)
            except Exception as e:
                logger.error(f"PDF QA API GET 요청 처리 중 오류: {e}")
                return JsonResponse({'error': '내부 서버 오류'}, status=500)
        
        def _handle_upload_pdf(self, request, data):
            """PDF 업로드 처리"""
            if 'file' not in request.FILES:
                return JsonResponse({'error': '파일이 업로드되지 않았습니다.'}, status=400)
            
            uploaded_file = request.FILES['file']
            result = self.client.upload_pdf_from_django_file(uploaded_file)
            return JsonResponse(result)
        
        def _handle_ask_question(self, data):
            """질문 처리"""
            question = data.get('question')
            pdf_id = data.get('pdf_id')
            
            if not question or not pdf_id:
                return JsonResponse({'error': '질문과 PDF ID가 필요합니다.'}, status=400)
            
            result = self.client.ask_question(
                question=question,
                pdf_id=pdf_id,
                use_conversation_context=data.get('use_conversation_context', True),
                max_chunks=data.get('max_chunks', 5)
            )
            return JsonResponse(result)
        
        def _handle_get_conversation_history(self, data):
            """대화 기록 조회 처리"""
            pdf_id = data.get('pdf_id')
            if not pdf_id:
                return JsonResponse({'error': 'PDF ID가 필요합니다.'}, status=400)
            
            result = self.client.get_conversation_history(
                pdf_id=pdf_id,
                max_items=data.get('max_items', 10)
            )
            return JsonResponse(result)
        
        def _handle_clear_conversation_history(self):
            """대화 기록 초기화 처리"""
            result = self.client.clear_conversation_history()
            return JsonResponse(result)
        
        def _handle_list_pdfs(self):
            """PDF 목록 조회 처리"""
            result = self.client.list_pdfs()
            return JsonResponse(result)
        
        def _handle_delete_pdf(self, data):
            """PDF 삭제 처리"""
            pdf_id = data.get('pdf_id')
            if not pdf_id:
                return JsonResponse({'error': 'PDF ID가 필요합니다.'}, status=400)
            
            result = self.client.delete_pdf(pdf_id)
            return JsonResponse(result)
        
        def _handle_configure_model(self, data):
            """모델 설정 변경 처리"""
            result = self.client.configure_model(
                model_type=data.get('model_type', 'ollama'),
                model_name=data.get('model_name', 'llama2:7b'),
                max_length=data.get('max_length', 512),
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 0.9)
            )
            return JsonResponse(result)

    # 함수형 뷰들
    
    @csrf_exempt
    @require_http_methods(["POST"])
    def upload_pdf_view(request):
        """PDF 업로드 전용 뷰"""
        try:
            if 'file' not in request.FILES:
                return JsonResponse({'error': '파일이 업로드되지 않았습니다.'}, status=400)
            
            client = PDFQAClient()
            uploaded_file = request.FILES['file']
            result = client.upload_pdf_from_django_file(uploaded_file)
            
            return JsonResponse(result)
            
        except PDFQAError as e:
            return JsonResponse({
                'error': str(e),
                'status_code': e.status_code
            }, status=e.status_code or 500)
        except Exception as e:
            logger.error(f"PDF 업로드 중 오류: {e}")
            return JsonResponse({'error': '파일 업로드 실패'}, status=500)
    
    @csrf_exempt
    @require_http_methods(["POST"])
    def ask_question_view(request):
        """질문하기 전용 뷰"""
        try:
            data = json.loads(request.body)
            
            question = data.get('question')
            pdf_id = data.get('pdf_id')
            
            if not question or not pdf_id:
                return JsonResponse({'error': '질문과 PDF ID가 필요합니다.'}, status=400)
            
            client = PDFQAClient()
            result = client.ask_question(
                question=question,
                pdf_id=pdf_id,
                use_conversation_context=data.get('use_conversation_context', True),
                max_chunks=data.get('max_chunks', 5)
            )
            
            return JsonResponse(result)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': '잘못된 JSON 형식'}, status=400)
        except PDFQAError as e:
            return JsonResponse({
                'error': str(e),
                'status_code': e.status_code
            }, status=e.status_code or 500)
        except Exception as e:
            logger.error(f"질문 처리 중 오류: {e}")
            return JsonResponse({'error': '질문 처리 실패'}, status=500)

# 유틸리티 함수들

def create_django_urls():
    """
    Django URL 패턴 생성 헬퍼 함수
    
    사용 예시:
    ```python
    # urls.py
    from django.urls import path, include
    from .pdf_qa_client import create_django_urls
    
    urlpatterns = [
        path('api/pdfqa/', include(create_django_urls())),
    ]
    ```
    """
    if not DJANGO_AVAILABLE:
        raise ImportError("Django가 설치되지 않았습니다.")
    
    from django.urls import path
    
    return [
        path('', PDFQAAPIView.as_view(), name='pdfqa_api'),
        path('upload/', upload_pdf_view, name='pdfqa_upload'),
        path('ask/', ask_question_view, name='pdfqa_ask'),
    ]

def get_client_from_settings() -> PDFQAClient:
    """Django settings에서 클라이언트 설정을 로드하여 생성"""
    return PDFQAClient()

# Django 모델 (선택적)
if DJANGO_AVAILABLE:
    from django.db import models
    
    class PDFDocument(models.Model):
        """PDF 문서 모델 (Django ORM용)"""
        pdf_id = models.CharField(max_length=100, unique=True)
        filename = models.CharField(max_length=255)
        upload_time = models.DateTimeField(auto_now_add=True)
        total_pages = models.IntegerField(default=0)
        total_chunks = models.IntegerField(default=0)
        processing_time = models.FloatField(default=0.0)
        
        class Meta:
            db_table = 'pdfqa_documents'
            ordering = ['-upload_time']
        
        def __str__(self):
            return f"{self.filename} ({self.pdf_id})"
    
    class ConversationHistory(models.Model):
        """대화 기록 모델"""
        pdf_document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE)
        question = models.TextField()
        answer = models.TextField()
        confidence_score = models.FloatField(default=0.0)
        question_type = models.CharField(max_length=50, default='')
        used_chunks = models.JSONField(default=list)
        created_at = models.DateTimeField(auto_now_add=True)
        
        class Meta:
            db_table = 'pdfqa_conversations'
            ordering = ['-created_at']

if __name__ == "__main__":
    # 테스트 코드
    client = PDFQAClient()
    
    try:
        status = client.health_check()
        print(f"API 서버 상태: {status}")
    except PDFQAError as e:
        print(f"API 연결 실패: {e}")
    
    print("Django 클라이언트 모듈이 정상적으로 로드되었습니다.")
