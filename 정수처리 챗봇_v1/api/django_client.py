"""
Django 백엔드에서 PDF QA API를 호출하기 위한 클라이언트

이 모듈은 Django 애플리케이션에서 PDF QA 시스템의 API를 쉽게 호출할 수 있도록
하는 클라이언트 클래스를 제공합니다.
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class PDFQAClient:
    """
    PDF QA API 클라이언트
    
    Django 백엔드에서 PDF QA 시스템과 통신하기 위한 클라이언트입니다.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        클라이언트 초기화
        
        Args:
            base_url: PDF QA API 서버의 기본 URL
            timeout: 요청 타임아웃 (초)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # 기본 헤더 설정
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        HTTP 요청 수행
        
        Args:
            method: HTTP 메서드 (GET, POST, etc.)
            endpoint: API 엔드포인트
            **kwargs: requests 라이브러리에 전달할 추가 인자들
            
        Returns:
            API 응답 데이터
            
        Raises:
            requests.RequestException: HTTP 요청 실패 시
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"API 요청 실패: {method} {url} - {e}")
            raise
    
    def ask_question(self, 
                    question: str, 
                    pdf_id: Optional[str] = None,
                    conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        질문에 대한 답변 요청
        
        Args:
            question: 사용자 질문
            pdf_id: PDF 문서 ID (선택사항)
            conversation_history: 이전 대화 기록 (선택사항)
            
        Returns:
            답변 결과
        """
        data = {
            "question": question,
            "pdf_id": pdf_id or "default",
            "conversation_history": conversation_history or []
        }
        
        return self._make_request("POST", "/django/ask", json=data)
    
    def upload_pdf(self, file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        PDF 파일 업로드
        
        Args:
            file_path: 업로드할 PDF 파일 경로
            filename: 파일명 (선택사항, 없으면 경로에서 추출)
            
        Returns:
            업로드 결과
        """
        if filename is None:
            import os
            filename = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'application/pdf')}
            return self._make_request("POST", "/upload_pdf", files=files)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        시스템 상태 조회
        
        Returns:
            시스템 상태 정보
        """
        return self._make_request("GET", "/status")
    
    def get_conversation_history(self, pdf_id: str, max_items: int = 10) -> Dict[str, Any]:
        """
        대화 기록 조회
        
        Args:
            pdf_id: PDF 문서 ID
            max_items: 최대 조회 항목 수
            
        Returns:
            대화 기록
        """
        params = {"pdf_id": pdf_id, "max_items": max_items}
        return self._make_request("GET", "/conversation_history", params=params)
    
    def clear_conversation_history(self) -> Dict[str, Any]:
        """
        대화 기록 초기화
        
        Returns:
            초기화 결과
        """
        return self._make_request("DELETE", "/conversation_history")
    
    def list_pdfs(self) -> Dict[str, Any]:
        """
        등록된 PDF 목록 조회
        
        Returns:
            PDF 목록
        """
        return self._make_request("GET", "/pdfs")
    
    def delete_pdf(self, pdf_id: str) -> Dict[str, Any]:
        """
        PDF 삭제
        
        Args:
            pdf_id: 삭제할 PDF ID
            
        Returns:
            삭제 결과
        """
        return self._make_request("DELETE", f"/pdfs/{pdf_id}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        서버 헬스 체크
        
        Returns:
            서버 상태
        """
        return self._make_request("GET", "/health")
    


# Django에서 사용하기 위한 편의 함수들

def create_qa_client(base_url: str = "http://localhost:8000") -> PDFQAClient:
    """
    PDF QA 클라이언트 생성 (Django settings에서 사용)
    
    Args:
        base_url: API 서버 URL
        
    Returns:
        PDFQAClient 인스턴스
    """
    return PDFQAClient(base_url=base_url)

def ask_question_simple(question: str, 
                       base_url: str = "http://localhost:8000",
                       pdf_id: Optional[str] = None) -> Dict[str, Any]:
    """
    간단한 질문-답변 함수 (Django view에서 사용)
    
    Args:
        question: 사용자 질문
        base_url: API 서버 URL
        pdf_id: PDF ID (선택사항)
        
    Returns:
        답변 결과
    """
    client = PDFQAClient(base_url=base_url)
    return client.ask_question(question, pdf_id)

# Django settings.py에 추가할 설정 예시
"""
# settings.py에 추가할 설정

# PDF QA API 설정
PDF_QA_API_URL = "http://localhost:8000"  # PDF QA 서버 URL
PDF_QA_TIMEOUT = 30  # 요청 타임아웃 (초)

# 사용 예시:
# from api.django_client import create_qa_client
# qa_client = create_qa_client(PDF_QA_API_URL)
"""

# Django view에서 사용 예시
"""
# views.py 예시

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from api.django_client import ask_question_simple

@csrf_exempt
@require_http_methods(["POST"])
def ask_question_view(request):
    try:
        data = json.loads(request.body)
        question = data.get('question', '')
        
        if not question:
            return JsonResponse({
                'success': False,
                'error': '질문이 필요합니다.'
            }, status=400)
        
        # PDF QA API 호출
        result = ask_question_simple(question)
        
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def upload_pdf_view(request):
    try:
        if 'file' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'PDF 파일이 필요합니다.'
            }, status=400)
        
        pdf_file = request.FILES['file']
        
        # 임시 파일로 저장
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in pdf_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        try:
            # PDF QA API에 업로드
            client = create_qa_client()
            result = client.upload_pdf(temp_path, pdf_file.name)
            
            return JsonResponse(result)
            
        finally:
            # 임시 파일 삭제
            os.unlink(temp_path)
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
"""
