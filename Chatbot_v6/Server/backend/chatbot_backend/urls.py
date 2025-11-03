"""
URL configuration for chatbot_backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ninja_extra import NinjaExtraAPI
from chatbot_proxy.views import router as chatbot_router

api = NinjaExtraAPI()
api.add_router("/chatbot/", chatbot_router)

# PDF 처리 엔드포인트를 루트 레벨에 추가
@api.post("/process-pdfs")
def proxy_process_pdfs(request):
    """PDF 처리 시작 프록시"""
    from chatbot_proxy.views import sync_make_chatbot_request
    from ninja.errors import HttpError
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        response_data = sync_make_chatbot_request(
            method='POST',
            endpoint='/api/process-pdfs',
            data={},
            timeout=10  # 시작 요청은 빠르게
        )
        return response_data
    except HttpError:
        raise
    except Exception as e:
        logger.error(f"PDF 처리 프록시 오류: {str(e)}")
        raise HttpError(500, "PDF 처리 시작 중 오류가 발생했습니다.")


@api.get("/process-pdfs/status")
def proxy_process_pdfs_status(request):
    """PDF 처리 상태 확인 프록시"""
    from chatbot_proxy.views import sync_make_chatbot_request
    from ninja.errors import HttpError
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        response_data = sync_make_chatbot_request(
            method='GET',
            endpoint='/api/process-pdfs/status',
            timeout=10
        )
        return response_data
    except HttpError:
        raise
    except Exception as e:
        logger.error(f"PDF 처리 상태 확인 프록시 오류: {str(e)}")
        raise HttpError(500, "PDF 처리 상태 확인 중 오류가 발생했습니다.")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', api.urls),
]