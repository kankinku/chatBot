"""
URL configuration for chatbot project.
"""
from django.urls import path
from ninja_extra import NinjaExtraAPI
from chatbot_proxy.views import router as chatbot_router

api = NinjaExtraAPI()
api.add_router("/chatbot/", chatbot_router)

urlpatterns = [
    path('api/', api.urls),
]
