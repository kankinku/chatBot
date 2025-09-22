from django.db import models
from django.utils import timezone
import json


class Conversation(models.Model):
    """챗봇 대화 세션"""
    session_id = models.CharField(max_length=100, unique=True, help_text="대화 세션 ID")
    user_ip = models.GenericIPAddressField(null=True, blank=True, help_text="사용자 IP")
    created_at = models.DateTimeField(default=timezone.now, help_text="대화 시작 시간")
    updated_at = models.DateTimeField(auto_now=True, help_text="마지막 대화 시간")
    is_active = models.BooleanField(default=True, help_text="활성 대화 여부")
    
    class Meta:
        db_table = 'conversations'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Conversation {self.session_id} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"


class ChatMessage(models.Model):
    """챗봇 메시지 (질문/답변)"""
    MESSAGE_TYPES = [
        ('user', '사용자 질문'),
        ('bot', '챗봇 답변'),
        ('system', '시스템 메시지'),
    ]
    
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES, help_text="메시지 타입")
    content = models.TextField(help_text="메시지 내용")
    confidence = models.FloatField(null=True, blank=True, help_text="답변 신뢰도 (0-1)")
    sources = models.JSONField(null=True, blank=True, help_text="참조 소스 정보")
    processing_time = models.FloatField(null=True, blank=True, help_text="처리 시간 (초)")
    created_at = models.DateTimeField(default=timezone.now, help_text="메시지 생성 시간")
    
    class Meta:
        db_table = 'chat_messages'
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."


class ChatLog(models.Model):
    """챗봇 시스템 로그"""
    LOG_LEVELS = [
        ('DEBUG', 'Debug'),
        ('INFO', 'Info'),
        ('WARNING', 'Warning'),
        ('ERROR', 'Error'),
        ('CRITICAL', 'Critical'),
    ]
    
    level = models.CharField(max_length=10, choices=LOG_LEVELS, help_text="로그 레벨")
    message = models.TextField(help_text="로그 메시지")
    module = models.CharField(max_length=100, null=True, blank=True, help_text="모듈명")
    user_ip = models.GenericIPAddressField(null=True, blank=True, help_text="사용자 IP")
    session_id = models.CharField(max_length=100, null=True, blank=True, help_text="세션 ID")
    created_at = models.DateTimeField(default=timezone.now, help_text="로그 생성 시간")
    
    class Meta:
        db_table = 'chat_logs'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['level', 'created_at']),
            models.Index(fields=['session_id', 'created_at']),
        ]
    
    def __str__(self):
        return f"[{self.level}] {self.message[:50]}..."


class ChatMetrics(models.Model):
    """챗봇 성능 메트릭"""
    session_id = models.CharField(max_length=100, null=True, blank=True, help_text="세션 ID")
    total_requests = models.IntegerField(default=0, help_text="총 요청 수")
    successful_requests = models.IntegerField(default=0, help_text="성공한 요청 수")
    failed_requests = models.IntegerField(default=0, help_text="실패한 요청 수")
    average_response_time = models.FloatField(default=0, help_text="평균 응답 시간")
    average_confidence = models.FloatField(default=0, help_text="평균 신뢰도")
    created_at = models.DateTimeField(default=timezone.now, help_text="메트릭 생성 시간")
    
    class Meta:
        db_table = 'chat_metrics'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Metrics {self.session_id}: {self.successful_requests}/{self.total_requests}"
