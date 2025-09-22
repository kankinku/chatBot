#!/usr/bin/env python3
"""
인사말 처리 핸들러

사용자의 인사말에 대해 시간대와 질문 유형에 맞는 응답을 생성합니다.
"""

import logging
import os
from typing import Dict, Optional
from datetime import datetime
from core.config.company_config import CompanyConfig

logger = logging.getLogger(__name__)

class GreetingHandler:
    """인사말 처리 핸들러"""
    
    def __init__(self, answer_generator=None):
        """인사말 핸들러 초기화"""
        self.answer_generator = answer_generator
        self.company_config = CompanyConfig()
        
        # 개선된 기본 인사말 템플릿 (LLM 실패 시 폴백용)
        self.fallback_greetings = {
            "morning": [
                "좋은 아침입니다!",
                "상쾌한 아침이네요!",
                "좋은 하루 시작하세요!"
            ],
            "afternoon": [
                "안녕하세요!",
                "반갑습니다!",
                "좋은 오후입니다!"
            ],
            "evening": [
                "좋은 저녁입니다!",
                "편안한 저녁 되세요!",
                "오늘 하루도 수고하셨습니다!"
            ],
            "night": [
                "좋은 밤 되세요!",
                "편안한 밤 되세요!",
                "안녕히 주무세요!"
            ]
        }
        
        # 회사별 맞춤 인사말 (환경변수로 설정 가능)
        from core.config.unified_config import get_config
        self.company_greetings = {
            "default": get_config("COMPANY_GREETING", "범용 RAG 시스템에 오신 것을 환영합니다!"),
            "general": get_config("COMPANY_GREETING", "범용 RAG 시스템에 오신 것을 환영합니다!"),
        }
    
    def get_greeting_response(self, question: str) -> Dict:
        """
        인사말 응답 생성
        
        Args:
            question: 사용자 질문
            
        Returns:
            응답 정보가 담긴 딕셔너리
        """
        try:
            # 시간대 감지
            current_hour = datetime.now().hour
            time_of_day = self._get_time_of_day(current_hour)
            
            # 질문 유형 감지
            greeting_type = self._classify_greeting_type(question)
            
            # 기본 응답 생성
            response = self._generate_fallback_response(greeting_type, time_of_day)
            
            return {
                "answer": response,
                "greeting_type": greeting_type,
                "time_of_day": time_of_day,
                "confidence_score": 0.8,
                "generation_time": 0.001,
                "method": "fallback"
            }
            
        except Exception as e:
            logger.error(f"인사말 응답 생성 중 오류 발생: {e}")
            return {
                "answer": f"안녕하세요! {self.company_config.get_company_name()}에 오신 것을 환영합니다!",
                "greeting_type": "general",
                "time_of_day": "unknown",
                "confidence_score": 0.7,
                "generation_time": 0.001,
                "method": "error_fallback"
            }
    
    def _generate_fallback_response(self, greeting_type: str, time_of_day: str) -> str:
        """폴백 응답 생성"""
        import random
        
        # 시간대별 인사말 랜덤 선택
        time_greetings = self.fallback_greetings.get(time_of_day, self.fallback_greetings["afternoon"])
        time_greeting = random.choice(time_greetings)
        
        company_name = self.company_config.get_company_name()
        
        if greeting_type == "help":
            return f"{time_greeting} 도움이 필요하시군요! {company_name}에서 무엇을 도와드릴까요?"
        elif greeting_type == "first_time":
            return f"{time_greeting} 처음 사용하시는군요! {company_name}에 오신 것을 환영합니다!"
        else:
            return f"{time_greeting} {company_name}에 오신 것을 환영합니다!"
    
    def _get_time_greeting(self, time_of_day: str) -> str:
        """시간대별 인사말 반환"""
        import random
        greetings = self.fallback_greetings.get(time_of_day, self.fallback_greetings["afternoon"])
        return random.choice(greetings)
    
    def _get_time_of_day(self, hour: int) -> str:
        """현재 시간대 반환"""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _classify_greeting_type(self, question: str) -> str:
        """인사말 유형 분류 (개선된 버전)"""
        question_lower = question.lower()
        
        # 도움 요청 (명확한 인사 제외, 안내/설명/도움 류는 help로 분류)
        help_keywords = [
            "도움", "도와", "어떻게", "사용법", "사용", "방법", "알려", "가르쳐", "설명",
            "무엇", "무엇을", "무엇인가", "무엇인가요", "뭐야", "뭐예요",
            "기능", "시스템", "할 수", "궁금"
        ]
        if any(keyword in question_lower for keyword in help_keywords):
            return "help"
        
        # 첫 사용 (확장된 키워드)
        first_time_keywords = [
            "처음", "첫", "새로", "처음 사용", "새", "신규", "첫방문", "첫 방문",
            "처음이에요", "처음입니다", "첫번째", "첫 번째", "새로운", "신규 사용자",
            "처음 접속", "처음 들어왔", "처음 왔", "처음 왔어", "처음 왔어요"
        ]
        if any(keyword in question_lower for keyword in first_time_keywords):
            return "first_time"
        
        # 일반 인사 (기본 인사말 패턴)
        greeting_keywords = [
            "안녕", "안녕하세요", "안녕하십니까", "반갑습니다", "반가워", "하이", "hi", "hello"
        ]
        if any(keyword in question_lower for keyword in greeting_keywords):
            return "general"
        
        # 기본값은 도움 요청으로 처리
        return "help"
    
    def get_statistics(self) -> Dict:
        """핸들러 통계 반환"""
        return {
            "handler_type": "인사말 처리",
            "company_name": self.company_config.get_company_name(),
            "fallback_greetings_count": sum(len(greetings) for greetings in self.fallback_greetings.values()),
            "handler_version": "2.2.0"
        }


