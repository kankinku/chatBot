#!/usr/bin/env python3
"""
회사 정보 설정 관리

환경변수나 설정 파일에서 회사 정보를 읽어오는 모듈
"""

import os
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class CompanyConfig:
    """회사 정보 설정 관리 클래스"""
    
    def __init__(self):
        """회사 정보 초기화"""
        self.company_info = self._load_company_info()
    
    def _load_company_info(self) -> Dict:
        """회사 정보 로드 (환경변수 우선, 기본값 폴백)"""
        
        # 기본 회사 정보 (환경변수로 설정 가능)
        default_info = {
            "name": os.getenv("COMPANY_NAME", "범용 RAG 시스템"),
            "description": os.getenv("COMPANY_DESCRIPTION", "문서 검색 및 데이터베이스 쿼리 시스템"),
            "tagline": os.getenv("COMPANY_TAGLINE", "지능형 문서 검색 및 데이터 분석"),
            "role": os.getenv("COMPANY_ROLE", "문서 검색, 데이터 분석, AI 답변 생성"),
            "services": [
                "문서 검색",
                "데이터베이스 쿼리", 
                "AI 답변 생성",
                "대화형 인터페이스",
                "문서 분석",
                "데이터 시각화"
            ],
            "features": [
                "실시간 문서 검색",
                "AI 기반 답변",
                "다차원 데이터 분석",
                "시각화 대시보드",
                "자동 리포트 생성",
                "모바일 앱 지원"
            ],
            "location": os.getenv("COMPANY_LOCATION", ""),
            "website": os.getenv("COMPANY_WEBSITE", ""),
            "email": os.getenv("COMPANY_EMAIL", ""),
            "system_type": os.getenv("SYSTEM_TYPE", "범용 RAG 시스템"),
            "target_users": [
                "시스템 관리자",
                "데이터 분석가", 
                "사용자",
                "일반 사용자"
            ],
            "main_functions": [
                "문서 검색",
                "데이터 분석",
                "AI 답변", 
                "시각화",
                "보고서 생성"
            ]
        }
        
        # 환경변수에서 회사 정보 로드 (있는 경우)
        env_mapping = {
            "COMPANY_NAME": "name",
            "COMPANY_DESCRIPTION": "description", 
            "COMPANY_TAGLINE": "tagline",
            "COMPANY_ROLE": "role",
            "COMPANY_LOCATION": "location",
            "COMPANY_WEBSITE": "website",
            "COMPANY_EMAIL": "email",
            "SYSTEM_TYPE": "system_type"
        }
        
        for env_key, config_key in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value:
                if env_key in ["COMPANY_SERVICES", "COMPANY_FEATURES", "TARGET_USERS", "MAIN_FUNCTIONS"]:
                    # 쉼표로 구분된 문자열을 리스트로 변환
                    default_info[config_key] = [item.strip() for item in env_value.split(",")]
                else:
                    default_info[config_key] = env_value
                logger.info(f"환경변수에서 {config_key} 로드: {env_value}")
        
        return default_info
    
    def get_company_info(self) -> Dict:
        """전체 회사 정보 반환"""
        return self.company_info.copy()
    
    def get_company_name(self) -> str:
        """회사명 반환"""
        return self.company_info["name"]
    
    def get_company_description(self) -> str:
        """회사 설명 반환"""
        return self.company_info["description"]
    
    def get_company_tagline(self) -> str:
        """회사 슬로건 반환"""
        return self.company_info["tagline"]
    
    def get_company_role(self) -> str:
        """회사 역할 반환"""
        return self.company_info["role"]
    
    def get_services(self) -> List[str]:
        """주요 서비스 목록 반환"""
        return self.company_info["services"].copy()
    
    def get_features(self) -> List[str]:
        """주요 기능 목록 반환"""
        return self.company_info["features"].copy()
    
    def get_target_users(self) -> List[str]:
        """타겟 사용자 목록 반환"""
        return self.company_info["target_users"].copy()
    
    def get_main_functions(self) -> List[str]:
        """주요 기능 목록 반환"""
        return self.company_info["main_functions"].copy()
    
    def get_greeting_context(self) -> str:
        """인사말 생성을 위한 컨텍스트 반환"""
        return f"""
당신은 {self.company_info['name']}의 AI 어시스턴트입니다.

회사 정보:
- 이름: {self.company_info['name']}
- 설명: {self.company_info['description']}
- 슬로건: {self.company_info['tagline']}
- 역할: {self.company_info['role']}

주요 서비스: {', '.join(self.company_info['services'])}
주요 기능: {', '.join(self.company_info['features'])}

사용자와 친근하고 전문적으로 대화하며, 교통 시스템에 대한 질문에 답변하고 도움을 제공합니다.
시간대에 맞는 인사말을 사용하고, 필요시 회사 정보를 자연스럽게 언급합니다.
"""
    
    def get_help_context(self) -> str:
        """도움말 생성을 위한 컨텍스트 반환"""
        return f"""
{self.company_info['name']}에서 제공하는 서비스:

1. 교통 데이터 분석
   - 실시간 교통량 모니터링
   - 교통 패턴 분석 및 예측
   - 사고 데이터 처리 및 분석

2. 교통 정책 지원
   - 데이터 기반 정책 제안
   - 교통 개선 방안 제시
   - 성과 분석 및 평가

3. 사용자 지원
   - 교통 정보 조회
   - 시스템 사용법 안내
   - 맞춤형 리포트 생성

어떤 도움이 필요하신지 구체적으로 말씀해 주시면 더 자세히 안내해 드리겠습니다.
"""
    
    def reload_config(self):
        """설정 재로드"""
        self.company_info = self._load_company_info()
        logger.info("회사 정보 설정 재로드 완료")


