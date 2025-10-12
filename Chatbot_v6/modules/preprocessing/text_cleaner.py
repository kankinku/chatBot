"""
Text Cleaner - 텍스트 정리

텍스트 정리 및 정규화를 담당합니다 (단일 책임).
"""

from __future__ import annotations

import re
from typing import Optional

from modules.core.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """
    텍스트 정리기
    
    단일 책임: 텍스트 정리 및 기본 정규화만 수행
    """
    
    def __init__(
        self,
        remove_extra_whitespace: bool = True,
        normalize_hyphens: bool = True,
        remove_control_chars: bool = True,
    ):
        """
        Args:
            remove_extra_whitespace: 연속된 공백 제거
            normalize_hyphens: 하이픈 정규화
            remove_control_chars: 제어 문자 제거
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_hyphens = normalize_hyphens
        self.remove_control_chars = remove_control_chars
        
        logger.debug("TextCleaner initialized", config={
            "remove_extra_whitespace": remove_extra_whitespace,
            "normalize_hyphens": normalize_hyphens,
            "remove_control_chars": remove_control_chars,
        })
    
    def clean(self, text: str) -> str:
        """
        텍스트 정리
        
        Args:
            text: 입력 텍스트
            
        Returns:
            정리된 텍스트
        """
        if not text:
            return ""
        
        result = text
        
        # 제어 문자 제거
        if self.remove_control_chars:
            result = self._remove_control_chars(result)
        
        # 하이픈 정규화
        if self.normalize_hyphens:
            result = self._normalize_hyphens(result)
        
        # 연속된 공백 제거
        if self.remove_extra_whitespace:
            result = self._remove_extra_whitespace(result)
        
        return result.strip()
    
    def _remove_control_chars(self, text: str) -> str:
        """제어 문자 제거"""
        # 개행과 탭은 유지, 나머지 제어 문자 제거
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    def _normalize_hyphens(self, text: str) -> str:
        """하이픈 정규화"""
        # 다양한 하이픈 문자를 표준 하이픈으로 통일
        return text.replace('\u2010', '-')\
                   .replace('\u2011', '-')\
                   .replace('\u2012', '-')\
                   .replace('\u2013', '-')\
                   .replace('\u2014', '-')\
                   .replace('\u2015', '-')
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """연속된 공백 제거"""
        # 연속된 공백을 하나로
        text = re.sub(r'[ \t]+', ' ', text)
        # 연속된 개행을 최대 2개로
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    
    def clean_for_hash(self, text: str) -> str:
        """
        해시 생성용 텍스트 정리
        
        중복 제거나 비교를 위한 정규화된 텍스트를 반환합니다.
        """
        # 모든 공백을 단일 공백으로
        text = re.sub(r'[\s\u00A0]+', ' ', text.strip())
        # 하이픈 정규화
        text = self._normalize_hyphens(text)
        return text

