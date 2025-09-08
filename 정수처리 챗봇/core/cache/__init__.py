"""
캐시 관련 모듈

이 패키지는 빠른 응답을 위한 캐싱 시스템을 담당합니다.
"""

from .fast_cache import FastCache, get_question_cache, get_sql_cache

__all__ = [
    'FastCache',
    'get_question_cache',
    'get_sql_cache'
]
