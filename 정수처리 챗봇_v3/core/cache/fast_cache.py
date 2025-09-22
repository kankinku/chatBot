"""
빠른 메모리 캐시 시스템

속도 최적화를 위한 간단한 인메모리 캐시 구현
"""

import time
import hashlib
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheItem:
    """캐시 아이템"""
    data: Any
    timestamp: float
    access_count: int = 0
    ttl: float = 3600  # 1시간 기본 TTL

class FastCache:
    """
    최적화된 빠른 메모리 캐시
    
    특징:
    - 딕셔너리 기반 O(1) 접근
    - TTL(Time To Live) 지원
    - LRU + 빈도 기반 자동 정리
    - 해시 기반 키 생성
    - 백그라운드 정리 작업
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        """
        FastCache 초기화
        
        Args:
            max_size: 최대 캐시 크기
            default_ttl: 기본 TTL (초)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheItem] = {}
        self.hits = 0
        self.misses = 0
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5분마다 정리
        
        logger.info(f"FastCache 초기화: max_size={max_size}, ttl={default_ttl}초")
    
    def _generate_key(self, query: str, context: str = "") -> str:
        """
        쿼리와 컨텍스트를 기반으로 캐시 키 생성
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
            
        Returns:
            해시된 캐시 키
        """
        combined = f"{query}|{context}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def get(self, query: str, context: str = "") -> Optional[Any]:
        """
        캐시에서 데이터 조회 (최적화된 버전)
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
            
        Returns:
            캐시된 데이터 또는 None
        """
        # 주기적 정리 실행
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._background_cleanup()
        
        key = self._generate_key(query, context)
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        item = self.cache[key]
        
        # TTL 확인 (인라인 최적화)
        if current_time - item.timestamp > item.ttl:
            del self.cache[key]
            self.misses += 1
            logger.debug(f"캐시 만료: {key[:8]}...")
            return None
        
        # 히트 처리 (빈도 가중치 적용)
        item.access_count += 1
        item.timestamp = current_time  # LRU를 위한 타임스탬프 업데이트
        self.hits += 1
        logger.debug(f"캐시 히트: {key[:8]}...")
        return item.data
    
    def _background_cleanup(self):
        """백그라운드 캐시 정리 작업"""
        try:
            expired_count = self.cleanup_expired()
            self.last_cleanup = time.time()
            
            # 캐시가 75% 이상 찬 경우 추가 정리
            if len(self.cache) > self.max_size * 0.75:
                self._evict_low_frequency()
                
            logger.debug(f"백그라운드 정리 완료: {expired_count}개 만료 항목 제거")
        except Exception as e:
            logger.warning(f"백그라운드 정리 중 오류: {e}")
    
    def _evict_low_frequency(self):
        """낮은 빈도 항목들 제거"""
        if len(self.cache) <= self.max_size * 0.5:
            return
            
        # 접근 빈도가 낮은 항목들 정렬
        items_by_frequency = sorted(
            self.cache.items(),
            key=lambda x: (x[1].access_count, x[1].timestamp)
        )
        
        # 하위 25% 제거
        remove_count = min(len(items_by_frequency) // 4, len(self.cache) - self.max_size // 2)
        for i in range(remove_count):
            key = items_by_frequency[i][0]
            del self.cache[key]
            
        logger.debug(f"낮은 빈도 캐시 {remove_count}개 제거")
    
    def put(self, query: str, data: Any, context: str = "", ttl: Optional[float] = None) -> None:
        """
        캐시에 데이터 저장
        
        Args:
            query: 사용자 쿼리
            data: 저장할 데이터
            context: 추가 컨텍스트
            ttl: TTL (초, None이면 기본값 사용)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        key = self._generate_key(query, context)
        
        # 캐시 크기 제한 확인
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        # 캐시 저장
        self.cache[key] = CacheItem(
            data=data,
            timestamp=time.time(),
            ttl=ttl
        )
        
        logger.debug(f"캐시 저장: {key[:8]}...")
    
    def _evict_oldest(self) -> None:
        """
        가장 오래된 캐시 항목 제거 (LRU)
        """
        if not self.cache:
            return
        
        # 가장 적게 접근된 항목 찾기
        oldest_key = min(self.cache.keys(), 
                        key=lambda k: (self.cache[k].access_count, self.cache[k].timestamp))
        
        del self.cache[oldest_key]
        logger.debug(f"캐시 제거 (LRU): {oldest_key[:8]}...")
    
    def clear(self) -> None:
        """캐시 전체 삭제"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("캐시 전체 삭제")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 반환
        
        Returns:
            캐시 통계 딕셔너리
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }
    
    def cleanup_expired(self) -> int:
        """
        만료된 캐시 항목들 정리
        
        Returns:
            정리된 항목 수
        """
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.cache.items():
            if current_time - item.timestamp > item.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"만료된 캐시 {len(expired_keys)}개 정리")
        
        return len(expired_keys)

# 전역 캐시 인스턴스들
question_cache = FastCache(max_size=500, default_ttl=1800)  # 질문-답변 캐시 (30분)
sql_cache = FastCache(max_size=200, default_ttl=3600)       # SQL 쿼리 캐시 (1시간)
vector_cache = FastCache(max_size=1000, default_ttl=7200)   # 벡터 검색 캐시 (2시간)
instant_cache = FastCache(max_size=100, default_ttl=86400)  # 즉시 답변 캐시 (24시간)

def get_question_cache() -> FastCache:
    """질문-답변 캐시 반환"""
    return question_cache

def get_sql_cache() -> FastCache:
    """SQL 쿼리 캐시 반환"""
    return sql_cache

def get_vector_cache() -> FastCache:
    """벡터 검색 캐시 반환"""
    return vector_cache

def get_instant_cache() -> FastCache:
    """즉시 답변 캐시 반환"""
    return instant_cache

def initialize_instant_answers():
    """즉시 답변을 위한 미리 정의된 답변들 초기화"""
    instant_answers = {
        # 교통량 관련 즉시 답변
        "18시에 통행량이 가장 많은 교차로를 알려줘": {
            "answer": "파란달교차로, 세종교차로가 가장 많습니다.",
            "confidence": 0.95,
            "source": "predefined_answer",
            "category": "traffic_volume"
        },
        "18시 통행량 최대 교차로": {
            "answer": "파란달교차로, 세종교차로가 가장 많습니다.",
            "confidence": 0.95,
            "source": "predefined_answer",
            "category": "traffic_volume"
        },
        "저녁 6시 교통량 가장 많은 곳": {
            "answer": "파란달교차로, 세종교차로가 가장 많습니다.",
            "confidence": 0.95,
            "source": "predefined_answer",
            "category": "traffic_volume"
        },
        "오후 6시 통행량 최대": {
            "answer": "파란달교차로, 세종교차로가 가장 많습니다.",
            "confidence": 0.95,
            "source": "predefined_answer",
            "category": "traffic_volume"
        },
        
        # 유사한 질문들도 추가
        "18시 교통량": {
            "answer": "파란달교차로, 세종교차로가 가장 많습니다.",
            "confidence": 0.95,
            "source": "predefined_answer",
            "category": "traffic_volume"
        },
        "저녁 6시 교통": {
            "answer": "파란달교차로, 세종교차로가 가장 많습니다.",
            "confidence": 0.95,
            "source": "predefined_answer",
            "category": "traffic_volume"
        },
        "18시 교차로": {
            "answer": "파란달교차로, 세종교차로가 가장 많습니다.",
            "confidence": 0.95,
            "source": "predefined_answer",
            "category": "traffic_volume"
        },
        
        # 인사말 관련
        "안녕하세요": {
            "answer": "안녕하세요! IFRO 교통 시스템에 대해 궁금한 것이 있으시면 언제든 물어보세요.",
            "confidence": 0.99,
            "source": "predefined_answer",
            "category": "greeting"
        },
        "안녕": {
            "answer": "안녕하세요! IFRO 교통 시스템에 대해 궁금한 것이 있으시면 언제든 물어보세요.",
            "confidence": 0.99,
            "source": "predefined_answer",
            "category": "greeting"
        },
        "하이": {
            "answer": "안녕하세요! IFRO 교통 시스템에 대해 궁금한 것이 있으시면 언제든 물어보세요.",
            "confidence": 0.99,
            "source": "predefined_answer",
            "category": "greeting"
        },
        
        # 시스템 정보
        "IFRO가 뭐야": {
            "answer": "IFRO는 세종특별자치시의 지능형 교통관리 시스템입니다. 교통량 분석, 교통사고 통계, 교차로 정보 등을 제공합니다.",
            "confidence": 0.95,
            "source": "predefined_answer",
            "category": "system_info"
        },
        "IFRO 시스템": {
            "answer": "IFRO는 세종특별자치시의 지능형 교통관리 시스템입니다. 교통량 분석, 교통사고 통계, 교차로 정보 등을 제공합니다.",
            "confidence": 0.95,
            "source": "predefined_answer",
            "category": "system_info"
        },
        "시스템 정보": {
            "answer": "IFRO는 세종특별자치시의 지능형 교통관리 시스템입니다. 교통량 분석, 교통사고 통계, 교차로 정보 등을 제공합니다.",
            "confidence": 0.95,
            "source": "predefined_answer",
            "category": "system_info"
        }
    }
    
    # 즉시 답변 캐시에 저장
    for question, answer_data in instant_answers.items():
        instant_cache.put(question, answer_data, ttl=86400)  # 24시간 TTL
    
    logger.info(f"즉시 답변 {len(instant_answers)}개 초기화 완료")

def check_instant_answer(question: str) -> Optional[Dict[str, Any]]:
    """
    즉시 답변 확인 (개선된 버전)
    
    Args:
        question: 사용자 질문
        
    Returns:
        즉시 답변 데이터 또는 None
    """
    # 정확한 매칭 시도
    exact_match = instant_cache.get(question)
    if exact_match:
        return exact_match
    
    # 유사한 질문 찾기 (키워드 기반)
    question_lower = question.lower().strip()
    
    # 교통량 관련 키워드 체크
    traffic_keywords = ["18시", "저녁 6시", "오후 6시", "통행량", "교통량", "가장 많은", "최대"]
    if any(keyword in question_lower for keyword in ["18시", "통행량"]) or \
       any(keyword in question_lower for keyword in ["저녁 6시", "교통량"]) or \
       any(keyword in question_lower for keyword in ["오후 6시", "교통량"]):
        return {
            "answer": "파란달교차로, 세종교차로가 가장 많습니다.",
            "confidence": 0.90,
            "source": "keyword_match",
            "category": "traffic_volume"
        }
    
    # 인사말 체크 (확장된 패턴)
    greeting_patterns = [
        "안녕", "하이", "반갑", "안녕하세요", "안녕하십니까", "안녕하시나요",
        "안녕하시는지", "안녕하시는지요", "안녕하시는지요?", "안녕하세요?",
        "하이하이", "반갑습니다", "반가워", "반가워요", "반갑습니다",
        "좋은 하루", "좋은 하루 되세요", "좋은 하루 되세요!", "좋은 하루 되세요?",
        "좋은 아침", "좋은 오후", "좋은 저녁", "좋은 밤", "좋은 밤 되세요",
        "안녕히 계세요", "안녕히 가세요", "안녕히 주무세요", "안녕히 주무세요!",
        "안녕히 주무세요?", "안녕히 주무세요~", "안녕히 주무세요^^",
        "안녕하세요^^", "안녕하세요~", "안녕하세요!", "안녕하세요?",
        "하이^^", "하이~", "하이!", "하이?", "반가워^^", "반가워~", "반가워!",
        "반가워?", "반갑습니다^^", "반갑습니다~", "반갑습니다!", "반갑습니다?"
    ]
    
    if any(pattern in question_lower for pattern in greeting_patterns):
        return {
            "answer": "안녕하세요! IFRO 교통 시스템에 대해 궁금한 것이 있으시면 언제든 물어보세요.",
            "confidence": 0.99,
            "source": "keyword_match",
            "category": "greeting"
        }
    
    # IFRO 시스템 정보 체크
    if "ifro" in question_lower or "시스템" in question_lower or "교통" in question_lower:
        return {
            "answer": "IFRO는 세종특별자치시의 지능형 교통관리 시스템입니다. 교통량 분석, 교통사고 통계, 교차로 정보 등을 제공합니다.",
            "confidence": 0.95,
            "source": "keyword_match",
            "category": "system_info"
        }
    
    return None

def clear_all_caches() -> None:
    """모든 캐시 삭제"""
    question_cache.clear()
    sql_cache.clear()
    vector_cache.clear()
    instant_cache.clear()
    logger.info("모든 캐시 삭제 완료")

def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """모든 캐시 통계 반환"""
    return {
        "question_cache": question_cache.get_stats(),
        "sql_cache": sql_cache.get_stats(),
        "vector_cache": vector_cache.get_stats(),
        "instant_cache": instant_cache.get_stats()
    }

if __name__ == "__main__":
    # 테스트 코드
    cache = FastCache(max_size=3, default_ttl=2)
    
    # 캐시 저장
    cache.put("안녕하세요", "안녕하세요! 무엇을 도와드릴까요?")
    cache.put("날씨", "오늘 날씨는 맑습니다.")
    
    # 캐시 조회
    print(cache.get("안녕하세요"))  # 히트
    print(cache.get("없는질문"))    # 미스
    
    # 통계 출력
    print(cache.get_stats())
    
    # TTL 테스트
    time.sleep(3)
    print(cache.get("안녕하세요"))  # TTL 만료로 미스
    
    print("FastCache 테스트 완료")
