"""
Prometheus Metrics

RAG 시스템의 실시간 성능 메트릭 수집.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from modules.core.logger import get_logger

logger = get_logger(__name__)


class RAGMetrics:
    """
    RAG 시스템 메트릭
    
    Prometheus를 통한 실시간 모니터링.
    Grafana 대시보드로 시각화 가능.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: 메트릭 수집 활성화 여부
        """
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        
        if not self.enabled:
            if not PROMETHEUS_AVAILABLE:
                logger.warning(
                    "Prometheus client not available. "
                    "Install: pip install prometheus-client"
                )
            return
        
        # 요청 카운터
        self.requests_total = Counter(
            'rag_requests_total',
            'Total number of requests',
            ['question_type', 'status']  # 레이블
        )
        
        # 응답 시간 히스토그램
        self.response_time = Histogram(
            'rag_response_seconds',
            'Response time in seconds',
            ['stage'],  # retrieval, generation, etc
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # 신뢰도 게이지
        self.confidence = Gauge(
            'rag_answer_confidence',
            'Answer confidence score',
            ['question_type']
        )
        
        # LLM 타임아웃 카운터
        self.llm_timeouts = Counter(
            'rag_llm_timeouts_total',
            'Number of LLM timeouts'
        )
        
        # 캐시 히트율
        self.cache_hits = Counter(
            'rag_cache_hits_total',
            'Cache hit count',
            ['cache_type']  # embedding, analysis, etc
        )
        
        self.cache_misses = Counter(
            'rag_cache_misses_total',
            'Cache miss count',
            ['cache_type']
        )
        
        # 컨텍스트 수
        self.contexts_used = Histogram(
            'rag_contexts_used',
            'Number of contexts used',
            buckets=[1, 2, 3, 4, 5, 6, 8, 10]
        )
        
        # 검색 결과 수
        self.search_results = Histogram(
            'rag_search_results',
            'Number of search results',
            ['search_type'],  # vector, bm25, hybrid
            buckets=[10, 20, 30, 40, 50, 75, 100]
        )
        
        # 시스템 정보
        self.system_info = Info(
            'rag_system',
            'RAG system information'
        )
        
        logger.info("RAG metrics initialized")
    
    @contextmanager
    def track_request(self, question_type: str = "general"):
        """
        요청 추적 컨텍스트 매니저
        
        Example:
            with metrics.track_request("numeric"):
                answer = pipeline.ask("pH는?")
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception as e:
            status = "error"
            raise
        finally:
            elapsed = time.time() - start_time
            
            # 메트릭 기록
            self.requests_total.labels(
                question_type=question_type,
                status=status
            ).inc()
            
            self.response_time.labels(
                stage="total"
            ).observe(elapsed)
    
    @contextmanager
    def track_stage(self, stage_name: str):
        """
        파이프라인 단계 추적
        
        Example:
            with metrics.track_stage("retrieval"):
                spans = retriever.search(query)
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.response_time.labels(stage=stage_name).observe(elapsed)
    
    def record_confidence(self, confidence: float, question_type: str = "general"):
        """신뢰도 기록"""
        if self.enabled:
            self.confidence.labels(question_type=question_type).set(confidence)
    
    def record_llm_timeout(self):
        """LLM 타임아웃 기록"""
        if self.enabled:
            self.llm_timeouts.inc()
    
    def record_cache_hit(self, cache_type: str = "embedding"):
        """캐시 히트 기록"""
        if self.enabled:
            self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str = "embedding"):
        """캐시 미스 기록"""
        if self.enabled:
            self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_contexts_used(self, count: int):
        """사용된 컨텍스트 수 기록"""
        if self.enabled:
            self.contexts_used.observe(count)
    
    def record_search_results(self, count: int, search_type: str = "hybrid"):
        """검색 결과 수 기록"""
        if self.enabled:
            self.search_results.labels(search_type=search_type).observe(count)
    
    def set_system_info(self, info: Dict[str, str]):
        """시스템 정보 설정"""
        if self.enabled:
            self.system_info.info(info)


# 전역 메트릭 인스턴스
_metrics: Optional[RAGMetrics] = None


def get_metrics(enabled: bool = True) -> RAGMetrics:
    """전역 메트릭 인스턴스 가져오기"""
    global _metrics
    if _metrics is None:
        _metrics = RAGMetrics(enabled=enabled)
    return _metrics


def reset_metrics():
    """메트릭 초기화 (테스트용)"""
    global _metrics
    _metrics = None

