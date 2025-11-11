"""
검색-생성 연결 강화 모듈

상위 2-3개 청크만 사용하여 노이즈 감소 및 답변 품질 향상
정수처리 도메인 특화 필터링 및 재순위화 로직 제공
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .pdf_processor import TextChunk

logger = logging.getLogger(__name__)

class FilteringStrategy(Enum):
    """필터링 전략"""
    TOP_K = "top_k"  # 상위 K개
    THRESHOLD = "threshold"  # 임계값 기반
    ADAPTIVE = "adaptive"  # 적응형
    QUALITY_BASED = "quality_based"  # 품질 기반

@dataclass
class SearchFilterConfig:
    """검색 필터 설정"""
    max_chunks: int = 3  # 최대 청크 수 (2-3개)
    min_chunks: int = 1  # 최소 청크 수
    confidence_threshold: float = 0.25  # 신뢰도 임계값(완화)
    base_confidence: float = 0.65  # confidence 미존재 시 기본 매핑값
    diversity_threshold: float = 0.8  # 다양성 임계값 (중복 제거)
    process_coherence_weight: float = 0.3  # 공정 일관성 가중치
    measurement_bonus: float = 0.1  # 측정값 포함 보너스
    strategy: FilteringStrategy = FilteringStrategy.QUALITY_BASED

@dataclass
class SearchResult:
    """검색 결과"""
    chunk: TextChunk
    confidence: float
    relevance_score: float
    process_type: Optional[str] = None
    has_measurements: bool = False
    diversity_score: float = 1.0
    final_score: float = 0.0

class EnhancedSearchFilter:
    """향상된 검색 필터"""
    
    def __init__(self, config: Optional[SearchFilterConfig] = None):
        """필터 초기화"""
        self.config = config or SearchFilterConfig()
        
        # 정수처리 공정 가중치
        self.process_weights = {
            "취수": 1.0,
            "응집": 1.2,  # 핵심 공정
            "응집지": 1.2,
            "침전": 1.1,
            "여과": 1.2,  # 핵심 공정
            "소독": 1.1,
            "배수": 1.0,
            "슬러지처리": 1.0,
            "수질관리": 1.3,  # 매우 중요
            "일반": 0.9
        }
        
        # 중요 키워드 패턴
        self.important_keywords = [
            r'mg/L|㎎/L|ppm|NTU|탁도|pH|염소',  # 수질 지표
            r'응집제|PAC|황산알루미늄|염화제이철',  # 응집 관련
            r'여과속도|역세척|체류시간|접촉시간',  # 공정 파라미터
            r'수질기준|정수기준|먹는물|음용수',  # 기준 관련
            r'효율|제거율|처리율|성능'  # 성능 지표
        ]
        
        logger.info(f"검색 필터 초기화 (최대 청크: {self.config.max_chunks}개, 전략: {self.config.strategy.value})")
    
    def calculate_relevance_score(self, chunk: TextChunk, query: str) -> float:
        """청크의 관련성 점수 계산"""
        content = chunk.content.lower()
        query_lower = query.lower()
        
        # 기본 점수 (키워드 매칭)
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content)
        base_score = matches / len(query_words) if query_words else 0.0
        
        # 정수처리 공정 가중치 적용
        process_type = chunk.metadata.get('process_type', '일반')
        process_weight = self.process_weights.get(process_type, 1.0)
        
        # 중요 키워드 보너스
        import re
        keyword_bonus = 0.0
        for pattern in self.important_keywords:
            if re.search(pattern, content):
                keyword_bonus += 0.05
        
        # 측정값 포함 보너스
        measurement_bonus = 0.0
        if chunk.metadata.get('measurements'):
            measurement_bonus = self.config.measurement_bonus
        
        final_score = (base_score * process_weight) + keyword_bonus + measurement_bonus
        return min(final_score, 1.0)
    
    def calculate_diversity_score(self, chunk: TextChunk, selected_chunks: List[SearchResult]) -> float:
        """다양성 점수 계산 (중복 제거)"""
        if not selected_chunks:
            return 1.0
        
        content = chunk.content.lower()
        
        # 기존 청크들과의 유사도 계산
        similarities = []
        for selected in selected_chunks:
            selected_content = selected.chunk.content.lower()
            
            # 단순 문자열 유사도 (Jaccard)
            words1 = set(content.split())
            words2 = set(selected_content.split())
            
            if not words1 and not words2:
                similarity = 1.0
            elif not words1 or not words2:
                similarity = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union
            
            similarities.append(similarity)
        
        # 최대 유사도를 기준으로 다양성 점수 계산
        max_similarity = max(similarities) if similarities else 0.0
        diversity_score = 1.0 - max_similarity
        
        return diversity_score
    
    def calculate_process_coherence(self, chunks: List[SearchResult]) -> float:
        """공정 일관성 점수 계산"""
        if not chunks:
            return 0.0
        
        process_types = [chunk.chunk.metadata.get('process_type', '일반') for chunk in chunks]
        
        # 같은 공정 타입의 비율
        from collections import Counter
        process_counts = Counter(process_types)
        max_count = max(process_counts.values()) if process_counts else 0
        coherence_score = max_count / len(chunks)
        
        return coherence_score
    
    def filter_search_results(self, 
                            search_results: List[Dict], 
                            query: str,
                            expected_answer: Optional[str] = None) -> List[SearchResult]:
        """검색 결과 필터링 및 재순위화"""
        if not search_results:
            return []
        
        logger.info(f"검색 결과 필터링 시작: {len(search_results)}개 → 최대 {self.config.max_chunks}개")
        
        # 1. 검색 결과를 SearchResult 객체로 변환
        processed_results = []
        for result in search_results:
            # 청크 추출
            if hasattr(result, 'chunk'):
                chunk = result.chunk
                _conf = getattr(result, 'confidence', None)
                if _conf is None:
                    _conf = getattr(result, 'score', None)
                if _conf is None:
                    _conf = getattr(result, 'rerank_score', None)
                base_confidence = float(_conf) if _conf is not None else self.config.base_confidence
            else:
                # 직접 청크인 경우
                chunk = result
                base_confidence = self.config.base_confidence
            
            # TextChunk로 변환 (필요한 경우)
            if not isinstance(chunk, TextChunk):
                chunk = TextChunk(
                    content=getattr(chunk, 'content', str(chunk)),
                    page_number=getattr(chunk, 'page_number', 1),
                    chunk_id=getattr(chunk, 'chunk_id', f"chunk_{len(processed_results)}"),
                    metadata=getattr(chunk, 'metadata', {})
                )
            
            # 관련성 점수 계산
            relevance_score = self.calculate_relevance_score(chunk, query)
            
            # 측정값 포함 여부 확인
            has_measurements = False
            if chunk.metadata:
                if isinstance(chunk.metadata, dict):
                    has_measurements = bool(chunk.metadata.get('measurements'))
                elif isinstance(chunk.metadata, list):
                    # 리스트인 경우 measurements 키가 있는지 확인
                    has_measurements = any('measurements' in item for item in chunk.metadata if isinstance(item, dict))
            
            search_result = SearchResult(
                chunk=chunk,
                confidence=base_confidence,
                relevance_score=relevance_score,
                process_type=chunk.metadata.get('process_type') if isinstance(chunk.metadata, dict) else None,
                has_measurements=has_measurements
            )
            processed_results.append(search_result)
        
        # 2. 필터링 전략 적용
        if self.config.strategy == FilteringStrategy.TOP_K:
            filtered_results = self._filter_top_k(processed_results)
        elif self.config.strategy == FilteringStrategy.THRESHOLD:
            filtered_results = self._filter_by_threshold(processed_results)
        elif self.config.strategy == FilteringStrategy.ADAPTIVE:
            filtered_results = self._filter_adaptive(processed_results, query)
        else:  # QUALITY_BASED
            filtered_results = self._filter_quality_based(processed_results, query)
        
        logger.info(f"검색 결과 필터링 완료: {len(filtered_results)}개 청크 선택")
        
        # 필터링 통계 로깅
        if filtered_results:
            avg_confidence = sum(r.confidence for r in filtered_results) / len(filtered_results)
            avg_relevance = sum(r.relevance_score for r in filtered_results) / len(filtered_results)
            process_types = [r.process_type for r in filtered_results if r.process_type]
            logger.info(f"평균 신뢰도: {avg_confidence:.3f}, 평균 관련성: {avg_relevance:.3f}, 공정 유형: {process_types}")
        
        return filtered_results
    
    def _filter_top_k(self, results: List[SearchResult]) -> List[SearchResult]:
        """상위 K개 필터링"""
        # 신뢰도 기준 정렬
        sorted_results = sorted(results, key=lambda x: x.confidence, reverse=True)
        return sorted_results[:self.config.max_chunks]
    
    def _filter_by_threshold(self, results: List[SearchResult]) -> List[SearchResult]:
        """임계값 기반 필터링"""
        filtered = [r for r in results if r.confidence >= self.config.confidence_threshold]
        return filtered[:self.config.max_chunks]
    
    def _filter_adaptive(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """적응형 필터링"""
        # 쿼리 복잡도에 따른 청크 수 조정
        query_words = len(query.split())
        if query_words <= 3:
            max_chunks = min(2, self.config.max_chunks)
        elif query_words <= 6:
            max_chunks = min(3, self.config.max_chunks)
        else:
            max_chunks = self.config.max_chunks
        
        # 관련성 점수 기준 정렬
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        return sorted_results[:max_chunks]
    
    def _filter_quality_based(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """품질 기반 필터링 (추천)"""
        selected_results = []
        
        # 관련성 점수로 정렬
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        
        for result in sorted_results:
            if len(selected_results) >= self.config.max_chunks:
                break
            
            # 최소 신뢰도 체크
            if result.confidence < self.config.confidence_threshold:
                continue
            
            # 다양성 점수 계산
            diversity_score = self.calculate_diversity_score(result.chunk, selected_results)
            result.diversity_score = diversity_score
            
            # 다양성 임계값 체크
            if diversity_score < (1.0 - self.config.diversity_threshold):
                continue  # 너무 유사한 내용이면 제외
            
            # 최종 점수 계산
            final_score = (
                result.relevance_score * 0.4 +
                result.confidence * 0.3 +
                diversity_score * 0.2 +
                (self.config.measurement_bonus if result.has_measurements else 0.0) * 0.1
            )
            result.final_score = final_score
            
            selected_results.append(result)
        
        # 최소 청크 수 보장
        if len(selected_results) < self.config.min_chunks and results:
            remaining_needed = self.config.min_chunks - len(selected_results)
            remaining_results = [r for r in sorted_results if r not in selected_results]
            selected_results.extend(remaining_results[:remaining_needed])
        
        # 최종 점수로 재정렬
        selected_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return selected_results
    
    def get_filter_stats(self, results: List[SearchResult]) -> Dict[str, Any]:
        """필터링 통계 정보"""
        if not results:
            return {}
        
        return {
            'total_selected': len(results),
            'avg_confidence': sum(r.confidence for r in results) / len(results),
            'avg_relevance': sum(r.relevance_score for r in results) / len(results),
            'avg_diversity': sum(r.diversity_score for r in results) / len(results),
            'process_distribution': {
                r.process_type: sum(1 for x in results if x.process_type == r.process_type)
                for r in results if r.process_type
            },
            'measurement_chunks': sum(1 for r in results if r.has_measurements),
            'config': {
                'max_chunks': self.config.max_chunks,
                'strategy': self.config.strategy.value,
                'confidence_threshold': self.config.confidence_threshold
            }
        }

# 편의 함수들
def create_enhanced_search_filter(max_chunks: int = 3,
                                 confidence_threshold: float = 0.4) -> EnhancedSearchFilter:
    """향상된 검색 필터 생성"""
    config = SearchFilterConfig(
        max_chunks=max_chunks,
        confidence_threshold=confidence_threshold,
        strategy=FilteringStrategy.QUALITY_BASED
    )
    return EnhancedSearchFilter(config)

def create_conservative_search_filter() -> EnhancedSearchFilter:
    """보수적 검색 필터 (높은 품질, 적은 수)"""
    config = SearchFilterConfig(
        max_chunks=2,
        min_chunks=1,
        confidence_threshold=0.6,
        diversity_threshold=0.9,
        strategy=FilteringStrategy.QUALITY_BASED
    )
    return EnhancedSearchFilter(config)

def create_comprehensive_search_filter() -> EnhancedSearchFilter:
    """포괄적 검색 필터 (낮은 임계값, 많은 수)"""
    config = SearchFilterConfig(
        max_chunks=5,
        min_chunks=2,
        confidence_threshold=0.3,
        diversity_threshold=0.7,
        strategy=FilteringStrategy.ADAPTIVE
    )
    return EnhancedSearchFilter(config)
