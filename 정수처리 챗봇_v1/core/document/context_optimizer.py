"""
컨텍스트 최적화 모듈

검색 결과에서 상위 2-3개의 가장 관련성 높은 청크만 선별하여
답변 생성 시 노이즈를 줄이고 정확도를 향상시킵니다.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from .text_chunk import TextChunk
from .vector_store import HybridVectorStore

logger = logging.getLogger(__name__)

@dataclass
class ContextOptimizationConfig:
    """컨텍스트 최적화 설정"""
    max_context_chunks: int = 3  # 최대 컨텍스트 청크 수 (2-3개)
    min_relevance_score: float = 0.3  # 최소 관련성 점수
    diversity_threshold: float = 0.7  # 다양성 임계값 (중복 제거용)
    process_weight: float = 0.3  # 공정 매칭 가중치
    keyword_weight: float = 0.2  # 키워드 매칭 가중치
    semantic_weight: float = 0.5  # 의미적 유사도 가중치

class ContextOptimizer:
    """컨텍스트 최적화 클래스"""
    
    def __init__(self, config: Optional[ContextOptimizationConfig] = None):
        """최적화기 초기화"""
        self.config = config or ContextOptimizationConfig()
        
        # 정수처리 도메인 특화 키워드
        self.domain_keywords = {
            'quantitative': [
                'mae', 'mse', 'rmse', 'r²', 'r2', '정확도', '성능', '지표',
                '주입률', '농도', '속도', '압력', '온도', '유량', '개도율', '효율'
            ],
            'processes': [
                '착수', '약품', '혼화', '응집', '침전', '여과', '소독',
                'ems', 'pms', '에너지', '관리', '제어', '모니터링'
            ],
            'models': [
                'n-beats', 'xgb', 'lstm', 'ai', '모델', '알고리즘', '예측', '분석'
            ],
            'equipment': [
                '교반기', '펌프', '밸브', '센서', '모터', '설비', '장비'
            ]
        }
        
        logger.info(f"컨텍스트 최적화기 초기화 완료 (최대 청크: {self.config.max_context_chunks}개)")
    
    def optimize_context(self, 
                        query: str,
                        search_results: List[Tuple[TextChunk, float]],
                        answer_target: Optional[str] = None,
                        target_type: Optional[str] = None) -> List[Tuple[TextChunk, float]]:
        """
        검색 결과에서 최적의 컨텍스트 청크들을 선별
        
        Args:
            query: 사용자 질문
            search_results: 검색 결과 [(청크, 점수), ...]
            answer_target: 답변 목표
            target_type: 목표 유형
            
        Returns:
            최적화된 컨텍스트 청크들
        """
        if not search_results:
            return []
        
        logger.info(f"컨텍스트 최적화 시작: {len(search_results)}개 후보 → {self.config.max_context_chunks}개 선별")
        
        # 1단계: 기본 필터링 (최소 관련성 점수)
        filtered_results = self._filter_by_relevance(search_results)
        
        # 2단계: 다중 기준 점수 계산
        scored_results = self._calculate_multi_criteria_scores(
            query, filtered_results, answer_target, target_type
        )
        
        # 3단계: 다양성 기반 선별 (중복 제거)
        diverse_results = self._select_diverse_chunks(scored_results)
        
        # 4단계: 최종 상위 N개 선별
        final_results = diverse_results[:self.config.max_context_chunks]
        
        logger.info(f"컨텍스트 최적화 완료: {len(final_results)}개 청크 선별")
        
        # 선별 결과 로깅 (디버깅용)
        for i, (chunk, score) in enumerate(final_results, 1):
            process_keywords = chunk.metadata.get('process_keywords', []) if chunk.metadata else []
            logger.debug(f"선별된 청크 {i}: 점수={score:.3f}, 공정={process_keywords}, 내용={chunk.content[:50]}...")
        
        return final_results
    
    def _filter_by_relevance(self, search_results: List[Tuple[TextChunk, float]]) -> List[Tuple[TextChunk, float]]:
        """최소 관련성 점수로 필터링"""
        filtered = [(chunk, score) for chunk, score in search_results 
                   if score >= self.config.min_relevance_score]
        
        logger.debug(f"관련성 필터링: {len(search_results)} → {len(filtered)}개")
        return filtered
    
    def _calculate_multi_criteria_scores(self, 
                                       query: str,
                                       search_results: List[Tuple[TextChunk, float]],
                                       answer_target: Optional[str],
                                       target_type: Optional[str]) -> List[Tuple[TextChunk, float]]:
        """다중 기준 점수 계산"""
        query_lower = query.lower()
        target_lower = answer_target.lower() if answer_target else ""
        
        rescored_results = []
        
        for chunk, original_score in search_results:
            # 1. 의미적 유사도 (원본 점수)
            semantic_score = original_score * self.config.semantic_weight
            
            # 2. 키워드 매칭 점수
            keyword_score = self._calculate_keyword_matching_score(
                chunk, query_lower, target_lower, target_type
            ) * self.config.keyword_weight
            
            # 3. 공정 매칭 점수
            process_score = self._calculate_process_matching_score(
                chunk, query_lower, target_lower
            ) * self.config.process_weight
            
            # 최종 점수 계산
            final_score = semantic_score + keyword_score + process_score
            
            rescored_results.append((chunk, final_score))
        
        # 점수순으로 정렬
        rescored_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"다중 기준 점수 계산 완료: 가중치 적용 (의미:{self.config.semantic_weight}, 키워드:{self.config.keyword_weight}, 공정:{self.config.process_weight})")
        return rescored_results
    
    def _calculate_keyword_matching_score(self, 
                                        chunk: TextChunk,
                                        query_lower: str,
                                        target_lower: str,
                                        target_type: Optional[str]) -> float:
        """키워드 매칭 점수 계산"""
        content_lower = chunk.content.lower()
        score = 0.0
        
        # 쿼리 키워드 매칭
        query_words = set(query_lower.split())
        for word in query_words:
            if len(word) > 2 and word in content_lower:
                score += 0.1
        
        # 답변 목표 키워드 매칭
        if target_lower:
            target_words = set(target_lower.split())
            for word in target_words:
                if len(word) > 2 and word in content_lower:
                    score += 0.15
        
        # 도메인 특화 키워드 매칭
        if target_type:
            domain_keywords = []
            if target_type == "quantitative_value":
                domain_keywords = self.domain_keywords['quantitative']
            elif target_type == "qualitative_definition":
                domain_keywords = self.domain_keywords['processes'] + self.domain_keywords['models']
            elif target_type == "procedural":
                domain_keywords = self.domain_keywords['equipment'] + self.domain_keywords['processes']
            
            for keyword in domain_keywords:
                if keyword in content_lower:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_process_matching_score(self, 
                                        chunk: TextChunk,
                                        query_lower: str,
                                        target_lower: str) -> float:
        """공정 매칭 점수 계산"""
        score = 0.0
        
        # 청크의 공정 키워드 확인
        chunk_process_keywords = []
        if chunk.metadata and 'process_keywords' in chunk.metadata:
            chunk_process_keywords = chunk.metadata['process_keywords']
        
        # 쿼리에서 공정 키워드 추출
        query_process_keywords = []
        for process_name, process_info in {
            'intake': ['착수', '수위', '유입량'],
            'coagulation': ['약품', '응집제', '주입률'],
            'mixing_flocculation': ['혼화', '응집', '교반'],
            'sedimentation': ['침전', '슬러지'],
            'filtration': ['여과', '여과지'],
            'disinfection': ['소독', '염소'],
            'ems': ['ems', '에너지', '전력'],
            'pms': ['pms', '모터', '진단']
        }.items():
            if any(keyword in query_lower for keyword in process_info):
                query_process_keywords.append(process_name)
        
        # 공정 매칭 점수 계산
        if query_process_keywords and chunk_process_keywords:
            matched_processes = set(query_process_keywords) & set(chunk_process_keywords)
            if matched_processes:
                score += len(matched_processes) * 0.3
        
        # 답변 목표의 공정 매칭
        if target_lower:
            for process_name, process_info in {
                'intake': ['착수', '수위', '유입량'],
                'coagulation': ['약품', '응집제', '주입률'],
                'mixing_flocculation': ['혼화', '응집', '교반'],
                'sedimentation': ['침전', '슬러지'],
                'filtration': ['여과', '여과지'],
                'disinfection': ['소독', '염소'],
                'ems': ['ems', '에너지', '전력'],
                'pms': ['pms', '모터', '진단']
            }.items():
                if any(keyword in target_lower for keyword in process_info):
                    if process_name in chunk_process_keywords:
                        score += 0.4
        
        return min(score, 1.0)
    
    def _select_diverse_chunks(self, scored_results: List[Tuple[TextChunk, float]]) -> List[Tuple[TextChunk, float]]:
        """다양성 기반 청크 선별 (중복 제거)"""
        if len(scored_results) <= self.config.max_context_chunks:
            return scored_results
        
        selected = []
        remaining = scored_results.copy()
        
        # 첫 번째는 최고 점수 청크 선택
        if remaining:
            selected.append(remaining.pop(0))
        
        # 나머지는 다양성을 고려하여 선택
        while len(selected) < self.config.max_context_chunks and remaining:
            best_candidate = None
            best_diversity_score = -1
            best_idx = -1
            
            for i, (candidate_chunk, candidate_score) in enumerate(remaining):
                # 이미 선택된 청크들과의 다양성 점수 계산
                diversity_score = self._calculate_diversity_score(candidate_chunk, selected)
                
                # 원본 점수와 다양성 점수를 조합
                combined_score = candidate_score * 0.7 + diversity_score * 0.3
                
                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score
                    best_candidate = (candidate_chunk, candidate_score)
                    best_idx = i
            
            if best_candidate and best_idx >= 0:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        logger.debug(f"다양성 기반 선별 완료: {len(selected)}개 청크 선택")
        return selected
    
    def _calculate_diversity_score(self, 
                                 candidate_chunk: TextChunk,
                                 selected_chunks: List[Tuple[TextChunk, float]]) -> float:
        """후보 청크와 이미 선택된 청크들 간의 다양성 점수 계산"""
        if not selected_chunks:
            return 1.0
        
        candidate_content = candidate_chunk.content.lower()
        candidate_words = set(candidate_content.split())
        
        min_diversity = 1.0
        
        for selected_chunk, _ in selected_chunks:
            selected_content = selected_chunk.content.lower()
            selected_words = set(selected_content.split())
            
            # Jaccard 유사도 계산
            if candidate_words and selected_words:
                intersection = len(candidate_words & selected_words)
                union = len(candidate_words | selected_words)
                similarity = intersection / union if union > 0 else 0.0
                diversity = 1.0 - similarity
                
                min_diversity = min(min_diversity, diversity)
        
        return min_diversity
    
    def get_context_summary(self, optimized_chunks: List[Tuple[TextChunk, float]]) -> Dict[str, Any]:
        """최적화된 컨텍스트의 요약 정보 반환"""
        if not optimized_chunks:
            return {"total_chunks": 0}
        
        # 공정별 분포
        process_distribution = {}
        total_length = 0
        avg_score = 0.0
        
        for chunk, score in optimized_chunks:
            total_length += len(chunk.content)
            avg_score += score
            
            if chunk.metadata and 'process_keywords' in chunk.metadata:
                for process in chunk.metadata['process_keywords']:
                    process_distribution[process] = process_distribution.get(process, 0) + 1
        
        avg_score /= len(optimized_chunks)
        
        return {
            "total_chunks": len(optimized_chunks),
            "avg_score": avg_score,
            "total_length": total_length,
            "avg_length": total_length // len(optimized_chunks),
            "process_distribution": process_distribution,
            "chunk_types": [chunk.metadata.get('chunk_type', 'unknown') 
                           for chunk, _ in optimized_chunks if chunk.metadata and isinstance(chunk.metadata, dict)]
        }
