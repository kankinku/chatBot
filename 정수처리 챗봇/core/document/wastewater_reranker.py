"""
정수처리 도메인 특화 재순위화 모델

기존 PDFReranker를 정수처리 데이터로 미세조정한 버전
도메인 특화 점수와 일반 재순위화 점수를 결합하여 최적 성능 제공
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import re

from .pdf_processor import TextChunk
from .pdf_reranker import PDFReranker

logger = logging.getLogger(__name__)

@dataclass
class WastewaterRerankConfig:
    """정수처리 재순위화 설정"""
    base_reranker_weight: float = 0.6  # 기본 재순위화 가중치
    domain_weight: float = 0.4  # 도메인 특화 가중치
    process_relevance_weight: float = 0.3  # 공정 관련성 가중치
    measurement_weight: float = 0.2  # 측정값 관련성 가중치
    keyword_weight: float = 0.2  # 키워드 매칭 가중치
    context_weight: float = 0.3  # 문맥 일관성 가중치
    enable_process_boost: bool = True  # 공정 부스팅 활성화
    enable_measurement_boost: bool = True  # 측정값 부스팅 활성화

class WastewaterReranker:
    """정수처리 도메인 특화 재순위화기"""
    
    def __init__(self, config: Optional[WastewaterRerankConfig] = None):
        """재순위화기 초기화"""
        self.config = config or WastewaterRerankConfig()
        
        # 기본 PDF 재순위화기
        self.base_reranker = PDFReranker()
        
        # 정수처리 공정 중요도 가중치
        self.process_importance = {
            "수질관리": 1.0,  # 가장 중요
            "응집": 0.9,
            "여과": 0.9,
            "소독": 0.8,
            "침전": 0.8,
            "응집지": 0.7,
            "취수": 0.6,
            "배수": 0.6,
            "슬러지처리": 0.5,
            "일반": 0.4
        }
        
        # 중요 키워드 패턴과 가중치
        self.keyword_patterns = {
            # 수질 지표 (높은 가중치)
            r'탁도|NTU|turbidity': 1.0,
            r'pH|수소이온농도': 1.0,
            r'잔류염소|residual\s*chlorine': 1.0,
            r'대장균|E\.?\s*coli|coliform': 1.0,
            r'일반세균|total\s*bacteria': 0.9,
            
            # 응집 관련
            r'PAC|poly\s*aluminum\s*chloride': 0.9,
            r'황산알루미늄|alum|aluminum\s*sulfate': 0.9,
            r'염화제이철|ferric\s*chloride': 0.8,
            r'응집제|coagulant': 0.8,
            r'혼화|mixing|rapid\s*mix': 0.7,
            r'플록|floc': 0.7,
            
            # 여과 관련
            r'여과속도|filtration\s*rate': 0.9,
            r'역세척|backwash': 0.8,
            r'모래여과|sand\s*filter': 0.8,
            r'활성탄|activated\s*carbon': 0.7,
            r'여과부하|filter\s*load': 0.7,
            
            # 소독 관련
            r'염소투입|chlorination': 0.9,
            r'CT값|CT\s*value': 0.8,
            r'UV|ultraviolet': 0.7,
            r'오존|ozone': 0.7,
            
            # 측정값 패턴
            r'\d+(?:\.\d+)?\s*(?:mg/L|㎎/L|ppm)': 0.8,
            r'\d+(?:\.\d+)?\s*NTU': 0.9,
            r'\d+(?:\.\d+)?\s*(?:m³/h|㎥/h|m3/h)': 0.7,
            r'\d+(?:\.\d+)?\s*(?:℃|°C)': 0.6
        }
        
        # 부정적 키워드 (점수 감소)
        self.negative_patterns = {
            r'폐수|wastewater|sewage': -0.3,  # 정수처리가 아닌 폐수처리
            r'하수|sewerage': -0.3,
            r'산업폐수|industrial\s*wastewater': -0.2
        }
        
        logger.info(f"정수처리 재순위화기 초기화 (도메인 가중치: {self.config.domain_weight:.1f})")
    
    def rerank(self, query: str, chunks: List[TextChunk], 
               top_k: int = 10) -> List[Tuple[TextChunk, float]]:
        """정수처리 도메인 특화 재순위화"""
        if not chunks:
            return []
        
        logger.info(f"정수처리 재순위화 시작: {len(chunks)}개 청크 → 상위 {top_k}개")
        
        try:
            # 1. 기본 재순위화 점수 (크로스엔코더)
            base_scores = self._get_base_rerank_scores(query, chunks)
            
            # 2. 도메인 특화 점수 계산
            domain_scores = self._calculate_domain_scores(query, chunks)
            
            # 3. 최종 점수 결합
            final_scores = []
            for i, chunk in enumerate(chunks):
                base_score = base_scores[i] if i < len(base_scores) else 0.0
                domain_score = domain_scores[i] if i < len(domain_scores) else 0.0
                
                # 가중 평균
                final_score = (
                    self.config.base_reranker_weight * base_score +
                    self.config.domain_weight * domain_score
                )
                
                final_scores.append((chunk, final_score))
            
            # 4. 점수 기준 정렬
            ranked_results = sorted(final_scores, key=lambda x: x[1], reverse=True)
            
            # 5. 상위 K개 반환
            top_results = ranked_results[:top_k]
            
            logger.info(f"재순위화 완료: 평균 점수 {np.mean([score for _, score in top_results]):.3f}")
            
            return top_results
            
        except Exception as e:
            logger.error(f"재순위화 실패: {e}")
            # 실패 시 원본 순서 유지
            return [(chunk, 0.5) for chunk in chunks[:top_k]]
    
    def _get_base_rerank_scores(self, query: str, chunks: List[TextChunk]) -> List[float]:
        """기본 재순위화 점수 획득"""
        try:
            # PDFReranker 사용
            reranked_results = self.base_reranker.rerank(query, chunks)
            
            # 점수 추출 (순서 유지)
            chunk_to_score = {id(result[0]): result[1] for result in reranked_results}
            scores = [chunk_to_score.get(id(chunk), 0.0) for chunk in chunks]
            
            return scores
            
        except Exception as e:
            logger.warning(f"기본 재순위화 실패, 기본값 사용: {e}")
            return [0.5] * len(chunks)
    
    def _calculate_domain_scores(self, query: str, chunks: List[TextChunk]) -> List[float]:
        """도메인 특화 점수 계산"""
        scores = []
        
        for chunk in chunks:
            # 개별 점수 구성요소 계산
            process_score = self._calculate_process_relevance(query, chunk)
            measurement_score = self._calculate_measurement_relevance(query, chunk)
            keyword_score = self._calculate_keyword_matching(query, chunk)
            context_score = self._calculate_context_coherence(query, chunk)
            
            # 가중 합산
            domain_score = (
                self.config.process_relevance_weight * process_score +
                self.config.measurement_weight * measurement_score +
                self.config.keyword_weight * keyword_score +
                self.config.context_weight * context_score
            )
            
            scores.append(domain_score)
        
        return scores
    
    def _calculate_process_relevance(self, query: str, chunk: TextChunk) -> float:
        """공정 관련성 점수"""
        score = 0.0
        
        # 청크의 공정 유형 확인
        process_type = chunk.metadata.get('process_type', '일반')
        process_importance = self.process_importance.get(process_type, 0.4)
        
        # 쿼리와 청크 내용에서 공정 키워드 매칭
        query_lower = query.lower()
        content_lower = chunk.content.lower()
        
        # 공정 키워드 매칭 점수
        process_keywords = {
            "응집": ["응집", "coagulation", "PAC", "황산알루미늄", "혼화"],
            "침전": ["침전", "sedimentation", "침전지", "슬러지"],
            "여과": ["여과", "filtration", "filter", "모래", "활성탄", "역세척"],
            "소독": ["소독", "disinfection", "염소", "chlorine", "UV", "오존"],
            "수질관리": ["수질", "water quality", "탁도", "pH", "대장균"]
        }
        
        matching_processes = []
        for process, keywords in process_keywords.items():
            query_match = any(keyword in query_lower for keyword in keywords)
            content_match = any(keyword in content_lower for keyword in keywords)
            
            if query_match and content_match:
                matching_processes.append(process)
                score += self.process_importance.get(process, 0.5)
        
        # 공정 부스팅
        if self.config.enable_process_boost and matching_processes:
            score *= 1.2  # 20% 부스트
        
        # 공정 중요도 반영
        score *= process_importance
        
        return min(score, 1.0)
    
    def _calculate_measurement_relevance(self, query: str, chunk: TextChunk) -> float:
        """측정값 관련성 점수"""
        score = 0.0
        
        # 청크에 측정값이 있는지 확인
        has_measurements = False
        if chunk.metadata:
            if isinstance(chunk.metadata, dict):
                has_measurements = bool(chunk.metadata.get('measurements'))
            elif isinstance(chunk.metadata, list):
                has_measurements = any('measurements' in item for item in chunk.metadata if isinstance(item, dict))
        if has_measurements:
            score += 0.3
        
        # 쿼리와 청크 내용에서 측정값 패턴 매칭
        text = query + " " + chunk.content
        
        for pattern, weight in self.keyword_patterns.items():
            if 'mg/L' in pattern or 'NTU' in pattern or r'\d+' in pattern:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches * weight * 0.1
        
        # 측정값 부스팅
        if self.config.enable_measurement_boost and has_measurements:
            score *= 1.15  # 15% 부스트
        
        return min(score, 1.0)
    
    def _calculate_keyword_matching(self, query: str, chunk: TextChunk) -> float:
        """키워드 매칭 점수"""
        score = 0.0
        text = query + " " + chunk.content
        
        # 긍정적 키워드 매칭
        for pattern, weight in self.keyword_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * weight * 0.1
        
        # 부정적 키워드 페널티
        for pattern, penalty in self.negative_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * penalty
        
        # 기본 키워드 매칭 (쿼리 단어들)
        query_words = set(query.lower().split())
        content_words = set(chunk.content.lower().split())
        
        if query_words:
            overlap = len(query_words.intersection(content_words))
            overlap_ratio = overlap / len(query_words)
            score += overlap_ratio * 0.3
        
        return max(0.0, min(score, 1.0))
    
    def _calculate_context_coherence(self, query: str, chunk: TextChunk) -> float:
        """문맥 일관성 점수"""
        score = 0.0
        
        # 청크 길이 점수 (너무 짧거나 길면 감점)
        content_length = len(chunk.content)
        if 100 <= content_length <= 500:
            score += 0.3
        elif 50 <= content_length <= 800:
            score += 0.2
        else:
            score += 0.1
        
        # 문장 완성도 (문장 부호로 판단)
        sentences = re.split(r'[.!?]', chunk.content)
        complete_sentences = len([s for s in sentences if len(s.strip()) > 10])
        if complete_sentences >= 2:
            score += 0.2
        elif complete_sentences >= 1:
            score += 0.1
        
        # 정수처리 맥락 일관성
        context_keywords = ["정수", "수처리", "정수장", "상수도", "음용수", "먹는물"]
        context_matches = sum(1 for keyword in context_keywords if keyword in chunk.content)
        score += min(context_matches * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def get_rerank_explanation(self, query: str, chunk: TextChunk) -> Dict[str, Any]:
        """재순위화 점수 설명"""
        process_score = self._calculate_process_relevance(query, chunk)
        measurement_score = self._calculate_measurement_relevance(query, chunk)
        keyword_score = self._calculate_keyword_matching(query, chunk)
        context_score = self._calculate_context_coherence(query, chunk)
        
        domain_score = (
            self.config.process_relevance_weight * process_score +
            self.config.measurement_weight * measurement_score +
            self.config.keyword_weight * keyword_score +
            self.config.context_weight * context_score
        )
        
        return {
            'chunk_id': chunk.chunk_id,
            'process_type': chunk.metadata.get('process_type', '일반'),
            'scores': {
                'process_relevance': process_score,
                'measurement_relevance': measurement_score,
                'keyword_matching': keyword_score,
                'context_coherence': context_score,
                'domain_total': domain_score
            },
            'features': {
                'has_measurements': bool(chunk.metadata.get('measurements')),
                'content_length': len(chunk.content),
                'process_importance': self.process_importance.get(
                    chunk.metadata.get('process_type', '일반'), 0.4
                )
            }
        }
    
    def batch_rerank(self, queries: List[str], chunks_list: List[List[TextChunk]], 
                    top_k: int = 10) -> List[List[Tuple[TextChunk, float]]]:
        """여러 쿼리 일괄 재순위화"""
        results = []
        
        for i, (query, chunks) in enumerate(zip(queries, chunks_list)):
            try:
                reranked = self.rerank(query, chunks, top_k)
                results.append(reranked)
                logger.debug(f"배치 재순위화 {i+1}/{len(queries)} 완료")
            except Exception as e:
                logger.error(f"배치 재순위화 {i+1} 실패: {e}")
                results.append([(chunk, 0.5) for chunk in chunks[:top_k]])
        
        return results
    
    def get_reranker_stats(self) -> Dict[str, Any]:
        """재순위화기 통계 정보"""
        return {
            'config': {
                'base_reranker_weight': self.config.base_reranker_weight,
                'domain_weight': self.config.domain_weight,
                'process_relevance_weight': self.config.process_relevance_weight,
                'measurement_weight': self.config.measurement_weight
            },
            'process_importance': self.process_importance,
            'keyword_patterns_count': len(self.keyword_patterns),
            'negative_patterns_count': len(self.negative_patterns)
        }

# 편의 함수들
def create_wastewater_reranker(domain_weight: float = 0.4) -> WastewaterReranker:
    """정수처리 재순위화기 생성"""
    config = WastewaterRerankConfig(
        domain_weight=domain_weight,
        enable_process_boost=True,
        enable_measurement_boost=True
    )
    return WastewaterReranker(config)

def create_balanced_reranker() -> WastewaterReranker:
    """균형 잡힌 재순위화기 (기본 + 도메인 균형)"""
    config = WastewaterRerankConfig(
        base_reranker_weight=0.5,
        domain_weight=0.5,
        process_relevance_weight=0.3,
        measurement_weight=0.3,
        keyword_weight=0.2,
        context_weight=0.2
    )
    return WastewaterReranker(config)

def create_domain_focused_reranker() -> WastewaterReranker:
    """도메인 집중 재순위화기 (도메인 특화 강조)"""
    config = WastewaterRerankConfig(
        base_reranker_weight=0.3,
        domain_weight=0.7,
        process_relevance_weight=0.4,
        measurement_weight=0.3,
        keyword_weight=0.2,
        context_weight=0.1
    )
    return WastewaterReranker(config)
