"""
정확도 검증 메커니즘

검색 결과와 생성된 답변의 정확도를 검증하고 개선 방안을 제시합니다.
"""

import re
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np

from .pdf_processor import TextChunk
from .enhanced_vector_search import SearchResult

import logging
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """검증 결과"""
    is_accurate: bool
    confidence_score: float
    accuracy_issues: List[str]
    improvement_suggestions: List[str]
    detailed_analysis: Dict[str, Any]

class AccuracyValidator:
    """정확도 검증기"""
    
    def __init__(self, embedding_model: str = "jhgan/ko-sroberta-multitask"):
        """검증기 초기화"""
        self.embedding_model = SentenceTransformer(embedding_model, cache_folder="./models")
        
        # 정수처리 도메인 특화 검증 규칙
        self.validation_rules = self._build_validation_rules()
        
        logger.info("정확도 검증기 초기화 완료")
    
    def _build_validation_rules(self) -> Dict[str, Dict]:
        """검증 규칙 구축"""
        return {
            'numeric_validation': {
                'patterns': [
                    r'\d+(?:\.\d+)?\s*(?:~|\-|–|to)\s*\d+(?:\.\d+)?',  # 범위 값
                    r'\d+(?:\.\d+)?\s*(?:%|mg/l|mg/L|rpm|m³/h|m3/h)',    # 단위 포함 값
                    r'(?:mae|mse|rmse|r²|r2)\s*[:=]\s*\d+(?:\.\d+)?'     # 성능 지표
                ],
                'required_units': ['%', 'mg/l', 'rpm', 'm³/h', '분', '시간']
            },
            'keyword_validation': {
                'critical_keywords': [
                    'k-means', '군집분석', 'n-beats', 'xgb', 'mae', 'mse', 'r²',
                    '상관계수', '교반기', '회전속도', '응집제', '주입률'
                ],
                'process_keywords': [
                    '착수', '약품', '혼화', '응집', '침전', '여과', '소독'
                ]
            },
            'completeness_validation': {
                'min_length': 10,
                'max_length': 500,
                'required_elements': ['주어', '동사', '목적어']
            }
        }
    
    def validate_search_results(self, 
                              query: str,
                              search_results: List[SearchResult],
                              expected_answer: Optional[str] = None) -> ValidationResult:
        """검색 결과 정확도 검증"""
        logger.info(f"검색 결과 정확도 검증 시작: '{query}'")
        
        accuracy_issues = []
        improvement_suggestions = []
        detailed_analysis = {}
        
        # 1. 검색 결과 품질 검증
        result_quality = self._validate_result_quality(search_results)
        accuracy_issues.extend(result_quality['issues'])
        improvement_suggestions.extend(result_quality['suggestions'])
        detailed_analysis['result_quality'] = result_quality
        
        # 2. 쿼리-결과 관련성 검증
        relevance = self._validate_query_relevance(query, search_results)
        accuracy_issues.extend(relevance['issues'])
        improvement_suggestions.extend(relevance['suggestions'])
        detailed_analysis['relevance'] = relevance
        
        # 3. 도메인 특화 검증
        domain_validation = self._validate_domain_specific(query, search_results)
        accuracy_issues.extend(domain_validation['issues'])
        improvement_suggestions.extend(domain_validation['suggestions'])
        detailed_analysis['domain_validation'] = domain_validation
        
        # 4. 예상 답변과의 일치도 검증 (있는 경우)
        if expected_answer:
            answer_match = self._validate_answer_match(search_results, expected_answer)
            accuracy_issues.extend(answer_match['issues'])
            improvement_suggestions.extend(answer_match['suggestions'])
            detailed_analysis['answer_match'] = answer_match
        
        # 전체 정확도 점수 계산
        confidence_score = self._calculate_overall_confidence(detailed_analysis)
        is_accurate = confidence_score >= 0.7 and len(accuracy_issues) == 0
        
        return ValidationResult(
            is_accurate=is_accurate,
            confidence_score=confidence_score,
            accuracy_issues=accuracy_issues,
            improvement_suggestions=improvement_suggestions,
            detailed_analysis=detailed_analysis
        )
    
    def _validate_result_quality(self, search_results: List[SearchResult]) -> Dict[str, Any]:
        """검색 결과 품질 검증"""
        issues = []
        suggestions = []
        
        # 결과 수 검증
        if len(search_results) == 0:
            issues.append("검색 결과가 없습니다")
            suggestions.append("검색 쿼리를 더 일반적인 용어로 변경하거나 유사도 임계값을 낮춰보세요")
        elif len(search_results) < 3:
            issues.append("검색 결과가 너무 적습니다")
            suggestions.append("검색 범위를 확대하거나 쿼리를 확장해보세요")
        
        # 신뢰도 검증
        avg_confidence = np.mean([r.confidence for r in search_results]) if search_results else 0.0
        if avg_confidence < 0.3:
            issues.append("검색 결과의 평균 신뢰도가 낮습니다")
            suggestions.append("임베딩 모델을 변경하거나 쿼리 확장을 활성화해보세요")
        
        # 점수 분포 검증
        if len(search_results) > 1:
            scores = [r.score for r in search_results]
            score_variance = np.var(scores)
            if score_variance < 0.01:
                issues.append("검색 결과 점수의 분산이 너무 작습니다")
                suggestions.append("검색 알고리즘의 다양성을 높여보세요")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'metrics': {
                'result_count': len(search_results),
                'avg_confidence': avg_confidence,
                'score_variance': np.var([r.score for r in search_results]) if search_results else 0.0
            }
        }
    
    def _validate_query_relevance(self, query: str, search_results: List[SearchResult]) -> Dict[str, Any]:
        """쿼리-결과 관련성 검증"""
        issues = []
        suggestions = []
        
        if not search_results:
            return {'issues': issues, 'suggestions': suggestions, 'metrics': {}}
        
        # 쿼리 키워드 추출
        query_keywords = set(re.findall(r'\w+', query.lower()))
        
        # 각 결과의 관련성 검증
        relevance_scores = []
        for result in search_results:
            chunk_keywords = set(re.findall(r'\w+', result.chunk.content.lower()))
            common_keywords = query_keywords & chunk_keywords
            relevance = len(common_keywords) / len(query_keywords) if query_keywords else 0.0
            relevance_scores.append(relevance)
        
        avg_relevance = np.mean(relevance_scores)
        if avg_relevance < 0.3:
            issues.append("검색 결과와 쿼리의 관련성이 낮습니다")
            suggestions.append("쿼리를 더 구체적으로 작성하거나 도메인 특화 키워드를 사용해보세요")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'metrics': {
                'avg_relevance': avg_relevance,
                'min_relevance': min(relevance_scores),
                'max_relevance': max(relevance_scores)
            }
        }
    
    def _validate_domain_specific(self, query: str, search_results: List[SearchResult]) -> Dict[str, Any]:
        """도메인 특화 검증"""
        issues = []
        suggestions = []
        
        # 정수처리 도메인 키워드 검증
        domain_keywords = self.validation_rules['keyword_validation']['critical_keywords']
        query_lower = query.lower()
        
        # 쿼리에 중요한 도메인 키워드가 있는지 확인
        found_domain_keywords = [kw for kw in domain_keywords if kw in query_lower]
        
        if found_domain_keywords:
            # 검색 결과에서 해당 키워드가 포함된 결과가 있는지 확인
            keyword_coverage = 0
            for result in search_results:
                content_lower = result.chunk.content.lower()
                if any(kw in content_lower for kw in found_domain_keywords):
                    keyword_coverage += 1
            
            coverage_rate = keyword_coverage / len(search_results) if search_results else 0.0
            if coverage_rate < 0.5:
                issues.append("도메인 특화 키워드가 검색 결과에 충분히 반영되지 않았습니다")
                suggestions.append("도메인 특화 검색을 활성화하거나 키워드 가중치를 조정해보세요")
        
        # 숫자 값 검증
        numeric_validation = self._validate_numeric_values(query, search_results)
        issues.extend(numeric_validation['issues'])
        suggestions.extend(numeric_validation['suggestions'])
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'metrics': {
                'domain_keywords_found': len(found_domain_keywords),
                'keyword_coverage_rate': coverage_rate if 'coverage_rate' in locals() else 0.0,
                'numeric_validation': numeric_validation['metrics']
            }
        }
    
    def _validate_numeric_values(self, query: str, search_results: List[SearchResult]) -> Dict[str, Any]:
        """숫자 값 검증"""
        issues = []
        suggestions = []
        
        # 쿼리에서 숫자 패턴 추출
        query_numeric_patterns = []
        for pattern in self.validation_rules['numeric_validation']['patterns']:
            matches = re.findall(pattern, query, re.IGNORECASE)
            query_numeric_patterns.extend(matches)
        
        found_patterns = 0
        if query_numeric_patterns:
            # 검색 결과에서 해당 숫자 패턴이 있는지 확인
            for result in search_results:
                content = result.chunk.content
                for pattern in self.validation_rules['numeric_validation']['patterns']:
                    if re.search(pattern, content, re.IGNORECASE):
                        found_patterns += 1
                        break
            
            if found_patterns == 0:
                issues.append("쿼리의 숫자 값이 검색 결과에 반영되지 않았습니다")
                suggestions.append("숫자 값 검색을 강화하거나 단위 정규화를 적용해보세요")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'metrics': {
                'query_numeric_patterns': len(query_numeric_patterns),
                'found_patterns': found_patterns
            }
        }
    
    def _validate_answer_match(self, search_results: List[SearchResult], expected_answer: str) -> Dict[str, Any]:
        """예상 답변과의 일치도 검증"""
        issues = []
        suggestions = []
        
        if not search_results:
            return {'issues': issues, 'suggestions': suggestions, 'metrics': {}}
        
        # 검색 결과를 하나의 텍스트로 결합
        combined_content = " ".join([r.chunk.content for r in search_results])
        
        # 예상 답변과의 의미적 유사도 계산
        try:
            expected_embedding = self.embedding_model.encode([expected_answer])
            combined_embedding = self.embedding_model.encode([combined_content])
            
            similarity = np.dot(expected_embedding[0], combined_embedding[0]) / (
                np.linalg.norm(expected_embedding[0]) * np.linalg.norm(combined_embedding[0])
            )
            
            if similarity < 0.5:
                issues.append("검색 결과가 예상 답변과 의미적으로 일치하지 않습니다")
                suggestions.append("검색 알고리즘을 조정하거나 더 많은 관련 문서를 포함해보세요")
            
        except Exception as e:
            logger.warning(f"의미적 유사도 계산 실패: {e}")
            similarity = 0.0
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'metrics': {
                'semantic_similarity': similarity
            }
        }
    
    def _calculate_overall_confidence(self, detailed_analysis: Dict[str, Any]) -> float:
        """전체 신뢰도 계산"""
        confidence_factors = []
        
        # 결과 품질 기반 신뢰도
        if 'result_quality' in detailed_analysis:
            quality_metrics = detailed_analysis['result_quality']['metrics']
            if quality_metrics.get('result_count', 0) > 0:
                confidence_factors.append(min(quality_metrics.get('avg_confidence', 0.0), 1.0))
        
        # 관련성 기반 신뢰도
        if 'relevance' in detailed_analysis:
            relevance_metrics = detailed_analysis['relevance']['metrics']
            confidence_factors.append(min(relevance_metrics.get('avg_relevance', 0.0), 1.0))
        
        # 도메인 검증 기반 신뢰도
        if 'domain_validation' in detailed_analysis:
            domain_metrics = detailed_analysis['domain_validation']['metrics']
            keyword_coverage = domain_metrics.get('keyword_coverage_rate', 0.0)
            confidence_factors.append(keyword_coverage)
        
        # 답변 일치도 기반 신뢰도
        if 'answer_match' in detailed_analysis:
            answer_metrics = detailed_analysis['answer_match']['metrics']
            confidence_factors.append(min(answer_metrics.get('semantic_similarity', 0.0), 1.0))
        
        # 가중 평균으로 전체 신뢰도 계산
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.0
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """검증 통계 반환"""
        return {
            'validation_rules': {
                'numeric_patterns': len(self.validation_rules['numeric_validation']['patterns']),
                'critical_keywords': len(self.validation_rules['keyword_validation']['critical_keywords']),
                'process_keywords': len(self.validation_rules['keyword_validation']['process_keywords'])
            },
            'embedding_model': self.embedding_model.get_sentence_embedding_dimension()
        }
