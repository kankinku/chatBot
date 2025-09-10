"""
향상된 벡터 검색 모듈

기존 벡터 검색 시스템을 더욱 강화하여 정확도를 높입니다:
- 다중 임베딩 모델 앙상블
- 도메인 특화 쿼리 확장
- 컨텍스트 인식 검색
- 동적 가중치 조정
- 실시간 성능 모니터링
"""

import numpy as np
import time
import re
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .pdf_processor import TextChunk
from .vector_store import HybridVectorStore

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSearchConfig:
    """향상된 검색 설정"""
    # 다중 모델 설정
    primary_model: str = "jhgan/ko-sroberta-multitask"
    secondary_models: List[str] = None
    ensemble_weights: List[float] = None
    
    # 검색 파라미터
    max_candidates: int = 300
    final_top_k: int = 10
    similarity_threshold: float = 0.01
    
    # 쿼리 확장 설정
    enable_query_expansion: bool = True
    enable_domain_expansion: bool = True
    enable_synonym_expansion: bool = True
    
    # 동적 가중치 설정
    enable_dynamic_weights: bool = True
    context_aware_search: bool = True
    
    # 성능 모니터링
    enable_performance_monitoring: bool = True

@dataclass
class SearchResult:
    """검색 결과"""
    chunk: TextChunk
    score: float
    model_scores: Dict[str, float]
    query_expansions: List[str]
    context_relevance: float
    confidence: float

class EnhancedVectorSearcher:
    """향상된 벡터 검색기"""
    
    def __init__(self, 
                 vector_store: HybridVectorStore,
                 config: Optional[EnhancedSearchConfig] = None):
        """검색기 초기화"""
        self.config = config or EnhancedSearchConfig()
        self.vector_store = vector_store
        
        # 다중 임베딩 모델 로드
        self.embedding_models = {}
        self._load_embedding_models()
        
        # 도메인 특화 사전
        self.domain_dictionary = self._build_domain_dictionary()
        
        # 성능 통계
        self.performance_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'avg_confidence': 0.0,
            'query_expansion_rate': 0.0
        }
        
        logger.info(f"향상된 벡터 검색기 초기화 완료 (모델: {len(self.embedding_models)}개)")
    
    def _load_embedding_models(self):
        """다중 임베딩 모델 로드"""
        models_to_load = [self.config.primary_model]
        if self.config.secondary_models:
            models_to_load.extend(self.config.secondary_models)
        
        for model_name in models_to_load:
            try:
                self.embedding_models[model_name] = SentenceTransformer(
                    model_name,
                    cache_folder="./models"
                )
                logger.info(f"임베딩 모델 로드: {model_name}")
            except Exception as e:
                logger.warning(f"모델 로드 실패 {model_name}: {e}")
        
        # 앙상블 가중치 설정
        if not self.config.ensemble_weights:
            self.config.ensemble_weights = [1.0] + [0.3] * (len(self.embedding_models) - 1)
    
    def _build_domain_dictionary(self) -> Dict[str, Dict[str, List[str]]]:
        """정수처리 도메인 사전 구축"""
        return {
            'processes': {
                '착수': ['착수공정', '착수 공정', '수위제어', '유입량제어', '정수지관리'],
                '약품': ['약품공정', '약품 공정', '응집제주입', '응집제 주입', '화학처리'],
                '혼화응집': ['혼화응집공정', '혼화 응집 공정', '교반', '응집', '혼화'],
                '침전': ['침전공정', '침전 공정', '슬러지처리', '슬러지 처리', '침전지'],
                '여과': ['여과공정', '여과 공정', '여과지', '여과지관리', '역세척'],
                '소독': ['소독공정', '소독 공정', '염소주입', '염소 주입', '잔류염소']
            },
            'systems': {
                'ems': ['ems시스템', 'ems 시스템', '에너지관리', '전력관리', '펌프제어'],
                'pms': ['pms시스템', 'pms 시스템', '펌프관리', '모터진단', '예방정비'],
                'dashboard': ['대시보드', '모니터링', '운영화면', '관리화면', '제어화면']
            },
            'metrics': {
                '성능지표': ['성능 지표', '정확도', '오차', 'mae', 'mse', 'rmse', 'r²'],
                '운영지표': ['운영 지표', '효율', '효율성', '생산성', '품질'],
                '측정값': ['측정 값', '센서값', '센서 값', '실시간값', '실시간 값']
            },
            'equipment': {
                '교반기': ['교반기', '교반장치', '교반설비', '회전속도', 'rpm'],
                '펌프': ['펌프', '송수펌프', '송수 펌프', '펌프세부현황', '펌프 세부 현황'],
                '밸브': ['밸브', '개도율', '밸브개도율', '밸브 개도율', '유량제어'],
                '센서': ['센서', '측정기', '계측기', '모니터링장치']
            }
        }
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               context: Optional[str] = None) -> List[SearchResult]:
        """향상된 벡터 검색 수행"""
        start_time = time.time()
        
        logger.info(f"향상된 벡터 검색 시작: '{query}'")
        
        try:
            # 1. 쿼리 확장
            expanded_queries = self._expand_query(query, context)
            
            # 2. 다중 모델 검색
            all_results = []
            for model_name, model in self.embedding_models.items():
                model_results = self._search_with_model(
                    model, model_name, expanded_queries, self.config.max_candidates
                )
                all_results.extend(model_results)
            
            # 3. 앙상블 점수 계산
            ensemble_results = self._calculate_ensemble_scores(all_results)
            
            # 4. 컨텍스트 인식 재랭킹
            if self.config.context_aware_search:
                ensemble_results = self._context_aware_reranking(
                    ensemble_results, query, context
                )
            
            # 5. 결과 필터링 및 정렬
            final_results = self._filter_and_rank_results(ensemble_results, top_k)
            
            # 6. 성능 통계 업데이트
            self._update_performance_stats(start_time, final_results, expanded_queries)
            
            search_time = time.time() - start_time
            logger.info(f"향상된 벡터 검색 완료: {search_time:.3f}초, {len(final_results)}개 결과")
            
            return final_results
            
        except Exception as e:
            logger.error(f"향상된 벡터 검색 실패: {e}")
            return []
    
    def _expand_query(self, query: str, context: Optional[str] = None) -> List[str]:
        """쿼리 확장"""
        expanded_queries = [query]
        
        if not self.config.enable_query_expansion:
            return expanded_queries
        
        # 1. 도메인 특화 확장
        if self.config.enable_domain_expansion:
            domain_expansions = self._domain_specific_expansion(query)
            expanded_queries.extend(domain_expansions)
        
        # 2. 동의어 확장
        if self.config.enable_synonym_expansion:
            synonym_expansions = self._synonym_expansion(query)
            expanded_queries.extend(synonym_expansions)
        
        # 3. 컨텍스트 기반 확장
        if context:
            context_expansions = self._context_based_expansion(query, context)
            expanded_queries.extend(context_expansions)
        
        # 중복 제거
        expanded_queries = list(dict.fromkeys(expanded_queries))
        
        logger.debug(f"쿼리 확장: {len(expanded_queries)}개 쿼리 생성")
        return expanded_queries
    
    def _domain_specific_expansion(self, query: str) -> List[str]:
        """도메인 특화 쿼리 확장"""
        expansions = []
        query_lower = query.lower()
        
        for category, terms in self.domain_dictionary.items():
            for term, synonyms in terms.items():
                if term in query_lower:
                    for synonym in synonyms:
                        expanded_query = query_lower.replace(term, synonym)
                        if expanded_query != query_lower:
                            expansions.append(expanded_query)
        
        return expansions
    
    def _synonym_expansion(self, query: str) -> List[str]:
        """동의어 확장"""
        synonym_mappings = {
            'ai': ['인공지능', '머신러닝', '딥러닝', '모델'],
            '모델': ['ai', '알고리즘', '시스템'],
            '성능': ['효율', '정확도', '품질'],
            '지표': ['측정값', '수치', '값'],
            '관리': ['운영', '제어', '모니터링'],
            '시스템': ['장치', '설비', '기기'],
            '확인': ['조회', '검색', '찾기'],
            '제공': ['지원', '공급', '서비스'],
            '설정': ['구성', '조정', '변경'],
            '접근': ['사용', '이용', '활용']
        }
        
        expansions = []
        for word, synonyms in synonym_mappings.items():
            if word in query.lower():
                for synonym in synonyms:
                    expanded_query = query.lower().replace(word, synonym)
                    if expanded_query != query.lower():
                        expansions.append(expanded_query)
        
        return expansions
    
    def _context_based_expansion(self, query: str, context: str) -> List[str]:
        """컨텍스트 기반 확장"""
        expansions = []
        
        # 컨텍스트에서 관련 키워드 추출
        context_keywords = self._extract_context_keywords(context)
        
        for keyword in context_keywords:
            if keyword not in query.lower():
                expansions.append(f"{query} {keyword}")
        
        return expansions
    
    def _extract_context_keywords(self, context: str) -> List[str]:
        """컨텍스트에서 키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
        keywords = []
        
        for category, terms in self.domain_dictionary.items():
            for term, synonyms in terms.items():
                if term in context.lower():
                    keywords.extend(synonyms[:2])  # 상위 2개만
        
        return list(set(keywords))
    
    def _search_with_model(self, 
                          model: SentenceTransformer,
                          model_name: str,
                          queries: List[str],
                          top_k: int) -> List[Tuple[TextChunk, float, str]]:
        """특정 모델로 검색"""
        results = []
        
        for query in queries:
            try:
                # 쿼리 임베딩 생성
                query_embedding = model.encode([query], normalize_embeddings=True)[0]
                
                # 벡터 저장소에서 검색
                search_results = self.vector_store.search(
                    query_embedding,
                    top_k=top_k,
                    similarity_threshold=self.config.similarity_threshold
                )
                
                # 모델명과 함께 결과 저장
                for chunk, score in search_results:
                    results.append((chunk, score, model_name))
                    
            except Exception as e:
                logger.warning(f"모델 {model_name} 검색 실패: {e}")
        
        return results
    
    def _calculate_ensemble_scores(self, 
                                  all_results: List[Tuple[TextChunk, float, str]]) -> Dict[str, Dict]:
        """앙상블 점수 계산"""
        chunk_scores = {}
        
        for chunk, score, model_name in all_results:
            chunk_id = chunk.chunk_id
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    'chunk': chunk,
                    'model_scores': {},
                    'total_score': 0.0,
                    'model_count': 0
                }
            
            chunk_scores[chunk_id]['model_scores'][model_name] = score
            chunk_scores[chunk_id]['model_count'] += 1
        
        # 앙상블 점수 계산
        for chunk_id, data in chunk_scores.items():
            total_score = 0.0
            weight_sum = 0.0
            
            for i, (model_name, score) in enumerate(data['model_scores'].items()):
                weight = self.config.ensemble_weights[i] if i < len(self.config.ensemble_weights) else 0.1
                total_score += score * weight
                weight_sum += weight
            
            data['total_score'] = total_score / weight_sum if weight_sum > 0 else 0.0
        
        return chunk_scores
    
    def _context_aware_reranking(self, 
                                results: Dict[str, Dict],
                                query: str,
                                context: Optional[str]) -> Dict[str, Dict]:
        """컨텍스트 인식 재랭킹"""
        if not context:
            return results
        
        # 컨텍스트 관련성 점수 계산
        for chunk_id, data in results.items():
            chunk = data['chunk']
            context_relevance = self._calculate_context_relevance(chunk, context)
            
            # 컨텍스트 관련성에 따른 가중치 적용
            context_weight = 0.2  # 20% 가중치
            data['total_score'] = (
                data['total_score'] * (1 - context_weight) + 
                context_relevance * context_weight
            )
            data['context_relevance'] = context_relevance
        
        return results
    
    def _calculate_context_relevance(self, chunk: TextChunk, context: str) -> float:
        """컨텍스트 관련성 계산"""
        chunk_text = chunk.content.lower()
        context_lower = context.lower()
        
        # 공통 키워드 수 계산
        chunk_words = set(chunk_text.split())
        context_words = set(context_lower.split())
        common_words = chunk_words & context_words
        
        if not context_words:
            return 0.0
        
        # Jaccard 유사도
        jaccard_similarity = len(common_words) / len(chunk_words | context_words)
        
        # 도메인 특화 키워드 가중치
        domain_weight = 0.0
        for category, terms in self.domain_dictionary.items():
            for term, synonyms in terms.items():
                if term in context_lower and any(syn in chunk_text for syn in synonyms):
                    domain_weight += 0.1
        
        return min(jaccard_similarity + domain_weight, 1.0)
    
    def _filter_and_rank_results(self, 
                                results: Dict[str, Dict],
                                top_k: int) -> List[SearchResult]:
        """결과 필터링 및 정렬"""
        # 점수 순으로 정렬
        sorted_results = sorted(
            results.values(),
            key=lambda x: x['total_score'],
            reverse=True
        )
        
        # SearchResult 객체로 변환
        final_results = []
        for i, data in enumerate(sorted_results[:top_k]):
            chunk = data['chunk']
            total_score = data['total_score']
            model_scores = data['model_scores']
            context_relevance = data.get('context_relevance', 0.0)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(total_score, model_scores, context_relevance)
            
            result = SearchResult(
                chunk=chunk,
                score=total_score,
                model_scores=model_scores,
                query_expansions=[],  # TODO: 실제 확장된 쿼리들 저장
                context_relevance=context_relevance,
                confidence=confidence
            )
            final_results.append(result)
        
        return final_results
    
    def _calculate_confidence(self, 
                            total_score: float,
                            model_scores: Dict[str, float],
                            context_relevance: float) -> float:
        """신뢰도 계산"""
        # 기본 점수 기반 신뢰도
        base_confidence = min(total_score, 1.0)
        
        # 모델 일치도 기반 신뢰도
        if len(model_scores) > 1:
            scores = list(model_scores.values())
            score_variance = np.var(scores)
            consistency_confidence = max(0.0, 1.0 - score_variance)
        else:
            consistency_confidence = 0.5
        
        # 컨텍스트 관련성 기반 신뢰도
        context_confidence = context_relevance
        
        # 가중 평균
        confidence = (
            base_confidence * 0.5 +
            consistency_confidence * 0.3 +
            context_confidence * 0.2
        )
        
        return min(confidence, 1.0)
    
    def _update_performance_stats(self, 
                                 start_time: float,
                                 results: List[SearchResult],
                                 expanded_queries: List[str]):
        """성능 통계 업데이트"""
        if not self.config.enable_performance_monitoring:
            return
        
        search_time = time.time() - start_time
        avg_confidence = np.mean([r.confidence for r in results]) if results else 0.0
        expansion_rate = len(expanded_queries) / 1.0  # 원본 쿼리 대비 확장률
        
        # 이동 평균으로 업데이트
        self.performance_stats['total_searches'] += 1
        n = self.performance_stats['total_searches']
        
        self.performance_stats['avg_search_time'] = (
            (self.performance_stats['avg_search_time'] * (n - 1) + search_time) / n
        )
        self.performance_stats['avg_confidence'] = (
            (self.performance_stats['avg_confidence'] * (n - 1) + avg_confidence) / n
        )
        self.performance_stats['query_expansion_rate'] = (
            (self.performance_stats['query_expansion_rate'] * (n - 1) + expansion_rate) / n
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            'performance': self.performance_stats.copy(),
            'config': {
                'models_loaded': list(self.embedding_models.keys()),
                'ensemble_weights': self.config.ensemble_weights,
                'max_candidates': self.config.max_candidates,
                'similarity_threshold': self.config.similarity_threshold
            }
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self.performance_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'avg_confidence': 0.0,
            'query_expansion_rate': 0.0
        }
        logger.info("성능 통계 초기화 완료")

# 편의 함수들
def create_enhanced_searcher(vector_store: HybridVectorStore,
                           primary_model: str = "jhgan/ko-sroberta-multitask",
                           secondary_models: Optional[List[str]] = None) -> EnhancedVectorSearcher:
    """향상된 검색기 생성"""
    config = EnhancedSearchConfig(
        primary_model=primary_model,
        secondary_models=secondary_models or [
            "all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ],
        ensemble_weights=[1.0, 0.3, 0.2],
        max_candidates=300,
        final_top_k=10,
        similarity_threshold=0.2,
        enable_query_expansion=True,
        enable_domain_expansion=True,
        enable_synonym_expansion=True,
        enable_dynamic_weights=True,
        context_aware_search=True,
        enable_performance_monitoring=True
    )
    
    return EnhancedVectorSearcher(vector_store, config)

def create_fast_searcher(vector_store: HybridVectorStore) -> EnhancedVectorSearcher:
    """속도 최적화 검색기 생성"""
    config = EnhancedSearchConfig(
        primary_model="jhgan/ko-sroberta-multitask",
        secondary_models=None,
        max_candidates=100,
        final_top_k=5,
        similarity_threshold=0.3,
        enable_query_expansion=False,
        enable_domain_expansion=True,
        enable_synonym_expansion=False,
        enable_dynamic_weights=False,
        context_aware_search=False,
        enable_performance_monitoring=False
    )
    
    return EnhancedVectorSearcher(vector_store, config)
