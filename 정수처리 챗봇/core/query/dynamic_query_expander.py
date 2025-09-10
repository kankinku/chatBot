"""
동적 쿼리 확장 모듈

Qwen 모델을 활용하여 정수처리 도메인 특화 쿼리 확장을 수행합니다.
기존 하드코딩된 키워드 매칭에서 LLM 기반 의미적 확장으로 진화합니다.
"""

import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

# 캐싱을 위한 import
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class QueryExpansionConfig:
    """쿼리 확장 설정"""
    max_expanded_queries: int = 5  # 최대 확장 쿼리 수
    cache_enabled: bool = True     # 캐싱 활성화
    cache_size: int = 1000        # 캐시 크기
    timeout_seconds: float = 2.0   # Qwen 호출 타임아웃
    fallback_enabled: bool = True  # 폴백 활성화
    expansion_temperature: float = 0.3  # 생성 온도 (낮을수록 보수적)

class DynamicQueryExpander:
    """동적 쿼리 확장기 (Qwen 기반)"""
    
    def __init__(self, 
                 ollama_interface,
                 config: Optional[QueryExpansionConfig] = None):
        """확장기 초기화"""
        self.ollama = ollama_interface
        self.config = config or QueryExpansionConfig()
        
        # 캐시 초기화
        self.query_cache = {} if self.config.cache_enabled else None
        
        # 정수처리 도메인 프롬프트 템플릿
        self.expansion_prompt_template = """
정수처리 전문가로서 다음 질문을 분석하고 관련된 검색 쿼리들을 생성해주세요.

원본 질문: {original_query}

다음 지침을 따라 3-5개의 확장된 검색 쿼리를 생성해주세요:

1. 정수처리 공정별 관점에서 확장 (착수, 약품, 혼화응집, 침전, 여과, 소독)
2. 기술적 세부사항 포함 (모델명, 성능지표, 운영조건 등)
3. 동의어 및 전문용어 활용
4. 구체적이고 검색 가능한 형태로 작성

확장된 쿼리들 (한 줄에 하나씩):
1. 
2. 
3. 
4. 
5. 
"""
        
        # 폴백용 하드코딩 확장 규칙
        self.fallback_expansion_rules = {
            # 공정별 확장
            '착수': ['착수공정', '수위제어', '유입량조절', '정수지관리', 'k-means 군집분석'],
            '약품': ['약품공정', '응집제주입', 'n-beats 모델', '탁도측정', '알칼리도'],
            '혼화응집': ['혼화응집공정', '교반속도', '회전속도제어', 'G값계산', '동점성계수'],
            '침전': ['침전공정', '슬러지처리', '수집기운전', '대차스케줄', '침전지관리'],
            '여과': ['여과공정', '여과지세척', '역세척주기', '여과운전', '수위관리'],
            '소독': ['소독공정', '염소주입', '잔류염소농도', '체류시간관리', '전차염'],
            
            # 시스템별 확장
            'ems': ['ems시스템', '에너지관리', '전력피크', '펌프제어', '전력사용량'],
            'pms': ['pms시스템', '모터진단', '예방정비', '설비관리', '진동온도'],
            
            # 모델별 확장
            'n-beats': ['n-beats 모델', 'n-beats 성능', 'mae mse rmse', 'r2 정확도'],
            'xgb': ['xgb 모델', 'extreme gradient boosting', 'xgb 성능'],
            'lstm': ['lstm 모델', 'long short-term memory', 'lstm 성능'],
            
            # 성능지표 확장
            '성능': ['성능지표', 'mae', 'mse', 'rmse', 'r²', 'r2', '정확도', '오차'],
            '모델': ['ai 모델', '머신러닝 모델', '예측 모델', '알고리즘', '성능평가']
        }
        
        logger.info(f"동적 쿼리 확장기 초기화 완료 (캐시: {self.config.cache_enabled}, 타임아웃: {self.config.timeout_seconds}초)")
    
    def expand_query(self, 
                    original_query: str,
                    context: Optional[str] = None,
                    answer_target: Optional[str] = None) -> List[str]:
        """
        쿼리를 동적으로 확장
        
        Args:
            original_query: 원본 쿼리
            context: 추가 컨텍스트
            answer_target: 답변 목표
            
        Returns:
            확장된 쿼리 리스트
        """
        if not original_query.strip():
            return [original_query]
        
        # 캐시 확인
        cache_key = self._generate_cache_key(original_query, context, answer_target)
        if self.query_cache and cache_key in self.query_cache:
            logger.debug(f"캐시에서 쿼리 확장 결과 반환: {original_query[:30]}...")
            return self.query_cache[cache_key]
        
        start_time = time.time()
        
        try:
            # Qwen을 사용한 동적 확장
            expanded_queries = self._expand_with_qwen(original_query, context, answer_target)
            
            # 결과 검증 및 정리
            expanded_queries = self._validate_and_clean_queries(original_query, expanded_queries)
            
            expansion_time = time.time() - start_time
            logger.info(f"Qwen 기반 쿼리 확장 완료: {expansion_time:.3f}초, {len(expanded_queries)}개 쿼리 생성")
            
        except Exception as e:
            logger.warning(f"Qwen 기반 확장 실패, 폴백 사용: {e}")
            
            if self.config.fallback_enabled:
                expanded_queries = self._expand_with_fallback(original_query, context, answer_target)
                expansion_time = time.time() - start_time
                logger.info(f"폴백 기반 쿼리 확장 완료: {expansion_time:.3f}초, {len(expanded_queries)}개 쿼리 생성")
            else:
                expanded_queries = [original_query]
        
        # 캐시에 저장
        if self.query_cache:
            # 캐시 크기 제한
            if len(self.query_cache) >= self.config.cache_size:
                # 가장 오래된 항목 제거 (간단한 FIFO)
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
            
            self.query_cache[cache_key] = expanded_queries
        
        return expanded_queries
    
    def _expand_with_qwen(self, 
                         original_query: str,
                         context: Optional[str],
                         answer_target: Optional[str]) -> List[str]:
        """Qwen 모델을 사용한 쿼리 확장"""
        
        # 프롬프트 구성
        prompt = self.expansion_prompt_template.format(
            original_query=original_query
        )
        
        # 컨텍스트 추가
        if context:
            prompt += f"\n\n추가 컨텍스트: {context}"
        
        if answer_target:
            prompt += f"\n답변 목표: {answer_target}"
        
        # Qwen 호출 (타임아웃 적용)
        try:
            response = self._call_qwen_with_timeout(prompt)
            
            # 응답 파싱
            expanded_queries = self._parse_qwen_response(response, original_query)
            
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Qwen 호출 실패: {e}")
            raise
    
    def _call_qwen_with_timeout(self, prompt: str) -> str:
        """타임아웃을 적용한 Qwen 호출"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Qwen 호출 타임아웃")
        
        # 타임아웃 설정
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.config.timeout_seconds))
        
        try:
            response = self.ollama.generate(prompt)
            signal.alarm(0)  # 타임아웃 해제
            return response
        except Exception as e:
            signal.alarm(0)  # 타임아웃 해제
            raise e
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _parse_qwen_response(self, response: str, original_query: str) -> List[str]:
        """Qwen 응답 파싱"""
        if not response:
            return [original_query]
        
        expanded_queries = [original_query]  # 원본 쿼리 포함
        
        # 번호가 있는 리스트 형태 파싱
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # 번호 제거 (1., 2., -, * 등)
            cleaned_line = re.sub(r'^[\d\-\*\•]+\.?\s*', '', line).strip()
            
            if cleaned_line and len(cleaned_line) > 5:  # 최소 길이 체크
                # 중복 제거
                if cleaned_line not in expanded_queries:
                    expanded_queries.append(cleaned_line)
                
                # 최대 개수 제한
                if len(expanded_queries) >= self.config.max_expanded_queries + 1:  # +1은 원본 쿼리
                    break
        
        return expanded_queries
    
    def _expand_with_fallback(self, 
                            original_query: str,
                            context: Optional[str],
                            answer_target: Optional[str]) -> List[str]:
        """폴백 기반 쿼리 확장 (하드코딩된 규칙)"""
        
        expanded_queries = [original_query]
        query_lower = original_query.lower()
        
        # 키워드 기반 확장
        for keyword, expansions in self.fallback_expansion_rules.items():
            if keyword in query_lower:
                for expansion in expansions[:2]:  # 각 키워드당 최대 2개
                    # 원본 쿼리에 확장 키워드 추가
                    expanded_query = f"{original_query} {expansion}"
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
                    
                    if len(expanded_queries) >= self.config.max_expanded_queries:
                        break
                
                if len(expanded_queries) >= self.config.max_expanded_queries:
                    break
        
        # 답변 목표 기반 확장
        if answer_target and len(expanded_queries) < self.config.max_expanded_queries:
            target_query = f"{answer_target} {original_query}"
            if target_query not in expanded_queries:
                expanded_queries.append(target_query)
        
        return expanded_queries[:self.config.max_expanded_queries]
    
    def _validate_and_clean_queries(self, original_query: str, queries: List[str]) -> List[str]:
        """확장된 쿼리들을 검증하고 정리"""
        cleaned_queries = []
        original_lower = original_query.lower()
        
        for query in queries:
            if not query or not query.strip():
                continue
            
            query = query.strip()
            
            # 너무 짧거나 긴 쿼리 제거
            if len(query) < 5 or len(query) > 200:
                continue
            
            # 원본과 너무 유사한 쿼리 제거 (단, 원본은 유지)
            if query.lower() != original_lower:
                # 간단한 유사도 체크 (Jaccard)
                original_words = set(original_lower.split())
                query_words = set(query.lower().split())
                
                if original_words and query_words:
                    intersection = len(original_words & query_words)
                    union = len(original_words | query_words)
                    similarity = intersection / union
                    
                    # 너무 유사하면 제외 (90% 이상)
                    if similarity > 0.9:
                        continue
            
            # 중복 제거
            if query not in cleaned_queries:
                cleaned_queries.append(query)
        
        # 원본 쿼리가 없으면 추가
        if original_query not in cleaned_queries:
            cleaned_queries.insert(0, original_query)
        
        return cleaned_queries[:self.config.max_expanded_queries]
    
    def _generate_cache_key(self, 
                          original_query: str,
                          context: Optional[str],
                          answer_target: Optional[str]) -> str:
        """캐시 키 생성"""
        key_components = [original_query]
        
        if context:
            key_components.append(f"ctx:{context}")
        
        if answer_target:
            key_components.append(f"target:{answer_target}")
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        if not self.query_cache:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cache_size": len(self.query_cache),
            "cache_limit": self.config.cache_size,
            "hit_rate": "N/A"  # 실제 구현시 hit/miss 카운터 추가 가능
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        if self.query_cache:
            self.query_cache.clear()
            logger.info("쿼리 확장 캐시 초기화 완료")

# 배치 처리를 위한 유틸리티 함수
def expand_queries_batch(expander: DynamicQueryExpander, 
                        queries: List[str],
                        contexts: Optional[List[str]] = None,
                        answer_targets: Optional[List[str]] = None) -> List[List[str]]:
    """
    여러 쿼리를 배치로 확장
    
    Args:
        expander: 쿼리 확장기
        queries: 원본 쿼리 리스트
        contexts: 컨텍스트 리스트 (선택적)
        answer_targets: 답변 목표 리스트 (선택적)
        
    Returns:
        확장된 쿼리들의 리스트
    """
    results = []
    
    for i, query in enumerate(queries):
        context = contexts[i] if contexts and i < len(contexts) else None
        answer_target = answer_targets[i] if answer_targets and i < len(answer_targets) else None
        
        expanded = expander.expand_query(query, context, answer_target)
        results.append(expanded)
    
    return results
