"""
LLM 기반 쿼리 확장 시스템

Qwen 모델을 사용하여 정수처리 도메인 특화 쿼리 확장
하드코딩된 키워드에서 벗어나 동적이고 지능적인 쿼리 확장 제공
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama가 설치되지 않았습니다. pip install ollama")

logger = logging.getLogger(__name__)

class ExpansionStrategy(Enum):
    """쿼리 확장 전략"""
    SYNONYMS = "synonyms"  # 동의어 확장
    RELATED_CONCEPTS = "related_concepts"  # 관련 개념 확장
    TECHNICAL_TERMS = "technical_terms"  # 기술 용어 확장
    PROCESS_CONTEXT = "process_context"  # 공정 맥락 확장
    COMPREHENSIVE = "comprehensive"  # 종합 확장

@dataclass
class QueryExpansionConfig:
    """쿼리 확장 설정"""
    llm_model: str = "qwen2:1.5b-instruct-q4_K_M"
    max_expansions: int = 5  # 최대 확장 쿼리 수
    temperature: float = 0.3  # 창의성 조절
    max_tokens: int = 200  # 최대 토큰 수
    enable_caching: bool = True  # 캐싱 활성화
    expansion_strategies: List[ExpansionStrategy] = None
    
    def __post_init__(self):
        if self.expansion_strategies is None:
            self.expansion_strategies = [ExpansionStrategy.COMPREHENSIVE]

@dataclass
class ExpandedQuery:
    """확장된 쿼리"""
    original_query: str
    expanded_queries: List[str]
    expansion_type: ExpansionStrategy
    confidence: float
    technical_terms: List[str]
    related_processes: List[str]

class LLMQueryExpander:
    """LLM 기반 쿼리 확장기"""
    
    def __init__(self, config: Optional[QueryExpansionConfig] = None):
        """쿼리 확장기 초기화"""
        self.config = config or QueryExpansionConfig()
        
        # 캐시 (메모리 기반)
        self.query_cache = {} if self.config.enable_caching else None
        
        # 정수처리 도메인 컨텍스트
        self.domain_context = """
정수처리 시설에서는 다음과 같은 주요 공정이 있습니다:
1. 취수(원수) - 수원에서 물을 취수하는 과정
2. 응집 - PAC, 황산알루미늄 등 응집제를 투입하여 불순물을 응집시키는 과정
3. 침전 - 응집된 플록을 침전시켜 제거하는 과정
4. 여과 - 모래여과, 활성탄여과 등을 통해 미세 불순물을 제거하는 과정
5. 소독 - 염소, UV, 오존 등으로 병원균을 제거하는 과정
6. 배수 - 정수된 물을 배수관망을 통해 공급하는 과정

주요 수질 지표: 탁도(NTU), pH, 잔류염소, 대장균, 일반세균, 색도, 냄새, 맛
주요 단위: mg/L, ppm, NTU, CFU/mL, m³/h, m/h
"""
        
        # 정수처리 전문 용어 사전
        self.technical_terms = {
            "응집": ["coagulation", "flocculation", "응집제", "PAC", "황산알루미늄", "염화제이철", "혼화"],
            "침전": ["sedimentation", "settling", "침전지", "상등수", "슬러지", "체류시간"],
            "여과": ["filtration", "filter", "모래여과", "활성탄", "역세척", "여과속도", "여과부하"],
            "소독": ["disinfection", "chlorination", "염소", "UV", "오존", "잔류염소", "CT값"],
            "수질": ["water quality", "탁도", "pH", "대장균", "일반세균", "색도", "냄새", "맛"],
            "운영": ["operation", "관리", "모니터링", "제어", "자동화", "최적화"]
        }
        
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama가 설치되지 않아 쿼리 확장 기능이 제한됩니다.")
        
        logger.info(f"LLM 쿼리 확장기 초기화 (모델: {self.config.llm_model})")
    
    def expand_query(self, query: str, 
                    strategy: ExpansionStrategy = ExpansionStrategy.COMPREHENSIVE) -> ExpandedQuery:
        """쿼리 확장"""
        logger.info(f"쿼리 확장 시작: '{query}' (전략: {strategy.value})")
        
        # 캐시 확인
        cache_key = f"{query}_{strategy.value}"
        if self.query_cache and cache_key in self.query_cache:
            logger.debug("캐시에서 확장 쿼리 반환")
            return self.query_cache[cache_key]
        
        try:
            if not OLLAMA_AVAILABLE:
                # Ollama가 없으면 규칙 기반 확장
                return self._rule_based_expansion(query, strategy)
            
            # LLM 기반 확장
            expanded_query = self._llm_based_expansion(query, strategy)
            
            # 캐시 저장
            if self.query_cache:
                self.query_cache[cache_key] = expanded_query
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"쿼리 확장 실패: {e}")
            # 실패 시 규칙 기반 확장으로 대체
            return self._rule_based_expansion(query, strategy)
    
    def _llm_based_expansion(self, query: str, strategy: ExpansionStrategy) -> ExpandedQuery:
        """LLM 기반 쿼리 확장"""
        
        # 전략별 프롬프트 생성
        prompt = self._generate_expansion_prompt(query, strategy)
        
        try:
            # Ollama API 호출
            response = ollama.generate(
                model=self.config.llm_model,
                prompt=prompt,
                options={
                    'temperature': self.config.temperature,
                    'num_predict': self.config.max_tokens,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            # 응답 파싱
            expanded_queries = self._parse_llm_response(response['response'])
            
            # 기술 용어 및 관련 공정 추출
            technical_terms = self._extract_technical_terms(query + " " + " ".join(expanded_queries))
            related_processes = self._extract_related_processes(query + " " + " ".join(expanded_queries))
            
            return ExpandedQuery(
                original_query=query,
                expanded_queries=expanded_queries,
                expansion_type=strategy,
                confidence=0.8,  # LLM 기반이므로 높은 신뢰도
                technical_terms=technical_terms,
                related_processes=related_processes
            )
            
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            raise
    
    def _generate_expansion_prompt(self, query: str, strategy: ExpansionStrategy) -> str:
        """전략별 확장 프롬프트 생성"""
        
        base_prompt = f"""
{self.domain_context}

사용자 질문: "{query}"

위 정수처리 도메인 맥락을 바탕으로, 사용자 질문과 관련된 검색을 위한 확장 쿼리를 생성해주세요.
"""
        
        if strategy == ExpansionStrategy.SYNONYMS:
            specific_instruction = """
동의어와 유사한 표현을 중심으로 확장해주세요.
예: "응집" → "coagulation", "flocculation", "혼화"
"""
        elif strategy == ExpansionStrategy.RELATED_CONCEPTS:
            specific_instruction = """
관련 개념과 연관 용어를 중심으로 확장해주세요.
예: "여과" → "역세척", "여과속도", "모래층", "활성탄"
"""
        elif strategy == ExpansionStrategy.TECHNICAL_TERMS:
            specific_instruction = """
기술적 용어와 전문 표현을 중심으로 확장해주세요.
예: "탁도" → "NTU", "turbidity", "부유물질", "SS"
"""
        elif strategy == ExpansionStrategy.PROCESS_CONTEXT:
            specific_instruction = """
공정의 맥락과 순서를 고려하여 확장해주세요.
예: "침전" → "응집 후 침전", "침전지 운영", "슬러지 처리"
"""
        else:  # COMPREHENSIVE
            specific_instruction = """
동의어, 관련 개념, 기술 용어, 공정 맥락을 종합적으로 고려하여 확장해주세요.
정수처리 전문가가 검색할 만한 다양한 표현을 포함해주세요.
"""
        
        format_instruction = f"""
{specific_instruction}

다음 형식으로 최대 {self.config.max_expansions}개의 확장 쿼리를 생성해주세요:
1. [확장쿼리1]
2. [확장쿼리2]
3. [확장쿼리3]
...

확장 쿼리는 한국어로 작성하고, 원래 질문의 의도를 유지하면서 검색 성능을 향상시킬 수 있도록 해주세요.
"""
        
        return base_prompt + format_instruction
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """LLM 응답 파싱"""
        expanded_queries = []
        
        # 번호 매김 패턴으로 추출
        patterns = [
            r'\d+\.\s*(.+?)(?=\n\d+\.|\n$|$)',  # 1. 형태
            r'-\s*(.+?)(?=\n-|\n$|$)',  # - 형태
            r'•\s*(.+?)(?=\n•|\n$|$)',  # • 형태
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
            if matches:
                expanded_queries = [match.strip() for match in matches]
                break
        
        # 패턴 매칭 실패 시 줄 단위로 분할
        if not expanded_queries:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            expanded_queries = [line for line in lines if len(line) > 5]
        
        # 최대 개수 제한
        return expanded_queries[:self.config.max_expansions]
    
    def _rule_based_expansion(self, query: str, strategy: ExpansionStrategy) -> ExpandedQuery:
        """규칙 기반 쿼리 확장 (대체 방안)"""
        expanded_queries = []
        
        # 기본 동의어 확장
        for term, synonyms in self.technical_terms.items():
            if term in query:
                for synonym in synonyms[:2]:  # 상위 2개만
                    expanded_query = query.replace(term, synonym)
                    if expanded_query != query:
                        expanded_queries.append(expanded_query)
        
        # 관련 용어 추가
        query_lower = query.lower()
        if "응집" in query_lower:
            expanded_queries.extend([
                query + " PAC 투입",
                query + " 혼화시간",
                query + " 플록 형성"
            ])
        elif "여과" in query_lower:
            expanded_queries.extend([
                query + " 역세척",
                query + " 여과속도",
                query + " 모래층"
            ])
        elif "소독" in query_lower:
            expanded_queries.extend([
                query + " 염소투입",
                query + " 잔류염소",
                query + " CT값"
            ])
        
        # 중복 제거 및 제한
        unique_queries = list(dict.fromkeys(expanded_queries))[:self.config.max_expansions]
        
        # 기술 용어 추출
        technical_terms = self._extract_technical_terms(query)
        related_processes = self._extract_related_processes(query)
        
        return ExpandedQuery(
            original_query=query,
            expanded_queries=unique_queries,
            expansion_type=strategy,
            confidence=0.6,  # 규칙 기반이므로 중간 신뢰도
            technical_terms=technical_terms,
            related_processes=related_processes
        )
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """기술 용어 추출"""
        technical_terms = []
        text_lower = text.lower()
        
        # 단위 패턴
        unit_patterns = [
            r'\d+(?:\.\d+)?\s*(?:mg/L|㎎/L|ppm|NTU|도|℃|°C)',
            r'\d+(?:\.\d+)?\s*(?:%|시간|분|초)',
            r'\d+(?:\.\d+)?\s*(?:m³/h|㎥/h|m3/day|㎥/일)'
        ]
        
        for pattern in unit_patterns:
            matches = re.findall(pattern, text)
            technical_terms.extend(matches)
        
        # 전문 용어
        for category, terms in self.technical_terms.items():
            for term in terms:
                if term.lower() in text_lower:
                    technical_terms.append(term)
        
        return list(set(technical_terms))
    
    def _extract_related_processes(self, text: str) -> List[str]:
        """관련 공정 추출"""
        processes = []
        text_lower = text.lower()
        
        process_keywords = {
            "취수": ["취수", "원수", "수원"],
            "응집": ["응집", "혼화", "PAC", "황산알루미늄"],
            "침전": ["침전", "침전지", "슬러지"],
            "여과": ["여과", "필터", "모래", "활성탄"],
            "소독": ["소독", "염소", "UV", "오존"],
            "배수": ["배수", "송수", "가압"]
        }
        
        for process, keywords in process_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                processes.append(process)
        
        return processes
    
    def expand_multiple_queries(self, queries: List[str], 
                              strategy: ExpansionStrategy = ExpansionStrategy.COMPREHENSIVE) -> List[ExpandedQuery]:
        """여러 쿼리 일괄 확장"""
        expanded_queries = []
        
        for query in queries:
            try:
                expanded = self.expand_query(query, strategy)
                expanded_queries.append(expanded)
            except Exception as e:
                logger.error(f"쿼리 '{query}' 확장 실패: {e}")
        
        return expanded_queries
    
    def get_expansion_stats(self) -> Dict[str, Any]:
        """확장 통계 정보"""
        return {
            'config': {
                'llm_model': self.config.llm_model,
                'max_expansions': self.config.max_expansions,
                'temperature': self.config.temperature
            },
            'cache_size': len(self.query_cache) if self.query_cache else 0,
            'ollama_available': OLLAMA_AVAILABLE,
            'technical_terms_count': sum(len(terms) for terms in self.technical_terms.values())
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        if self.query_cache:
            self.query_cache.clear()
            logger.info("쿼리 확장 캐시 초기화 완료")

# 편의 함수들
def create_llm_query_expander(llm_model: str = "qwen2:1.5b-instruct-q4_K_M",
                             max_expansions: int = 5) -> LLMQueryExpander:
    """LLM 쿼리 확장기 생성"""
    config = QueryExpansionConfig(
        llm_model=llm_model,
        max_expansions=max_expansions,
        temperature=0.3,
        enable_caching=True
    )
    return LLMQueryExpander(config)

def create_conservative_expander() -> LLMQueryExpander:
    """보수적 쿼리 확장기 (적은 확장)"""
    config = QueryExpansionConfig(
        max_expansions=3,
        temperature=0.2,
        expansion_strategies=[ExpansionStrategy.SYNONYMS, ExpansionStrategy.TECHNICAL_TERMS]
    )
    return LLMQueryExpander(config)

def create_comprehensive_expander() -> LLMQueryExpander:
    """포괄적 쿼리 확장기 (많은 확장)"""
    config = QueryExpansionConfig(
        max_expansions=8,
        temperature=0.4,
        expansion_strategies=[ExpansionStrategy.COMPREHENSIVE]
    )
    return LLMQueryExpander(config)
