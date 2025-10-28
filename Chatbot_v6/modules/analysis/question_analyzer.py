"""
Question Analyzer

질문 유형 분류, 검색 가중치 결정, 쿼리 확장.
LRU 캐싱으로 성능 최적화 (256개).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from functools import lru_cache

from config.constants import QuestionType
from modules.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QuestionAnalysis:
    """질문 분석 결과"""
    qtype: str
    length: int
    key_token_count: int
    rrf_vector_weight: float
    rrf_bm25_weight: float
    threshold_adj: float
    has_number: bool = False
    has_unit: bool = False
    has_domain_keyword: bool = False
    expanded_query: str = ""  # 쿼리 확장 결과


class QuestionAnalyzer:
    """질문 유형 분류 및 검색 파라미터 결정"""
    
    def __init__(self, domain_dict_path: Optional[str] = None):
        """
        Args:
            domain_dict_path: 도메인 사전 경로
        """
        self.domain_dict_path = domain_dict_path
        self.domain_dict = self._load_domain_dict()
        
        logger.info("QuestionAnalyzer initialized",
                   has_domain_dict=self.domain_dict is not None)
    
    def _load_domain_dict(self) -> Optional[dict]:
        """도메인 사전 로드"""
        if not self.domain_dict_path:
            return None
        
        try:
            path = Path(self.domain_dict_path)
            if not path.exists():
                logger.warning(f"Domain dict not found: {self.domain_dict_path}")
                return None
            
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        except Exception as e:
            logger.warning(f"Failed to load domain dict: {e}")
            return None
    
    # 🚀 최적화 4: 질문 분석 결과 캐싱 (LRU 256개)
    @lru_cache(maxsize=256)
    def analyze(self, question: str) -> QuestionAnalysis:
        """
        질문 분석
        
        Args:
            question: 질문 텍스트
            
        Returns:
            QuestionAnalysis 객체
        """
        q_lower = question.lower().strip()
        
        # 토큰 추출
        tokens = re.findall(r"[\w\-/\.%°℃]+", q_lower)
        length = len(tokens)
        
        # 도메인 정보 추출
        units = set(self.domain_dict.get("units", [])) if self.domain_dict else set()
        domain_kw = self.domain_dict.get("keywords", []) if self.domain_dict else []
        
        # 특징 추출
        has_number = bool(re.search(r"\d", q_lower))
        has_unit = any(u.lower() in q_lower for u in units)
        has_domain_kw = any(kw.lower() in q_lower for kw in domain_kw if kw)
        
        numeric_like = has_number or has_unit or has_domain_kw
        
        # 질문 유형 분류
        qtype = self._classify_question_type(question, q_lower)
        
        # 가중치 결정
        vector_weight, bm25_weight = self._determine_weights(qtype)
        
        # 키워드 토큰 수
        key_token_count = len([t for t in tokens if len(t) >= 2])
        if self.domain_dict:
            domain_tokens = [
                t for t in tokens 
                if any(kw.lower() in t for kw in ["ai", "플랫폼", "공정", "모델", "알고리즘"])
            ]
            key_token_count += len(domain_tokens)
        
        # 임계값 조정
        threshold_adj = -0.02
        if qtype in ["system_info", "technical_spec"]:
            threshold_adj -= 0.1
        
        # 쿼리 확장
        expanded_query = self._expand_query(question)
        
        return QuestionAnalysis(
            qtype=qtype,
            length=length,
            key_token_count=key_token_count,
            rrf_vector_weight=vector_weight,
            rrf_bm25_weight=bm25_weight,
            threshold_adj=threshold_adj,
            has_number=has_number,
            has_unit=has_unit,
            has_domain_keyword=has_domain_kw,
            expanded_query=expanded_query,
        )
    
    def is_greeting(self, question: str) -> bool:
        """인사말인지 확인"""
        q_lower = question.lower().strip()
        
        # 인사말 패턴
        greeting_patterns = [
            '안녕', '반갑', '하이', 'hi', 'hello', '안녕하세요', 
            '안녕하십니까', '반갑습니다', '만나서 반가워', 
            '만나서 반가워요', '반가워', '반가워요',
            '좋은 아침', '좋은 오후', '좋은 저녁', '좋은 밤',
            '환영', '환영합니다', '오신 것을 환영'
        ]
        
        # 간단한 인사말만 체크 (너무 짧거나 패턴이 정확히 일치하는 경우)
        words = re.findall(r'\w+', q_lower)
        
        # 1-3단어로 구성되고 인사 패턴이 포함된 경우
        if len(words) <= 3:
            for pattern in greeting_patterns:
                if pattern in q_lower:
                    # 특수 케이스 제외 (예: "안녕하세요 처리 방법" 같은 질문)
                    if any(keyword in q_lower for keyword in ['방법', '처리', '설정', '문제', '오류']):
                        return False
                    return True
        
        return False
    
    def _classify_question_type(self, question: str, q_lower: str) -> str:
        """질문 유형 분류"""
        # 인사말 체크
        if self.is_greeting(question):
            return "greeting"
        
        # 패턴 매칭
        patterns = {
            "definition": r"(정의|무엇|란|의미|개념|설명|목적|기능|특징)",
            "procedural": r"(방법|절차|순서|어떻게|운영|조치|설정|접속|로그인)",
            "comparative": r"(비교|vs|더|높|낮|차이|장점|단점|차이점)",
            "problem": r"(문제|오류|이상|고장|원인|대응|대책|해결|증상)",
            "system_info": r"(시스템|플랫폼|대시보드|로그인|계정|비밀번호|주소|url)",
            "technical_spec": r"(모델|알고리즘|성능|지표|입력변수|설정값|고려사항)",
            "operational": r"(운영|모드|제어|알람|진단|결함|정보|현황)",
        }
        
        # 도메인 사전 패턴 추가
        if self.domain_dict:
            for key in ["definition", "procedural", "comparative", "problem"]:
                if key in self.domain_dict:
                    terms = self.domain_dict[key]
                    if terms:
                        patterns[key] += "|" + "|".join(re.escape(t) for t in terms[:10])
        
        # 숫자/단위 체크
        has_number = bool(re.search(r"\d", q_lower))
        has_unit = False
        if self.domain_dict:
            units = self.domain_dict.get("units", [])
            has_unit = any(u.lower() in q_lower for u in units)
        
        if has_number or has_unit:
            return "numeric"
        
        # 패턴 매칭
        for qtype, pattern in patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                return qtype
        
        return "general"
    
    def _determine_weights(self, qtype: str) -> tuple[float, float]:
        """질문 유형별 검색 가중치 결정"""
        weights = {
            "system_info": (0.4, 0.6),       # BM25 우선 (키워드)
            "technical_spec": (0.4, 0.6),    # BM25 우선
            "numeric": (0.4, 0.6),           # BM25 우선
            "operational": (0.7, 0.3),       # Vector 우선 (의미)
            "procedural": (0.7, 0.3),        # Vector 우선
            "definition": (0.7, 0.3),        # Vector 우선
            "comparative": (0.6, 0.4),       # 균형
            "problem": (0.6, 0.4),           # 균형
            "general": (0.58, 0.42),         # 기본값
        }
        
        return weights.get(qtype, (0.58, 0.42))
    
    def _expand_query(self, question: str) -> str:
        """
        쿼리 확장 최적화 - 정밀한 키워드 매핑
        
        질문 유형별로 선택적 쿼리 확장을 적용하여 노이즈 최소화
        """
        # 질문 유형 분석
        q_lower = question.lower()
        
        # 확장이 필요한 경우만 적용
        should_expand = any(keyword in q_lower for keyword in [
            "AI", "플랫폼", "공정", "수질", "탁도", "pH", "온도", "유량", "압력", "전력", "탄소"
        ])
        
        if not should_expand:
            return question
        
        # 정밀한 도메인 키워드 매핑 (정수장 특화)
        precise_expansions = {
            "AI": ["AI", "인공지능", "모델", "알고리즘"],
            "플랫폼": ["플랫폼", "대시보드"],
            "공정": ["공정", "처리"],
            "수질": ["수질", "탁도", "SS"],
            "pH": ["pH", "산성도"],
            "온도": ["온도", "수온"],
            "유량": ["유량", "유입량"],
            "압력": ["압력", "수압"],
            "전력": ["전력", "kWh"],
            "탄소": ["탄소", "CO2"]
        }
        
        expanded_terms = []
        original_terms = question.split()
        
        # 원본 단어 유지
        expanded_terms.extend(original_terms)
        
        # 선택적 확장 (최대 2개 키워드만)
        expansion_count = 0
        for term in original_terms:
            if expansion_count >= 2:  # 확장 제한
                break
                
            for key, synonyms in precise_expansions.items():
                if key.lower() in term.lower():
                    # 가장 관련성 높은 1개 키워드만 추가
                    best_synonym = synonyms[1] if len(synonyms) > 1 else synonyms[0]
                    if best_synonym not in expanded_terms:
                        expanded_terms.append(best_synonym)
                        expansion_count += 1
                    break
        
        # 중복 제거 및 정리
        unique_terms = list(dict.fromkeys(expanded_terms))
        expanded_query = " ".join(unique_terms)
        
        logger.debug(f"Optimized query expansion: '{question}' -> '{expanded_query}'")
        return expanded_query

