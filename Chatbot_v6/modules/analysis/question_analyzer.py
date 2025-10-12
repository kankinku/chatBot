"""
Question Analyzer - 질문 분석기

질문 유형, 키워드, 가중치를 분석합니다 (단일 책임).
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


class QuestionAnalyzer:
    """
    질문 분석기
    
    단일 책임: 질문 분석만 수행
    """
    
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
        )
    
    def _classify_question_type(self, question: str, q_lower: str) -> str:
        """질문 유형 분류"""
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

