"""
키워드 인식률 향상을 위한 유틸리티 모듈

이 모듈은 질문 분석과 답변 생성에서 키워드 인식률을 높이기 위한
다양한 기능들을 제공합니다.
"""

import re
import json
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class KeywordEnhancer:
    """
    키워드 인식률 향상 클래스
    
    주요 기능:
    1. 도메인별 전문 용어 사전 관리
    2. 키워드 정규화 및 표준화
    3. 키워드 가중치 계산
    4. 키워드 확장 및 추천
    """
    
    def __init__(self, domain: str = "general"):
        """
        KeywordEnhancer 초기화
        
        Args:
            domain: 도메인 (general, technical, business, academic 등)
        """
        self.domain = domain
        self.domain_keywords = self._load_domain_keywords(domain)
        self.synonym_dict = self._load_synonym_dictionary()
        self.abbreviation_dict = self._load_abbreviation_dictionary()
        
        logger.info(f"KeywordEnhancer 초기화 완료 (도메인: {domain})")
    
    def _load_domain_keywords(self, domain: str) -> Dict[str, float]:
        """
        도메인별 전문 키워드 로드
        
        Args:
            domain: 도메인 이름
            
        Returns:
            키워드와 가중치 딕셔너리
        """
        domain_keywords = {
            "general": {
                "시스템": 1.0, "프로그램": 1.0, "소프트웨어": 1.0,
                "데이터": 1.0, "정보": 1.0, "자료": 1.0,
                "사용자": 1.0, "관리자": 1.0, "고객": 1.0,
                "기능": 1.0, "특성": 1.0, "역할": 1.0,
                "설정": 1.0, "구성": 1.0, "환경": 1.0,
                "문제": 1.0, "오류": 1.0, "장애": 1.0,
                "해결": 1.0, "수정": 1.0, "개선": 1.0
            },
            "technical": {
                # IT/기술 도메인
                "API": 1.2, "인터페이스": 1.2, "프로토콜": 1.2,
                "데이터베이스": 1.2, "DB": 1.2, "쿼리": 1.2,
                "네트워크": 1.2, "서버": 1.2, "클라이언트": 1.2,
                "보안": 1.2, "인증": 1.2, "암호화": 1.2,
                "성능": 1.2, "최적화": 1.2, "캐싱": 1.2,
                "백업": 1.2, "복구": 1.2, "동기화": 1.2,
                "배포": 1.2, "버전": 1.2, "릴리즈": 1.2
            },
            "business": {
                # 비즈니스 도메인
                "매출": 1.2, "수익": 1.2, "비용": 1.2,
                "고객": 1.2, "서비스": 1.2, "제품": 1.2,
                "마케팅": 1.2, "판매": 1.2, "운영": 1.2,
                "전략": 1.2, "계획": 1.2, "목표": 1.2,
                "성과": 1.2, "지표": 1.2, "분석": 1.2,
                "리스크": 1.2, "규정": 1.2, "정책": 1.2
            },
            "academic": {
                # 학술 도메인
                "연구": 1.2, "분석": 1.2, "조사": 1.2,
                "데이터": 1.2, "샘플": 1.2, "통계": 1.2,
                "결과": 1.2, "결론": 1.2, "가설": 1.2,
                "방법론": 1.2, "이론": 1.2, "모델": 1.2,
                "검증": 1.2, "실험": 1.2, "평가": 1.2
            }
        }
        
        return domain_keywords.get(domain, domain_keywords["general"])
    
    def _load_synonym_dictionary(self) -> Dict[str, List[str]]:
        """
        동의어 사전 로드
        
        Returns:
            동의어 사전
        """
        return {
            # 일반 용어
            "방법": ["방식", "기법", "기술", "수단", "절차"],
            "과정": ["절차", "단계", "순서", "진행", "절차"],
            "결과": ["성과", "효과", "결과물", "산출물", "성과"],
            "문제": ["이슈", "과제", "해결사항", "장애", "오류"],
            "개선": ["향상", "발전", "고도화", "최적화", "개선"],
            "분석": ["검토", "조사", "연구", "평가", "분석"],
            
            # IT 용어
            "시스템": ["플랫폼", "솔루션", "도구", "시스템", "애플리케이션"],
            "데이터": ["정보", "자료", "내용", "데이터", "파일"],
            "관리": ["운영", "유지보수", "관리", "제어"],
            "보안": ["안전", "보호", "안전성", "보안", "인증"],
            "성능": ["효율", "속도", "품질", "성능", "처리속도"],
            
            # 비즈니스 용어
            "비용": ["금액", "가격", "지출", "비용", "경비"],
            "시간": ["기간", "소요시간", "기한", "시간", "기간"],
            "사용자": ["고객", "이용자", "사용자", "사용자"],
            "기능": ["특성", "역할", "작용", "기능", "서비스"],
            "구조": ["체계", "구성", "설계", "구조", "아키텍처"],
            
            # 환경 및 설정
            "환경": ["조건", "상황", "배경", "환경", "설정"],
            "요구사항": ["필요사항", "요구", "필요", "요구사항"],
            "정책": ["규정", "지침", "방침", "정책", "규칙"],
            "절차": ["순서", "과정", "단계", "절차", "방법"]
        }
    
    def _load_abbreviation_dictionary(self) -> Dict[str, str]:
        """
        약어 사전 로드
        
        Returns:
            약어 사전
        """
        return {
            # IT 약어
            "API": "Application Programming Interface",
            "DB": "데이터베이스",
            "UI": "사용자 인터페이스",
            "UX": "사용자 경험",
            "OS": "운영체제",
            "CPU": "중앙처리장치",
            "RAM": "메모리",
            "HDD": "하드디스크",
            "SSD": "솔리드스테이트드라이브",
            "LAN": "근거리통신망",
            "WAN": "광역통신망",
            "VPN": "가상사설망",
            "HTTP": "하이퍼텍스트 전송 프로토콜",
            "HTTPS": "보안 하이퍼텍스트 전송 프로토콜",
            "FTP": "파일 전송 프로토콜",
            "SMTP": "간이 메일 전송 프로토콜",
            "POP3": "메일 수신 프로토콜",
            "IMAP": "인터넷 메시지 접근 프로토콜",
            
            # 비즈니스 약어
            "CEO": "최고경영자",
            "CFO": "최고재무책임자",
            "CTO": "최고기술책임자",
            "HR": "인사",
            "IT": "정보기술",
            "R&D": "연구개발",
            "QA": "품질보증",
            "QC": "품질관리",
            "KPI": "핵심성과지표",
            "ROI": "투자수익률",
            "B2B": "기업간 거래",
            "B2C": "기업과 소비자 간 거래",
            "CRM": "고객관계관리",
            "ERP": "전사적 자원관리",
            "SaaS": "서비스형 소프트웨어",
            "PaaS": "서비스형 플랫폼",
            "IaaS": "서비스형 인프라"
        }
    
    def enhance_keywords(self, keywords: List[str]) -> List[str]:
        """
        키워드 향상 (확장 및 정규화)
        
        Args:
            keywords: 원본 키워드 리스트
            
        Returns:
            향상된 키워드 리스트
        """
        enhanced_keywords = []
        
        for keyword in keywords:
            # 1. 키워드 정규화
            normalized = self._normalize_keyword(keyword)
            enhanced_keywords.append(normalized)
            
            # 2. 동의어 확장
            synonyms = self._get_synonyms(normalized)
            enhanced_keywords.extend(synonyms)
            
            # 3. 약어 확장
            abbreviation = self._expand_abbreviation(normalized)
            if abbreviation:
                enhanced_keywords.append(abbreviation)
        
        # 중복 제거 및 정렬
        unique_keywords = list(set(enhanced_keywords))
        unique_keywords.sort(key=len, reverse=True)
        
        return unique_keywords
    
    def _normalize_keyword(self, keyword: str) -> str:
        """
        키워드 정규화
        
        Args:
            keyword: 원본 키워드
            
        Returns:
            정규화된 키워드
        """
        # 소문자 변환
        normalized = keyword.lower()
        
        # 특수문자 제거
        normalized = re.sub(r'[^\w가-힣]', '', normalized)
        
        # 연속된 공백 제거
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _get_synonyms(self, keyword: str) -> List[str]:
        """
        키워드의 동의어 반환
        
        Args:
            keyword: 키워드
            
        Returns:
            동의어 리스트
        """
        return self.synonym_dict.get(keyword, [])
    
    def _expand_abbreviation(self, keyword: str) -> Optional[str]:
        """
        약어 확장
        
        Args:
            keyword: 키워드
            
        Returns:
            확장된 약어 (없으면 None)
        """
        return self.abbreviation_dict.get(keyword.upper())
    
    def calculate_keyword_weight(self, keyword: str, context: str = "") -> float:
        """
        키워드 가중치 계산
        
        Args:
            keyword: 키워드
            context: 컨텍스트 (선택사항)
            
        Returns:
            가중치 점수
        """
        weight = 1.0
        
        # 1. 도메인 키워드 가중치
        if keyword in self.domain_keywords:
            weight *= self.domain_keywords[keyword]
        
        # 2. 컨텍스트에서의 빈도
        if context:
            frequency = context.lower().count(keyword.lower())
            weight *= (1.0 + frequency * 0.1)
        
        # 3. 키워드 길이 가중치 (긴 키워드가 더 중요)
        weight *= (1.0 + len(keyword) * 0.05)
        
        return weight
    
    def recommend_keywords(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        텍스트에서 키워드 추천
        
        Args:
            text: 분석할 텍스트
            max_keywords: 최대 추천 키워드 수
            
        Returns:
            (키워드, 가중치) 튜플 리스트
        """
        # 1. 텍스트에서 단어 추출
        words = re.findall(r'\b\w{2,}\b', text.lower())
        
        # 2. 단어 빈도 계산
        word_freq = Counter(words)
        
        # 3. 키워드 점수 계산
        keyword_scores = []
        for word, freq in word_freq.items():
            # 기본 점수
            score = freq
            
            # 도메인 키워드 가중치 적용
            weight = self.calculate_keyword_weight(word, text)
            score *= weight
            
            keyword_scores.append((word, score))
        
        # 4. 점수로 정렬하고 상위 키워드 반환
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores[:max_keywords]
    
    def extract_domain_specific_keywords(self, text: str) -> List[str]:
        """
        도메인 특화 키워드 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            도메인 특화 키워드 리스트
        """
        domain_keywords = []
        text_lower = text.lower()
        
        for keyword in self.domain_keywords.keys():
            if keyword in text_lower:
                domain_keywords.append(keyword)
        
        return domain_keywords
    
    def create_keyword_index(self, documents: List[str]) -> Dict[str, List[int]]:
        """
        문서 집합에서 키워드 인덱스 생성
        
        Args:
            documents: 문서 리스트
            
        Returns:
            키워드별 문서 인덱스
        """
        keyword_index = {}
        
        for doc_idx, document in enumerate(documents):
            # 문서에서 키워드 추출
            recommended_keywords = self.recommend_keywords(document, max_keywords=20)
            
            for keyword, _ in recommended_keywords:
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(doc_idx)
        
        return keyword_index
    
    def find_similar_keywords(self, target_keyword: str, 
                            keyword_candidates: List[str]) -> List[Tuple[str, float]]:
        """
        유사한 키워드 찾기
        
        Args:
            target_keyword: 대상 키워드
            keyword_candidates: 후보 키워드들
            
        Returns:
            (키워드, 유사도) 튜플 리스트
        """
        similarities = []
        target_normalized = self._normalize_keyword(target_keyword)
        
        for candidate in keyword_candidates:
            candidate_normalized = self._normalize_keyword(candidate)
            
            # 1. 정확한 매칭
            if target_normalized == candidate_normalized:
                similarities.append((candidate, 1.0))
                continue
            
            # 2. 부분 문자열 매칭
            if (len(target_normalized) >= 3 and len(candidate_normalized) >= 3 and
                (target_normalized in candidate_normalized or 
                 candidate_normalized in target_normalized)):
                similarity = min(len(target_normalized), len(candidate_normalized)) / \
                           max(len(target_normalized), len(candidate_normalized))
                similarities.append((candidate, similarity))
                continue
            
            # 3. 동의어 매칭
            target_synonyms = self._get_synonyms(target_normalized)
            if candidate_normalized in target_synonyms:
                similarities.append((candidate, 0.8))
                continue
        
        # 유사도로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities

# 사용 예시
if __name__ == "__main__":
    enhancer = KeywordEnhancer(domain="technical")
    
    # 키워드 향상 테스트
    original_keywords = ["시스템", "API", "데이터"]
    enhanced = enhancer.enhance_keywords(original_keywords)
    print(f"향상된 키워드: {enhanced}")
    
    # 키워드 추천 테스트
    text = "시스템 API를 통해 데이터를 처리하고 성능을 개선합니다."
    recommended = enhancer.recommend_keywords(text)
    print(f"추천 키워드: {recommended}")
