"""
법률 문서 스키마 및 정규화 모듈

법률 문서의 표준 스키마를 정의하고, 조문 참조 및 약칭을 정규화합니다.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class LegalDocumentType(Enum):
    """법률 문서 유형"""
    LAW = "law"  # 법률
    DECREE = "decree"  # 시행령
    RULE = "rule"  # 시행규칙
    ORDINANCE = "ordinance"  # 조례
    REGULATION = "regulation"  # 규정

@dataclass
class LegalMetadata:
    """법률 문서 메타데이터"""
    law_id: str  # 법률 고유 ID
    law_title: str  # 법률명
    article_no: Optional[str] = None  # 조 번호
    clause_no: Optional[str] = None  # 항 번호
    paragraph_no: Optional[str] = None  # 호 번호
    effective_date: Optional[str] = None  # 시행일
    aliases: List[str] = None  # 별칭/약칭
    section_hierarchy: Optional[str] = None  # 장/절/관 구조
    source_path: Optional[str] = None  # 원본 파일 경로
    source_span: Optional[Dict[str, int]] = None  # 원본 위치 (start, end)
    document_type: LegalDocumentType = LegalDocumentType.LAW
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.source_span is None:
            self.source_span = {"start": 0, "end": 0}

@dataclass
class LegalChunk:
    """법률 문서 청크"""
    text: str  # 본문 텍스트
    metadata: LegalMetadata  # 메타데이터
    chunk_id: Optional[str] = None  # 청크 고유 ID
    keywords: List[str] = None  # 추출된 키워드
    embedding: Optional[Any] = None  # 임베딩 벡터
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.chunk_id is None:
            self.chunk_id = f"{self.metadata.law_id}_{self.metadata.article_no}_{self.metadata.clause_no}"

@dataclass
class LegalDocument:
    """법률 문서"""
    law_id: str  # 법률 고유 ID
    title: str  # 법률명
    content: str  # 전체 내용
    chunks: List[LegalChunk]  # 청크 목록
    metadata: LegalMetadata  # 문서 메타데이터
    created_at: Optional[str] = None  # 생성일시
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class LegalNormalizer:
    """법률 문서 정규화 클래스"""
    
    def __init__(self):
        """정규화기 초기화"""
        # 조문 참조 정규화 패턴
        self.article_patterns = [
            r'제(\d+)조',  # 제10조
            r'(\d+)조',    # 10조
            r'제(\d+)조의(\d+)',  # 제10조의2
            r'(\d+)조의(\d+)',    # 10조의2
        ]
        
        # 항 참조 정규화 패턴
        self.clause_patterns = [
            r'제(\d+)항',  # 제2항
            r'(\d+)항',    # 2항
        ]
        
        # 호 참조 정규화 패턴
        self.paragraph_patterns = [
            r'제(\d+)호',  # 제1호
            r'(\d+)호',    # 1호
        ]
        
        # 법률 별칭 사전 (확장)
        self.aliases_dict = {
            '공정위': '공정거래위원회',
            '개인정보보호법': '개인정보 보호법',
            '정보통신망법': '정보통신망 이용촉진 및 정보보호 등에 관한 법률',
            '근로기준법': '근로기준법',
            '산업안전보건법': '산업안전보건법',
            '환경보전법': '환경정책기본법',
            '근기법': '근로기준법',
            '노동법': '근로기준법',
            '근로법': '근로기준법',
            '개보법': '개인정보 보호법',
            '정보보호법': '개인정보 보호법',
            '공정거래법': '독점규제 및 공정거래에 관한 법률',
            '독점규제법': '독점규제 및 공정거래에 관한 법률',
            '형사법': '형법',
            '민사법': '민법',
            '행정법': '행정기본법',
            '환경법': '환경정책기본법',
            '저작권보호법': '저작권법',
            '특허보호법': '특허법',
            '건설법': '건설산업기본법',
            '건설업법': '건설산업기본법'
        }
        
        # 법률 도메인 분류 (확장된 키워드)
        self.domain_keywords = {
            'labor': [
                '근로', '노동', '임금', '급여', '휴가', '해고', '퇴직', '산재', 
                '근무시간', '야근', '휴일근무', '최저임금', '연차', '육아휴직',
                '직장', '회사', '사업주', '근로자', '직원', '고용'
            ],
            'privacy': [
                '개인정보', '프라이버시', '정보보호', '수집', '이용', '제공', '동의',
                '민감정보', '고유식별정보', '개인정보처리방침', '정보주체', 
                '개인정보보호위원회', '개인정보처리자'
            ],
            'environment': [
                '환경', '오염', '배출', '폐기물', '대기', '수질', '토양', '소음',
                '환경영향평가', '환경보전', '공해', '친환경', '온실가스'
            ],
            'commerce': [
                '상거래', '공정거래', '독점', '경쟁', '소비자', '약관', '광고',
                '부당거래', '시장지배적지위', '담합', '불공정거래행위'
            ],
            'criminal': [
                '형사', '범죄', '처벌', '벌금', '징역', '형법', '고발', '고소',
                '수사', '기소', '재판', '판결', '선고', '실형', '집행유예'
            ],
            'civil': [
                '민사', '계약', '손해배상', '소유권', '민법', '채권', '채무',
                '불법행위', '계약위반', '손해', '배상', '소송', '민사소송'
            ],
            'administrative': [
                '행정', '허가', '신고', '처분', '행정법', '행정기관', '공무원',
                '행정처분', '행정소송', '행정심판', '취소', '정지', '과태료'
            ],
            'intellectual_property': [
                '지적재산권', '특허', '상표', '저작권', '디자인', '발명',
                '특허청', '저작물', '침해', '등록', '출원'
            ],
            'tax': [
                '세금', '세무', '국세', '지방세', '소득세', '법인세', '부가가치세',
                '세무서', '납세', '과세', '세율', '공제', '감면'
            ],
            'construction': [
                '건설', '건축', '시공', '설계', '건설업', '건축법', '건설기술',
                '안전관리', '품질관리', '건설공사'
            ]
        }
    
    def normalize_article_reference(self, text: str) -> str:
        """조문 참조 정규화"""
        normalized = text
        
        # 조 참조 정규화
        for pattern in self.article_patterns:
            matches = re.findall(pattern, normalized)
            for match in matches:
                if isinstance(match, tuple):
                    # 제10조의2 형태
                    original = f"제{match[0]}조의{match[1]}"
                    standard = f"제{match[0]}조의{match[1]}"
                else:
                    # 제10조 형태
                    original = f"제{match}조"
                    standard = f"제{match}조"
                normalized = normalized.replace(original, standard)
        
        return normalized
    
    def normalize_aliases(self, text: str) -> str:
        """별칭 및 약칭 정규화"""
        normalized = text
        
        for alias, full_name in self.aliases_dict.items():
            normalized = normalized.replace(alias, full_name)
        
        return normalized
    
    def extract_legal_references(self, text: str) -> Dict[str, List[str]]:
        """법률 참조 정보 추출"""
        references = {
            'articles': [],
            'clauses': [],
            'paragraphs': []
        }
        
        # 조 참조 추출
        for pattern in self.article_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    references['articles'].append(f"제{match[0]}조의{match[1]}")
                else:
                    references['articles'].append(f"제{match}조")
        
        # 항 참조 추출
        for pattern in self.clause_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                references['clauses'].append(f"제{match}항")
        
        # 호 참조 추출
        for pattern in self.paragraph_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                references['paragraphs'].append(f"제{match}호")
        
        return references
    
    def classify_legal_domain(self, text: str) -> List[str]:
        """법률 도메인 분류"""
        domains = []
        text_lower = text.lower()
        
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ['general']
    
    def normalize_legal_text(self, text: str) -> Dict[str, Any]:
        """법률 텍스트 종합 정규화"""
        result = {
            'normalized_text': text,
            'references': {},
            'domains': [],
            'aliases_applied': []
        }
        
        # 별칭 정규화
        normalized_text = self.normalize_aliases(text)
        applied_aliases = []
        for alias, full_name in self.aliases_dict.items():
            if alias in text and alias not in normalized_text:
                applied_aliases.append(f"{alias} -> {full_name}")
        
        # 조문 참조 정규화
        normalized_text = self.normalize_article_reference(normalized_text)
        
        # 참조 정보 추출
        references = self.extract_legal_references(normalized_text)
        
        # 도메인 분류
        domains = self.classify_legal_domain(normalized_text)
        
        result.update({
            'normalized_text': normalized_text,
            'references': references,
            'domains': domains,
            'aliases_applied': applied_aliases
        })
        
        return result
