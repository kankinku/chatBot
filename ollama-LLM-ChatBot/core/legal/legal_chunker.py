"""
법률 문서 청킹 모듈

법률 문서를 조·항 기준으로 청킹하고, 긴 문서의 경우 문단 기준으로 분할합니다.
메타데이터를 부여하여 검색 시 활용할 수 있도록 합니다.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .legal_schema import LegalDocument, LegalChunk, LegalMetadata, LegalNormalizer, LegalDocumentType

import logging
logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """청킹 설정"""
    max_chunk_size: int = 1000  # 최대 청크 크기 (문자 단위)
    min_chunk_size: int = 100   # 최소 청크 크기
    overlap_size: int = 150     # 중복 크기 (10~15%)
    article_based: bool = True  # 조문 기준 청킹 여부
    preserve_structure: bool = True  # 구조 보존 여부

class LegalChunker:
    """법률 문서 청킹 클래스"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """청킹기 초기화"""
        self.config = config or ChunkingConfig()
        self.normalizer = LegalNormalizer()
        
        # 법률 구조 패턴 정의
        self.structure_patterns = {
            'chapter': r'제(\d+)장\s*([^\n]*)',  # 제1장 총칙
            'section': r'제(\d+)절\s*([^\n]*)',  # 제1절 목적
            'article': r'제(\d+)조(?:의(\d+))?\s*\(([^)]+)\)',  # 제10조(목적)
            'clause': r'①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩',  # 항 번호
            'paragraph': r'\d+\.\s*',  # 호 번호
            'subparagraph': r'[가-힣]\.\s*',  # 목 번호
        }
        
        logger.info(f"법률 청킹기 초기화 완료 (최대 크기: {self.config.max_chunk_size}자)")
    
    def extract_document_structure(self, text: str) -> Dict[str, List[Dict]]:
        """문서 구조 추출"""
        structure = {
            'chapters': [],
            'sections': [],
            'articles': [],
            'clauses': []
        }
        
        lines = text.split('\n')
        current_chapter = None
        current_section = None
        current_article = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 장 추출
            chapter_match = re.match(self.structure_patterns['chapter'], line)
            if chapter_match:
                current_chapter = {
                    'number': chapter_match.group(1),
                    'title': chapter_match.group(2).strip(),
                    'line_no': i,
                    'text': line
                }
                structure['chapters'].append(current_chapter)
                continue
            
            # 절 추출
            section_match = re.match(self.structure_patterns['section'], line)
            if section_match:
                current_section = {
                    'number': section_match.group(1),
                    'title': section_match.group(2).strip(),
                    'line_no': i,
                    'text': line,
                    'chapter': current_chapter['number'] if current_chapter else None
                }
                structure['sections'].append(current_section)
                continue
            
            # 조 추출
            article_match = re.match(self.structure_patterns['article'], line)
            if article_match:
                article_no = article_match.group(1)
                sub_no = article_match.group(2) if article_match.group(2) else None
                title = article_match.group(3)
                
                current_article = {
                    'number': article_no,
                    'sub_number': sub_no,
                    'title': title,
                    'line_no': i,
                    'text': line,
                    'chapter': current_chapter['number'] if current_chapter else None,
                    'section': current_section['number'] if current_section else None,
                    'clauses': []
                }
                structure['articles'].append(current_article)
                continue
            
            # 항 추출
            clause_match = re.match(self.structure_patterns['clause'], line)
            if clause_match and current_article:
                clause_info = {
                    'symbol': clause_match.group(),
                    'line_no': i,
                    'text': line,
                    'article_no': current_article['number']
                }
                current_article['clauses'].append(clause_info)
                structure['clauses'].append(clause_info)
        
        return structure
    
    def chunk_by_article(self, text: str, law_id: str, law_title: str) -> List[LegalChunk]:
        """조문 기준 청킹"""
        chunks = []
        structure = self.extract_document_structure(text)
        
        for article in structure['articles']:
            article_text = self._extract_article_text(text, article, structure)
            
            if len(article_text) <= self.config.max_chunk_size:
                # 단일 청크로 처리
                chunk = self._create_chunk(
                    text=article_text,
                    law_id=law_id,
                    law_title=law_title,
                    article=article,
                    structure=structure
                )
                chunks.append(chunk)
            else:
                # 긴 조문을 여러 청크로 분할
                sub_chunks = self._split_long_article(article_text, law_id, law_title, article, structure)
                chunks.extend(sub_chunks)
        
        logger.info(f"조문 기준 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def chunk_by_paragraph(self, text: str, law_id: str, law_title: str) -> List[LegalChunk]:
        """문단 기준 청킹 (슬라이딩 윈도우)"""
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_start_idx = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 현재 청크에 문단 추가 시 크기 확인
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(test_chunk) <= self.config.max_chunk_size:
                current_chunk = test_chunk
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunk = self._create_paragraph_chunk(
                        text=current_chunk,
                        law_id=law_id,
                        law_title=law_title,
                        start_idx=chunk_start_idx,
                        end_idx=i-1
                    )
                    chunks.append(chunk)
                
                # 새 청크 시작 (오버랩 적용)
                current_chunk = paragraph
                chunk_start_idx = max(0, i - 1)  # 이전 문단과 약간 겹치도록
        
        # 마지막 청크 처리
        if current_chunk:
            chunk = self._create_paragraph_chunk(
                text=current_chunk,
                law_id=law_id,
                law_title=law_title,
                start_idx=chunk_start_idx,
                end_idx=len(paragraphs)-1
            )
            chunks.append(chunk)
        
        logger.info(f"문단 기준 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def _extract_article_text(self, full_text: str, article: Dict, structure: Dict) -> str:
        """조문 전체 텍스트 추출"""
        lines = full_text.split('\n')
        start_line = article['line_no']
        
        # 다음 조문까지의 텍스트 추출
        end_line = len(lines)
        for next_article in structure['articles']:
            if next_article['line_no'] > start_line:
                end_line = next_article['line_no']
                break
        
        article_lines = lines[start_line:end_line]
        return '\n'.join(article_lines).strip()
    
    def _split_long_article(self, article_text: str, law_id: str, law_title: str, 
                           article: Dict, structure: Dict) -> List[LegalChunk]:
        """긴 조문을 여러 청크로 분할"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', article_text)
        
        current_chunk = ""
        clause_no = 1
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= self.config.max_chunk_size:
                current_chunk = test_chunk
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunk = self._create_chunk(
                        text=current_chunk,
                        law_id=law_id,
                        law_title=law_title,
                        article=article,
                        structure=structure,
                        clause_no=str(clause_no)
                    )
                    chunks.append(chunk)
                    clause_no += 1
                
                # 새 청크 시작
                current_chunk = sentence
        
        # 마지막 청크 처리
        if current_chunk:
            chunk = self._create_chunk(
                text=current_chunk,
                law_id=law_id,
                law_title=law_title,
                article=article,
                structure=structure,
                clause_no=str(clause_no)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str, law_id: str, law_title: str, 
                     article: Dict, structure: Dict, clause_no: Optional[str] = None) -> LegalChunk:
        """조문 청크 생성"""
        # 정규화 수행
        normalized_result = self.normalizer.normalize_legal_text(text)
        
        # 메타데이터 생성
        metadata = LegalMetadata(
            law_id=law_id,
            law_title=law_title,
            article_no=article['number'],
            clause_no=clause_no,
            section_hierarchy=self._build_hierarchy(article, structure),
            aliases=self.normalizer.aliases_dict.get(law_title, [])
        )
        
        # 청크 생성
        chunk = LegalChunk(
            text=normalized_result['normalized_text'],
            metadata=metadata,
            keywords=self._extract_keywords(text)
        )
        
        return chunk
    
    def _create_paragraph_chunk(self, text: str, law_id: str, law_title: str,
                               start_idx: int, end_idx: int) -> LegalChunk:
        """문단 청크 생성"""
        # 정규화 수행
        normalized_result = self.normalizer.normalize_legal_text(text)
        
        # 메타데이터 생성
        metadata = LegalMetadata(
            law_id=law_id,
            law_title=law_title,
            source_span={"start": start_idx, "end": end_idx}
        )
        
        # 청크 생성
        chunk = LegalChunk(
            text=normalized_result['normalized_text'],
            metadata=metadata,
            keywords=self._extract_keywords(text)
        )
        
        return chunk
    
    def _build_hierarchy(self, article: Dict, structure: Dict) -> str:
        """계층 구조 문자열 생성"""
        hierarchy_parts = []
        
        if article.get('chapter'):
            chapter_title = next((c['title'] for c in structure['chapters'] 
                                if c['number'] == article['chapter']), '')
            hierarchy_parts.append(f"제{article['chapter']}장 {chapter_title}")
        
        if article.get('section'):
            section_title = next((s['title'] for s in structure['sections'] 
                                if s['number'] == article['section']), '')
            hierarchy_parts.append(f"제{article['section']}절 {section_title}")
        
        return " > ".join(hierarchy_parts) if hierarchy_parts else ""
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출 (개선된 버전)"""
        keywords = []
        
        # 1. 법률 용어 추출
        legal_terms = re.findall(r'[가-힣]{2,}(?:법|령|규칙|조례|규정)', text)
        keywords.extend(legal_terms)
        
        # 2. 조문 번호 추출
        article_refs = re.findall(r'제\d+조(?:의\d+)?', text)
        keywords.extend(article_refs)
        
        # 3. 중요 법률 개념 추출
        legal_concepts = re.findall(r'[가-힣]{2,}(?:권|의무|책임|벌금|제재|허가|신고|처분)', text)
        keywords.extend(legal_concepts)
        
        # 4. 중요 명사 (길이 3자 이상)
        important_nouns = re.findall(r'[가-힣]{3,}', text)
        # 빈도 기반 필터링
        noun_freq = {}
        for noun in important_nouns:
            noun_freq[noun] = noun_freq.get(noun, 0) + 1
        
        # 빈도 순으로 정렬하여 상위 8개 선택
        frequent_nouns = sorted(noun_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        keywords.extend([noun for noun, _ in frequent_nouns])
        
        # 5. 중복 제거 및 정리
        unique_keywords = list(set(keywords))
        
        # 6. 너무 짧은 키워드 제거 (2자 미만)
        filtered_keywords = [kw for kw in unique_keywords if len(kw) >= 2]
        
        return filtered_keywords[:15]  # 최대 15개로 제한
    
    def chunk_legal_document(self, document: LegalDocument) -> LegalDocument:
        """법률 문서 청킹"""
        if self.config.article_based:
            chunks = self.chunk_by_article(document.content, document.law_id, document.title)
        else:
            chunks = self.chunk_by_paragraph(document.content, document.law_id, document.title)
        
        # 문서 업데이트
        document.chunks = chunks
        
        logger.info(f"법률 문서 '{document.title}' 청킹 완료: {len(chunks)}개 청크")
        return document
