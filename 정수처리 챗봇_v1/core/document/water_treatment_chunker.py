"""
정수처리 도메인 특화 청킹 모듈

정수처리 공정별 의미 단위로 텍스트를 청킹하고,
슬라이딩 윈도우 기법으로 문맥 손실을 최소화합니다.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .text_chunk import TextChunk

logger = logging.getLogger(__name__)

@dataclass
class WaterTreatmentChunkingConfig:
    """정수처리 청킹 설정"""
    max_chunk_size: int = 512  # 기존 256자에서 512자로 확대
    min_chunk_size: int = 100   # 최소 청크 크기
    overlap_ratio: float = 0.25  # 25% 오버랩 (20-30% 범위)
    process_based: bool = True   # 공정 기준 청킹 여부
    preserve_context: bool = True  # 문맥 보존 여부

class WaterTreatmentChunker:
    """정수처리 도메인 특화 청킹 클래스"""
    
    def __init__(self, config: Optional[WaterTreatmentChunkingConfig] = None):
        """청킹기 초기화"""
        self.config = config or WaterTreatmentChunkingConfig()
        
        # 향상된 정수처리 공정별 패턴 정의 (도메인 분류 기반)
        self.process_patterns = self._load_enhanced_patterns()
        
        logger.info(f"정수처리 청킹기 초기화 완료 (최대 크기: {self.config.max_chunk_size}자, 오버랩: {self.config.overlap_ratio*100:.1f}%)")
    
    def _load_enhanced_patterns(self) -> Dict[str, Dict[str, Any]]:
        """향상된 패턴 로드"""
        try:
            import json
            import os
            
            pattern_file = "config/enhanced_patterns/enhanced_chunking_patterns.json"
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 패턴 구조 변환
                enhanced_patterns = {}
                for category, domains in data.get('patterns', {}).items():
                    for domain_name, domain_info in domains.items():
                        enhanced_patterns[domain_name] = {
                            'keywords': domain_info.get('keywords', []),
                            'patterns': domain_info.get('patterns', []),
                            'weight': domain_info.get('weight', 1.0)
                        }
                
                logger.info(f"향상된 청킹 패턴 로드 완료: {len(enhanced_patterns)}개 도메인")
                return enhanced_patterns
            else:
                logger.warning("향상된 패턴 파일이 없습니다. 기본 패턴을 사용합니다.")
                return self._get_default_patterns()
                
        except Exception as e:
            logger.error(f"향상된 패턴 로드 실패: {e}. 기본 패턴을 사용합니다.")
            return self._get_default_patterns()
    
    def _get_default_patterns(self) -> Dict[str, Dict[str, Any]]:
        """기본 패턴 반환"""
        return {
            'intake': {
                'keywords': ['착수', '수위', '목표값', '유입량', '정수지', 'k-means', '군집분석'],
                'patterns': [r'착수\s*공정', r'수위\s*제어', r'유입량\s*조절'],
                'weight': 1.0
            },
            'coagulation': {
                'keywords': ['약품', '응집제', '주입률', 'n-beats', '탁도', '알칼리도'],
                'patterns': [r'약품\s*공정', r'응집제\s*주입', r'n-beats\s*모델'],
                'weight': 1.2
            },
            'mixing_flocculation': {
                'keywords': ['혼화', '응집', '회전속도', 'rpm', '교반', 'g값'],
                'patterns': [r'혼화\s*응집\s*공정', r'교반\s*속도', r'회전속도\s*제어'],
                'weight': 1.0
            },
            'sedimentation': {
                'keywords': ['침전', '슬러지', '발생량', '수집기', '대차'],
                'patterns': [r'침전\s*공정', r'슬러지\s*처리', r'수집기\s*운전'],
                'weight': 1.0
            },
            'filtration': {
                'keywords': ['여과', '여과지', '세척', '주기', '운전'],
                'patterns': [r'여과\s*공정', r'여과지\s*세척', r'역세척\s*주기'],
                'weight': 1.0
            },
            'disinfection': {
                'keywords': ['소독', '염소', '잔류염소', '주입률', '체류시간'],
                'patterns': [r'소독\s*공정', r'염소\s*주입', r'잔류염소\s*농도'],
                'weight': 1.0
            },
            'ems': {
                'keywords': ['ems', '펌프', '제어', '전력', '피크', '에너지'],
                'patterns': [r'ems\s*시스템', r'에너지\s*관리', r'전력\s*피크'],
                'weight': 1.1
            },
            'pms': {
                'keywords': ['pms', '모터', '진단', '전류', '진동', '온도'],
                'patterns': [r'pms\s*시스템', r'모터\s*진단', r'예방\s*정비'],
                'weight': 1.1
            }
        }
    
    def chunk_text(self, text: str, pdf_id: str = None, strategy: str = "hybrid") -> List[TextChunk]:
        """
        정수처리 도메인 특화 텍스트 청킹
        
        Args:
            text: 원본 텍스트
            pdf_id: PDF 식별자
            strategy: 청킹 전략 ("process_based", "sliding_window", "hybrid")
            
        Returns:
            TextChunk 리스트
        """
        if not text.strip():
            return []
        
        logger.info(f"정수처리 청킹 시작: 전략={strategy}, 텍스트 길이={len(text)}자")
        
        if strategy == "process_based":
            return self._chunk_by_process(text, pdf_id)
        elif strategy == "sliding_window":
            return self._chunk_sliding_window(text, pdf_id)
        else:  # hybrid
            return self._chunk_hybrid(text, pdf_id)
    
    def _chunk_by_process(self, text: str, pdf_id: str = None) -> List[TextChunk]:
        """공정 기준 청킹"""
        chunks = []
        paragraphs = self._split_by_paragraphs(text)
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip() or len(paragraph.strip()) < self.config.min_chunk_size:
                continue
            
            # 공정 유형 식별
            process_type = self._identify_process_type(paragraph)
            
            # 공정별로 청크 분할
            if process_type and len(paragraph) > self.config.max_chunk_size:
                sub_chunks = self._split_long_process_section(paragraph, process_type, pdf_id, para_idx)
                chunks.extend(sub_chunks)
            else:
                # 단일 청크로 처리
                chunk_id = f"{pdf_id}_process_{para_idx}" if pdf_id else f"process_{para_idx}"
                
                metadata = {
                    "pdf_id": pdf_id,
                    "chunk_index": len(chunks),
                    "chunk_type": "process_based",
                    "process_type": process_type or "general",
                    "process_keywords": self._extract_process_keywords(paragraph, process_type)
                }
                
                chunks.append(TextChunk(
                    content=paragraph.strip(),
                    page_number=1,  # 페이지 번호는 별도 처리 필요
                    chunk_id=chunk_id,
                    pdf_id=pdf_id,
                    metadata=metadata
                ))
        
        logger.info(f"공정 기준 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def _chunk_sliding_window(self, text: str, pdf_id: str = None) -> List[TextChunk]:
        """슬라이딩 윈도우 청킹 (20-30% 오버랩)"""
        chunks = []
        
        # 문장 단위로 분할
        sentences = self._split_by_sentences(text)
        if not sentences:
            return chunks
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        overlap_size = int(self.config.max_chunk_size * self.config.overlap_ratio)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 청크 크기 체크
            if current_length + len(sentence) > self.config.max_chunk_size and current_chunk:
                # 현재 청크 저장
                chunk_content = ' '.join(current_chunk)
                if len(chunk_content.strip()) >= self.config.min_chunk_size:
                    chunk_id = f"{pdf_id}_sliding_{chunk_index}" if pdf_id else f"sliding_{chunk_index}"
                    
                    # 공정 키워드 추출
                    process_keywords = []
                    for process_name, process_info in self.process_patterns.items():
                        if any(keyword in chunk_content.lower() for keyword in process_info['keywords']):
                            process_keywords.append(process_name)
                    
                    metadata = {
                        "pdf_id": pdf_id,
                        "chunk_index": chunk_index,
                        "chunk_type": "sliding_window",
                        "overlap_ratio": self.config.overlap_ratio,
                        "process_keywords": process_keywords
                    }
                    
                    chunks.append(TextChunk(
                        content=chunk_content.strip(),
                        page_number=1,
                        chunk_id=chunk_id,
                        pdf_id=pdf_id,
                        metadata=metadata
                    ))
                    
                    chunk_index += 1
                
                # 슬라이딩 윈도우: 오버랩 부분 계산
                overlap_text = chunk_content[-overlap_size:] if len(chunk_content) > overlap_size else chunk_content
                overlap_sentences = self._split_by_sentences(overlap_text)
                
                # 문장 경계에서 오버랩 조정
                overlap_sentences = self._adjust_sentence_boundary(overlap_sentences)
                
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        # 마지막 청크 처리
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                chunk_id = f"{pdf_id}_sliding_{chunk_index}" if pdf_id else f"sliding_{chunk_index}"
                
                process_keywords = []
                for process_name, process_info in self.process_patterns.items():
                    if any(keyword in chunk_content.lower() for keyword in process_info['keywords']):
                        process_keywords.append(process_name)
                
                metadata = {
                    "pdf_id": pdf_id,
                    "chunk_index": chunk_index,
                    "chunk_type": "sliding_window",
                    "overlap_ratio": self.config.overlap_ratio,
                    "process_keywords": process_keywords
                }
                
                chunks.append(TextChunk(
                    content=chunk_content.strip(),
                    page_number=1,
                    chunk_id=chunk_id,
                    pdf_id=pdf_id,
                    metadata=metadata
                ))
        
        logger.info(f"슬라이딩 윈도우 청킹 완료: {len(chunks)}개 청크 생성 (오버랩: {self.config.overlap_ratio*100:.1f}%)")
        return chunks
    
    def _chunk_hybrid(self, text: str, pdf_id: str = None) -> List[TextChunk]:
        """하이브리드 청킹 (공정 기준 + 슬라이딩 윈도우)"""
        # 1단계: 공정별로 큰 섹션 분할
        process_sections = self._identify_process_sections(text)
        
        chunks = []
        for section_idx, (section_text, process_type) in enumerate(process_sections):
            if not section_text.strip():
                continue
            
            # 2단계: 각 섹션에 슬라이딩 윈도우 적용
            if len(section_text) > self.config.max_chunk_size:
                section_chunks = self._chunk_sliding_window(section_text, f"{pdf_id}_section_{section_idx}")
                
                # 공정 정보를 메타데이터에 추가
                for chunk in section_chunks:
                    if chunk.metadata:
                        chunk.metadata["parent_process"] = process_type
                        chunk.metadata["section_index"] = section_idx
                
                chunks.extend(section_chunks)
            else:
                # 단일 청크로 처리
                chunk_id = f"{pdf_id}_hybrid_{section_idx}" if pdf_id else f"hybrid_{section_idx}"
                
                metadata = {
                    "pdf_id": pdf_id,
                    "chunk_index": len(chunks),
                    "chunk_type": "hybrid",
                    "parent_process": process_type,
                    "section_index": section_idx,
                    "process_keywords": self._extract_process_keywords(section_text, process_type)
                }
                
                chunks.append(TextChunk(
                    content=section_text.strip(),
                    page_number=1,
                    chunk_id=chunk_id,
                    pdf_id=pdf_id,
                    metadata=metadata
                ))
        
        logger.info(f"하이브리드 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """문단 단위로 텍스트 분할"""
        # 빈 줄 기준으로 문단 분할
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """문장 단위로 텍스트 분할"""
        # 한국어 문장 분할 패턴
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z가-힣])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _identify_process_type(self, text: str) -> Optional[str]:
        """텍스트에서 정수처리 공정 유형 식별"""
        text_lower = text.lower()
        
        # 각 공정별 매칭 점수 계산
        process_scores = {}
        for process_name, process_info in self.process_patterns.items():
            score = 0
            
            # 키워드 매칭
            for keyword in process_info['keywords']:
                if keyword in text_lower:
                    score += 1
            
            # 패턴 매칭
            for pattern in process_info['patterns']:
                if re.search(pattern, text_lower):
                    score += 2  # 패턴 매칭에 더 높은 가중치
            
            if score > 0:
                process_scores[process_name] = score
        
        # 최고 점수 공정 반환
        if process_scores:
            return max(process_scores, key=process_scores.get)
        
        return None
    
    def _identify_process_sections(self, text: str) -> List[Tuple[str, str]]:
        """텍스트를 공정별 섹션으로 분할"""
        paragraphs = self._split_by_paragraphs(text)
        sections = []
        current_section = []
        current_process = None
        
        for paragraph in paragraphs:
            process_type = self._identify_process_type(paragraph)
            
            if process_type and process_type != current_process:
                # 새로운 공정 시작
                if current_section:
                    sections.append(('\n\n'.join(current_section), current_process or "general"))
                
                current_section = [paragraph]
                current_process = process_type
            else:
                current_section.append(paragraph)
        
        # 마지막 섹션 추가
        if current_section:
            sections.append(('\n\n'.join(current_section), current_process or "general"))
        
        return sections
    
    def _extract_process_keywords(self, text: str, process_type: Optional[str] = None) -> List[str]:
        """텍스트에서 공정별 키워드 추출"""
        text_lower = text.lower()
        keywords = []
        
        if process_type and process_type in self.process_patterns:
            # 특정 공정의 키워드 확인
            for keyword in self.process_patterns[process_type]['keywords']:
                if keyword in text_lower:
                    keywords.append(keyword)
        else:
            # 모든 공정에서 키워드 확인
            for process_info in self.process_patterns.values():
                for keyword in process_info['keywords']:
                    if keyword in text_lower and keyword not in keywords:
                        keywords.append(keyword)
        
        return keywords
    
    def _adjust_sentence_boundary(self, sentences: List[str]) -> List[str]:
        """문장 경계에서 오버랩 조정"""
        if not sentences:
            return sentences
        
        # 완전한 문장으로 끝나는 지점 찾기
        sentence_endings = ['.', '!', '?', '다.', '요.', '음.', '됨.', '함.', '임.']
        
        for i in range(len(sentences) - 1, -1, -1):
            sentence = sentences[i].strip()
            if any(sentence.endswith(ending) for ending in sentence_endings):
                return sentences[max(0, i-1):]  # 이전 문장부터 포함
        
        # 적절한 경계를 찾지 못한 경우 전체 반환
        return sentences
    
    def _split_long_process_section(self, text: str, process_type: str, pdf_id: str, section_idx: int) -> List[TextChunk]:
        """긴 공정 섹션을 여러 청크로 분할"""
        # 슬라이딩 윈도우 방식으로 분할
        temp_chunker = WaterTreatmentChunker(self.config)
        chunks = temp_chunker._chunk_sliding_window(text, f"{pdf_id}_proc_{process_type}_{section_idx}")
        
        # 공정 정보를 메타데이터에 추가
        for chunk in chunks:
            if chunk.metadata:
                chunk.metadata["parent_process"] = process_type
                chunk.metadata["is_split_section"] = True
        
        return chunks
