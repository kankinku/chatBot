"""
Data Types

모든 데이터 타입을 정의합니다 (One Source of Truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class Chunk:
    """
    텍스트 청크
    
    문서를 청킹한 기본 단위입니다.
    """
    doc_id: str                          # 문서 ID
    filename: str                        # 파일명
    page: Optional[int]                  # 페이지 번호 (없으면 None)
    start_offset: int                    # 시작 오프셋
    length: int                          # 길이
    text: str                            # 텍스트 내용
    neighbor_hint: Optional[Tuple[str, Optional[int], int]] = None  # 이웃 청크 힌트
    extra: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터
    
    def __post_init__(self):
        """검증"""
        if not self.text:
            raise ValueError("Chunk text cannot be empty")
        if self.length <= 0:
            raise ValueError("Chunk length must be positive")
        if self.start_offset < 0:
            raise ValueError("Chunk start_offset cannot be negative")


@dataclass
class RetrievedSpan:
    """
    검색된 청크
    
    검색 결과로 반환되는 청크와 점수 정보입니다.
    """
    chunk: Chunk                         # 청크
    source: str                          # 검색 소스 ("vector", "bm25", "hybrid", etc.)
    score: float                         # 검색 점수
    rank: int                            # 순위
    aux_scores: Dict[str, float] = field(default_factory=dict)  # 보조 점수들
    calibrated_conf: Optional[float] = None  # 캘리브레이션된 신뢰도
    
    def __post_init__(self):
        """검증"""
        if self.score < 0:
            raise ValueError("Score cannot be negative")
        if self.rank < 1:
            raise ValueError("Rank must be >= 1")


@dataclass
class Answer:
    """
    최종 답변
    
    파이프라인 실행 결과입니다.
    """
    text: str                            # 답변 텍스트
    confidence: float                    # 신뢰도 (0.0 ~ 1.0)
    sources: List[RetrievedSpan]         # 참조한 소스들
    metrics: Dict[str, Any]              # 성능 메트릭
    fallback_used: str = "none"          # 사용된 폴백 방식
    
    def __post_init__(self):
        """검증"""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (API 응답용)"""
        return {
            "answer": self.text,
            "confidence": self.confidence,
            "sources": [
                {
                    "text": span.chunk.text[:200],
                    "score": span.score,
                    "rank": span.rank,
                    "source": span.source,
                    "filename": span.chunk.filename,
                    "page": span.chunk.page,
                }
                for span in self.sources[:5]  # 상위 5개만
            ],
            "metrics": self.metrics,
            "fallback_used": self.fallback_used,
        }


@dataclass
class QuestionAnalysis:
    """
    질문 분석 결과
    
    질문 유형, 길이, 키워드 등을 분석한 결과입니다.
    """
    qtype: str                          # 질문 유형
    length: int                         # 토큰 길이
    key_token_count: int                # 키워드 토큰 수
    rrf_vector_weight: float            # 벡터 검색 가중치
    rrf_bm25_weight: float              # BM25 검색 가중치
    threshold_adj: float                # 임계값 조정
    has_number: bool = False            # 숫자 포함 여부
    has_unit: bool = False              # 단위 포함 여부
    has_domain_keyword: bool = False    # 도메인 키워드 포함 여부
    
    def __post_init__(self):
        """검증"""
        total_weight = self.rrf_vector_weight + self.rrf_bm25_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"RRF weights must sum to 1.0, got {total_weight}")


@dataclass
class ProcessingMetrics:
    """
    처리 메트릭
    
    파이프라인 각 단계의 성능 메트릭입니다.
    """
    total_time_ms: int = 0
    retrieval_time_ms: int = 0
    rerank_time_ms: int = 0
    generation_time_ms: int = 0
    
    chunks_retrieved: int = 0
    chunks_after_dedup: int = 0
    chunks_after_filter: int = 0
    chunks_used: int = 0
    
    cache_hit: bool = False
    fallback_used: str = "none"
    
    question_type: str = "general"
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "timing": {
                "total_ms": self.total_time_ms,
                "retrieval_ms": self.retrieval_time_ms,
                "rerank_ms": self.rerank_time_ms,
                "generation_ms": self.generation_time_ms,
            },
            "chunks": {
                "retrieved": self.chunks_retrieved,
                "after_dedup": self.chunks_after_dedup,
                "after_filter": self.chunks_after_filter,
                "used": self.chunks_used,
            },
            "meta": {
                "cache_hit": self.cache_hit,
                "fallback_used": self.fallback_used,
                "question_type": self.question_type,
                "confidence": self.confidence,
            },
        }


@dataclass
class DocumentMetadata:
    """
    문서 메타데이터
    
    PDF 문서의 메타 정보입니다.
    """
    doc_id: str
    filename: str
    file_path: str
    total_pages: int
    total_chars: int
    created_at: str
    processed_at: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "total_chars": self.total_chars,
            "created_at": self.created_at,
            "processed_at": self.processed_at,
            **self.extra,
        }

