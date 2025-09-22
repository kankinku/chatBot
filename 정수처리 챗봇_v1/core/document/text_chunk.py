"""
텍스트 청크 데이터 클래스

순환 import 문제를 해결하기 위해 별도 모듈로 분리
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class TextChunk:
    """텍스트 청크 데이터 클래스"""
    content: str
    page_number: int
    chunk_id: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[dict] = None
    pdf_id: Optional[str] = None
    filename: Optional[str] = None
    upload_time: Optional[str] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            'content': self.content,
            'page_number': self.page_number,
            'chunk_id': self.chunk_id,
            'metadata': self.metadata,
            'pdf_id': self.pdf_id,
            'filename': self.filename,
            'upload_time': self.upload_time
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TextChunk':
        """딕셔너리에서 생성"""
        return cls(
            content=data['content'],
            page_number=data['page_number'],
            chunk_id=data['chunk_id'],
            metadata=data.get('metadata', {}),
            pdf_id=data.get('pdf_id'),
            filename=data.get('filename'),
            upload_time=data.get('upload_time')
        )
