"""
질문-답변 로깅 시스템

각 질문에 대한 답변 생성 시 다음 정보를 로그 파일에 저장:
- 입력된 질문
- 생성된 답변  
- 사용된 청크
- 청크의 PDF 파일명/페이지 등 위치 메타데이터
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class QALogEntry:
    """질문-답변 로그 엔트리"""
    timestamp: str
    question: str
    answer: str
    chunks_used: List[Dict[str, Any]]
    generation_time_ms: float
    confidence_score: float
    model_name: str
    pipeline_type: str
    session_id: Optional[str] = None
    # 질문 분석 정보 추가
    question_analysis: Optional[Dict[str, Any]] = None

class QALogger:
    """질문-답변 로거"""
    
    def __init__(self, log_file_path: str = "logs/qa_log.jsonl"):
        self.log_file_path = Path(log_file_path)
        self._ensure_log_directory()
        
        # 로그 파일 핸들러 설정
        self.logger = logging.getLogger("qa_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # JSONL 파일 핸들러
        file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        
        # 로그 포맷터 (JSON 형태로 저장)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
    
    def _ensure_log_directory(self):
        """로그 디렉토리 생성"""
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _extract_chunk_info(self, chunks) -> List[Dict[str, Any]]:
        """청크에서 정보 추출"""
        chunk_info = []
        
        for chunk in chunks:
            info = {
                "chunk_id": getattr(chunk, 'chunk_id', None),
                "content_preview": getattr(chunk, 'content', '') if getattr(chunk, 'content', '') else "",
                "page_number": getattr(chunk, 'page_number', None),
                "filename": getattr(chunk, 'filename', None),
                "pdf_id": getattr(chunk, 'pdf_id', None),
                "upload_time": getattr(chunk, 'upload_time', None),
                "metadata": getattr(chunk, 'metadata', {})
            }
            
            # measurements 메타데이터가 있으면 포함
            if info["metadata"] and "measurements" in info["metadata"]:
                info["measurements"] = info["metadata"]["measurements"]
            
            chunk_info.append(info)
        
        return chunk_info
    
    def log_qa(self, 
               question: str,
               answer: str, 
               chunks_used: List,
               generation_time_ms: float,
               confidence_score: float,
               model_name: str,
               pipeline_type: str,
               session_id: Optional[str] = None,
               question_analysis: Optional[Dict[str, Any]] = None):
        """질문-답변 로그 기록"""
        
        # 청크 정보 추출
        chunk_info = self._extract_chunk_info(chunks_used)
        
        # 로그 엔트리 생성
        log_entry = QALogEntry(
            timestamp=datetime.now().isoformat(),
            question=question,
            answer=answer,
            chunks_used=chunk_info,
            generation_time_ms=generation_time_ms,
            confidence_score=confidence_score,
            model_name=model_name,
            pipeline_type=pipeline_type,
            session_id=session_id,
            question_analysis=question_analysis
        )
        
        # JSON 형태로 로그 파일에 기록 (가독성 개선)
        try:
            log_data = asdict(log_entry)
            
            # 파일에 직접 쓰기 (로거 대신)
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                # 구분선과 질문 번호 추가
                f.write('\n' + '='*100 + '\n')
                f.write(f'질문 #{log_entry.session_id or "unknown"} - {log_entry.timestamp}\n')
                f.write('='*100 + '\n\n')
                
                # JSON을 들여쓰기와 함께 저장
                formatted_json = json.dumps(log_data, ensure_ascii=False, indent=2, default=str)
                f.write(formatted_json + '\n\n')
                
                # 하단 구분선
                f.write('-'*100 + '\n')
                f.flush()  # 즉시 디스크에 쓰기
                
        except Exception as e:
            # 로깅 실패 시 표준 에러로 출력
            print(f"QA 로그 기록 실패: {e}", file=sys.stderr)

# 전역 QA 로거 인스턴스
qa_logger = QALogger()

# 편의 함수
def log_question_answer(question: str, answer: str, chunks_used: List, 
                       generation_time_ms: float, confidence_score: float,
                       model_name: str, pipeline_type: str, session_id: Optional[str] = None,
                       question_analysis: Optional[Dict[str, Any]] = None):
    """질문-답변 로그 기록 편의 함수"""
    qa_logger.log_qa(question, answer, chunks_used, generation_time_ms, 
                    confidence_score, model_name, pipeline_type, session_id, question_analysis)

import sys

