#!/usr/bin/env python3
"""
Chatbot 로깅 시스템

질문, 분류(SQL/PDF), 생성 결과를 체계적으로 기록하는 로깅 시스템
단계별 처리 로그 기능 추가
"""

import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

class QuestionType(Enum):
    """질문 유형"""
    SQL = "sql"
    PDF = "pdf"
    HYBRID = "hybrid"
    GREETING = "greeting"
    UNKNOWN = "unknown"

class ProcessingStep(Enum):
    """처리 단계"""
    START = "시작"
    SBERT_ROUTING = "SBERT로 질문처리"
    PDF_PIPELINE = "PDF파이프라인 시작"
    SQL_PIPELINE = "SQL파이프라인 시작"
    GREETING_PIPELINE = "인사말처리 시작"
    QUESTION_ANALYSIS = "질문분석"
    VECTOR_SEARCH = "벡터검색"
    CONTEXT_GENERATION = "컨텍스트생성"
    ANSWER_GENERATION = "답변생성"
    SQL_GENERATION = "SQL생성"
    DATABASE_EXECUTION = "데이터베이스실행"
    COMPLETION = "완료"
    ERROR = "오류"

@dataclass
class ChatbotLogEntry:
    """챗봇 로그 엔트리"""
    timestamp: str
    session_id: str
    user_question: str
    question_type: QuestionType
    intent: str
    keywords: List[str]
    processing_time: float
    confidence_score: float
    generated_sql: Optional[str] = None
    generated_answer: Optional[str] = None
    used_chunks: Optional[List[str]] = None
    error_message: Optional[str] = None
    model_name: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

@dataclass
class StepLogEntry:
    """단계별 로그 엔트리"""
    timestamp: str
    session_id: str
    step: ProcessingStep
    step_time: float
    details: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatbotLogger:
    """챗봇 전용 로거"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        로거 초기화
        
        Args:
            log_dir: 로그 파일 저장 디렉토리
        """
        try:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(exist_ok=True, parents=True)
            
            # 로그 파일 경로들
            self.detailed_log_path = self.log_dir / "chatbot_detailed.log"
            self.summary_log_path = self.log_dir / "chatbot_summary.log"
            self.sql_log_path = self.log_dir / "sql_queries.log"
            self.pdf_log_path = self.log_dir / "pdf_queries.log"
            self.step_log_path = self.log_dir / "step_processing.log"
            self.module_mode_log_path = self.log_dir / "module_mode.log"
            
            # 로거 설정
            self._setup_loggers()
            
            # 세션 카운터
            self.session_counter = 0
            
            # 콘솔 출력 대신 파일 로그에만 기록
            logging.getLogger("chatbot_summary").info(f"챗봇 로거 초기화 완료: {self.log_dir}")
            
        except Exception as e:
            logging.getLogger("chatbot_summary").error(f"챗봇 로거 초기화 실패: {e}")
            # 폴백 로거 설정
            self._setup_fallback_logger()
        
    def _setup_fallback_logger(self):
        """폴백 로거 설정 (기본 로깅)"""
        self.detailed_logger = logging.getLogger("chatbot_fallback")
        self.summary_logger = logging.getLogger("chatbot_fallback")
        self.sql_logger = logging.getLogger("chatbot_fallback")
        self.pdf_logger = logging.getLogger("chatbot_fallback")
        self.step_logger = logging.getLogger("chatbot_fallback")
        
        # 콘솔 핸들러 대신 파일 핸들러만 사용 (폴백 모드에서도)
        file_handler = logging.FileHandler(self.log_dir / "fallback.log", encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        for logger in [self.detailed_logger, self.summary_logger, self.sql_logger, self.pdf_logger, self.step_logger]:
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            logger.addHandler(file_handler)
        
        self.session_counter = 0
        
    def _setup_loggers(self):
        """로거 설정"""
        try:
            # 상세 로그 (모든 정보)
            self.detailed_logger = logging.getLogger("chatbot_detailed")
            self.detailed_logger.setLevel(logging.INFO)
            self.detailed_logger.handlers.clear()
            
            detailed_handler = logging.FileHandler(self.detailed_log_path, encoding='utf-8', mode='a')
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            detailed_handler.setFormatter(detailed_formatter)
            self.detailed_logger.addHandler(detailed_handler)
            
            # 요약 로그 (핵심 정보만)
            self.summary_logger = logging.getLogger("chatbot_summary")
            self.summary_logger.setLevel(logging.INFO)
            self.summary_logger.handlers.clear()
            
            summary_handler = logging.FileHandler(self.summary_log_path, encoding='utf-8', mode='a')
            summary_formatter = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            summary_handler.setFormatter(summary_formatter)
            self.summary_logger.addHandler(summary_handler)
            
            # SQL 전용 로그
            self.sql_logger = logging.getLogger("chatbot_sql")
            self.sql_logger.setLevel(logging.INFO)
            self.sql_logger.handlers.clear()
            
            sql_handler = logging.FileHandler(self.sql_log_path, encoding='utf-8', mode='a')
            sql_formatter = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            sql_handler.setFormatter(sql_formatter)
            self.sql_logger.addHandler(sql_handler)
            
            # PDF 전용 로그
            self.pdf_logger = logging.getLogger("chatbot_pdf")
            self.pdf_logger.setLevel(logging.INFO)
            self.pdf_logger.handlers.clear()
            
            pdf_handler = logging.FileHandler(self.pdf_log_path, encoding='utf-8', mode='a')
            pdf_formatter = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            pdf_handler.setFormatter(pdf_formatter)
            self.pdf_logger.addHandler(pdf_handler)
            
            # 단계별 처리 로그
            self.step_logger = logging.getLogger("chatbot_step")
            self.step_logger.setLevel(logging.INFO)
            self.step_logger.handlers.clear()
            
            step_handler = logging.FileHandler(self.step_log_path, encoding='utf-8', mode='a')
            step_formatter = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            step_handler.setFormatter(step_formatter)
            self.step_logger.addHandler(step_handler)
            
            # 모듈 모드 전용 로그
            self.module_mode_logger = logging.getLogger("chatbot_module_mode")
            self.module_mode_logger.setLevel(logging.INFO)
            self.module_mode_logger.handlers.clear()
            
            module_mode_handler = logging.FileHandler(self.module_mode_log_path, encoding='utf-8', mode='a')
            module_mode_formatter = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            module_mode_handler.setFormatter(module_mode_formatter)
            self.module_mode_logger.addHandler(module_mode_handler)
            
        except Exception as e:
            logging.getLogger("chatbot_summary").error(f"로거 설정 실패: {e}")
            self._setup_fallback_logger()
    
    def _generate_session_id(self) -> str:
        """세션 ID 생성"""
        self.session_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}_{self.session_counter:04d}"
    
    def log_step(self, 
                session_id: str,
                step: ProcessingStep,
                step_time: float,
                details: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        단계별 처리 로그 기록
        
        Args:
            session_id: 세션 ID
            step: 처리 단계
            step_time: 단계별 처리 시간
            details: 상세 정보
            metadata: 추가 메타데이터
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # 단계별 로그 메시지 생성
            step_msg = f"[{session_id}] {step.value} : {step_time:.3f}초"
            if details:
                step_msg += f" | {details}"
            
            # 단계별 로그 기록
            self.step_logger.info(step_msg)
            
            # 메타데이터가 있으면 상세 로그에도 기록
            if metadata:
                detailed_step_log = {
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "step": step.value,
                    "step_time": step_time,
                    "details": details,
                    "metadata": metadata
                }
                self.detailed_logger.info(f"STEP_LOG: {json.dumps(detailed_step_log, ensure_ascii=False)}")
                
        except Exception as e:
            logging.getLogger("chatbot_summary").error(f"단계별 로그 기록 실패: {e}")
    
    def log_question(self, 
                    user_question: str,
                    question_type: QuestionType,
                    intent: str,
                    keywords: List[str],
                    processing_time: float,
                    confidence_score: float,
                    generated_sql: Optional[str] = None,
                    generated_answer: Optional[str] = None,
                    used_chunks: Optional[List[str]] = None,
                    error_message: Optional[str] = None,
                    model_name: Optional[str] = None,
                    additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        질문 로깅
        
        Args:
            user_question: 사용자 질문
            question_type: 질문 유형
            intent: 질문 의도
            keywords: 추출된 키워드
            processing_time: 처리 시간
            confidence_score: 신뢰도 점수
            generated_sql: 생성된 SQL (SQL 질문인 경우)
            generated_answer: 생성된 답변 (PDF 질문인 경우)
            used_chunks: 사용된 청크들 (PDF 질문인 경우)
            error_message: 오류 메시지
            model_name: 사용된 모델명
            additional_info: 추가 정보
            
        Returns:
            세션 ID
        """
        try:
            session_id = self._generate_session_id()
            timestamp = datetime.now().isoformat()
            
            # 로그 엔트리 생성
            log_entry = ChatbotLogEntry(
                timestamp=timestamp,
                session_id=session_id,
                user_question=user_question,
                question_type=question_type,
                intent=intent,
                keywords=keywords,
                processing_time=processing_time,
                confidence_score=confidence_score,
                generated_sql=generated_sql,
                generated_answer=generated_answer,
                used_chunks=used_chunks,
                error_message=error_message,
                model_name=model_name,
                additional_info=additional_info
            )
            
            # 상세 로그 (JSON 형태)
            detailed_log = {
                "session_id": session_id,
                "timestamp": timestamp,
                "question": user_question,
                "type": question_type.value,
                "intent": intent,
                "keywords": keywords,
                "processing_time": processing_time,
                "confidence": confidence_score,
                "model": model_name,
                "sql": generated_sql,
                "answer": generated_answer,
                "used_chunks": used_chunks,
                "error": error_message,
                "additional_info": additional_info
            }
            
            self.detailed_logger.info(json.dumps(detailed_log, ensure_ascii=False, indent=2))
            
            # 요약 로그
            summary_msg = f"[{session_id}] {question_type.value.upper()} | {user_question[:50]}... | {processing_time:.2f}s | {confidence_score:.2f}"
            if error_message:
                summary_msg += f" | ERROR: {error_message}"
            self.summary_logger.info(summary_msg)
            
            # 유형별 전용 로그
            if question_type == QuestionType.SQL and generated_sql:
                sql_msg = f"[{session_id}] {user_question} | {generated_sql}"
                self.sql_logger.info(sql_msg)
            
            elif question_type == QuestionType.PDF and generated_answer:
                pdf_msg = f"[{session_id}] {user_question} | {generated_answer[:100]}..."
                self.pdf_logger.info(pdf_msg)
            
            return session_id
            
        except Exception as e:
            logging.getLogger("chatbot_summary").error(f"질문 로깅 실패: {e}")
            return "error_session"
            # 폴백 로깅
            try:
                fallback_msg = f"LOG_ERROR: {user_question} | {question_type.value} | {error_message or 'No error'}"
                self.detailed_logger.error(fallback_msg)
                return f"error_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            except:
                return f"fallback_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def log_sql_query(self, 
                     user_question: str,
                     generated_sql: str,
                     processing_time: float,
                     confidence_score: float,
                     model_name: Optional[str] = None) -> str:
        """SQL 쿼리 전용 로깅"""
        return self.log_question(
            user_question=user_question,
            question_type=QuestionType.SQL,
            intent="sql_generation",
            keywords=[],
            processing_time=processing_time,
            confidence_score=confidence_score,
            generated_sql=generated_sql,
            model_name=model_name
        )
    
    def log_pdf_query(self,
                     user_question: str,
                     generated_answer: str,
                     used_chunks: List[str],
                     processing_time: float,
                     confidence_score: float,
                     model_name: Optional[str] = None) -> str:
        """PDF 질문 전용 로깅"""
        return self.log_question(
            user_question=user_question,
            question_type=QuestionType.PDF,
            intent="pdf_qa",
            keywords=[],
            processing_time=processing_time,
            confidence_score=confidence_score,
            generated_answer=generated_answer,
            used_chunks=used_chunks,
            model_name=model_name
        )
    
    def log_greeting(self,
                    user_question: str,
                    greeting_response: str,
                    processing_time: float,
                    confidence_score: float,
                    greeting_type: str = "general") -> str:
        """인사말 전용 로깅"""
        return self.log_question(
            user_question=user_question,
            question_type=QuestionType.GREETING,
            intent=f"greeting_{greeting_type}",
            keywords=[],
            processing_time=processing_time,
            confidence_score=confidence_score,
            generated_answer=greeting_response,
            additional_info={"greeting_type": greeting_type}
        )
    
    def log_error(self, 
                 user_question: str,
                 error_message: str,
                 question_type: QuestionType = QuestionType.UNKNOWN) -> str:
        """오류 로깅"""
        return self.log_question(
            user_question=user_question,
            question_type=question_type,
            intent="error",
            keywords=[],
            processing_time=0.0,
            confidence_score=0.0,
            error_message=error_message
        )
    
    def log_module_mode(self, 
                       message: str,
                       session_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        모듈 모드 전용 로깅
        
        Args:
            message: 로그 메시지
            session_id: 세션 ID (선택사항)
            metadata: 추가 메타데이터 (선택사항)
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # 모듈 모드 로그 메시지 생성
            if session_id:
                log_msg = f"[{session_id}] {message}"
            else:
                log_msg = message
            
            # 모듈 모드 로그 기록
            self.module_mode_logger.info(log_msg)
            
            # 메타데이터가 있으면 상세 로그에도 기록
            if metadata:
                detailed_module_log = {
                    "timestamp": timestamp,
                    "session_id": session_id,
                    "message": message,
                    "metadata": metadata
                }
                self.detailed_logger.info(f"MODULE_MODE_LOG: {json.dumps(detailed_module_log, ensure_ascii=False)}")
                
        except Exception as e:
            logging.getLogger("chatbot_summary").error(f"모듈 모드 로그 기록 실패: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """로그 통계 조회"""
        try:
            stats = {
                "total_sessions": self.session_counter,
                "log_files": {
                    "detailed": str(self.detailed_log_path),
                    "summary": str(self.summary_log_path),
                    "sql": str(self.sql_log_path),
                    "pdf": str(self.pdf_log_path),
                    "step": str(self.step_log_path)
                }
            }
            
            # 파일 크기 정보
            for log_type, log_path in stats["log_files"].items():
                if os.path.exists(log_path):
                    stats[f"{log_type}_size"] = os.path.getsize(log_path)
                else:
                    stats[f"{log_type}_size"] = 0
            
            return stats
            
        except Exception as e:
            logging.getLogger("chatbot_summary").error(f"로그 통계 조회 실패: {e}")
            return {
                "total_sessions": self.session_counter,
                "error": str(e)
            }
    
    def clear_logs(self):
        """로그 파일들 초기화"""
        try:
            for log_path in [self.detailed_log_path, self.summary_log_path, 
                            self.sql_log_path, self.pdf_log_path, self.step_log_path]:
                if log_path.exists():
                    log_path.unlink()
            
            self.session_counter = 0
            logging.getLogger("chatbot_summary").info("모든 로그 파일이 초기화되었습니다.")
            
        except Exception as e:
            logging.getLogger("chatbot_summary").error(f"로그 초기화 실패: {e}")

def _ensure_error_log_handler(logger_name: str = "error_file"):
    """Ensure error.log file handler is attached once."""
    root_dir = Path(__file__).resolve().parents[1]
    logs_dir = root_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = logs_dir / "error.log"

    logger = logging.getLogger(logger_name)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "_is_error_log", False) for h in logger.handlers):
        handler = logging.FileHandler(error_log_path, encoding='utf-8')
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s')
        handler.setFormatter(formatter)
        # mark handler to detect duplicates
        handler._is_error_log = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def log_exception_to_error_log(message: str):
    """Write an error message into logs/error.log."""
    try:
        err_logger = _ensure_error_log_handler()
        err_logger.error(message)
    except Exception:
        pass

# 전역 로거 인스턴스
try:
    chatbot_logger = ChatbotLogger()
except Exception as e:
    logging.getLogger("chatbot_summary").error(f"전역 로거 생성 실패: {e}")
    # 최소한의 로거 생성
    chatbot_logger = None
