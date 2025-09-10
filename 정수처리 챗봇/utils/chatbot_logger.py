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
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

# 로깅 에러 무시를 위한 전역 설정
logging.raiseExceptions = False

# 에러 로거 import
try:
    from .error_logger import log_error, log_system_error
except ImportError:
    # 상대 import 실패 시 절대 import 시도
    try:
        from utils.error_logger import log_error, log_system_error
    except ImportError:
        # 폴백 함수들
        def log_error(error, context=None, additional_info=None):
            print(f"에러 로그 기록 실패: {error}", file=sys.stderr)
        
        def log_system_error(message, module="unknown", additional_info=None):
            print(f"시스템 에러: {message}", file=sys.stderr)

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
    
    def __init__(self, log_dir: str = "logs", reset_qa_log: bool = True):
        """
        로거 초기화
        
        Args:
            log_dir: 로그 파일 저장 디렉토리
            reset_qa_log: qa_log 파일을 초기화할지 여부
        """
        try:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(exist_ok=True, parents=True)
            
            # qa_log만 유지
            self.qa_log_path = self.log_dir / "qa_log.jsonl"
            
            # qa_log 초기화 (새로 시작할 때마다) - 효율적인 초기화 방식
            if reset_qa_log and self.qa_log_path.exists():
                try:
                    # 1단계: 먼저 내용만 초기화 시도 (가장 안전하고 빠름)
                    try:
                        with open(self.qa_log_path, 'w', encoding='utf-8') as f:
                            f.write('')
                        try:
                            logging.getLogger("chatbot_summary").info("qa_log 파일 내용이 초기화되었습니다.")
                        except (ValueError, OSError):
                            # 로깅 에러 무시
                            pass
                    except (OSError, IOError, PermissionError) as e:
                        # 2단계: 내용 초기화 실패 시 파일 삭제 시도
                        error_code = getattr(e, 'winerror', None) if hasattr(e, 'winerror') else None
                        if error_code == 32 or isinstance(e, PermissionError):
                            # 파일이 잠겨있으면 삭제 시도
                            max_retries = 3
                            base_delay = 0.2
                            
                            for attempt in range(max_retries):
                                try:
                                    self.qa_log_path.unlink()
                                    try:
                                        logging.getLogger("chatbot_summary").info("qa_log 파일이 삭제되었습니다.")
                                    except (ValueError, OSError):
                                        # 로깅 에러 무시
                                        pass
                                    break
                                except (OSError, IOError, PermissionError) as delete_e:
                                    if attempt < max_retries - 1:
                                        delay = base_delay * (2 ** attempt)
                                        time.sleep(delay)
                                        continue
                                    else:
                                        try:
                                            logging.getLogger("chatbot_summary").warning("qa_log 초기화 실패 (파일 잠금), 계속 진행")
                                        except (ValueError, OSError):
                                            # 로깅 에러 무시
                                            pass
                        else:
                            try:
                                logging.getLogger("chatbot_summary").warning(f"qa_log 초기화 실패, 계속 진행: {e}")
                            except (ValueError, OSError):
                                # 로깅 에러 무시
                                pass
                except Exception as e:
                    try:
                        logging.getLogger("chatbot_summary").warning(f"qa_log 초기화 실패, 계속 진행: {e}")
                    except (ValueError, OSError):
                        # 로깅 에러 무시
                        pass
            
            # 로거 설정
            self._setup_loggers()
            
            # 세션 카운터
            self.session_counter = 0
            
            # 콘솔 출력 대신 파일 로그에만 기록
            try:
                logging.getLogger("chatbot_summary").info(f"챗봇 로거 초기화 완료: {self.log_dir}")
            except (ValueError, OSError):
                # 로깅 에러 무시
                pass
            
        except Exception as e:
            try:
                logging.getLogger("chatbot_summary").error(f"챗봇 로거 초기화 실패: {e}")
            except (ValueError, OSError):
                # 로깅 에러 무시
                pass
            # 에러 로그에 기록
            log_error(e, "ChatbotLogger.__init__", {"log_dir": log_dir})
            # 폴백 로거 설정
            self._setup_fallback_logger()
        
    def _setup_fallback_logger(self):
        """폴백 로거 설정 (기본 로깅)"""
        self.summary_logger = logging.getLogger("chatbot_fallback")
        self.step_logger = logging.getLogger("chatbot_fallback")
        
        # 로그 출력 비활성화 (폴백 모드에서도)
        null_handler = logging.NullHandler()
        
        for logger in [self.summary_logger, self.step_logger]:
            logger.setLevel(logging.CRITICAL)
            logger.handlers.clear()
            logger.addHandler(null_handler)
        
        self.session_counter = 0
        
    def _setup_loggers(self):
        """로거 설정 - 콘솔 출력만"""
        try:
            # 요약 로그 (핵심 정보만) - 콘솔 출력
            self.summary_logger = logging.getLogger("chatbot_summary")
            self.summary_logger.setLevel(logging.INFO)
            self.summary_logger.handlers.clear()
            
            # 콘솔 핸들러 추가 (핵심 정보만) - 안전한 스트림 처리
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s')
            console_handler.setFormatter(formatter)
            
            # 스트림 에러 무시를 위한 필터 추가
            class StreamErrorFilter(logging.Filter):
                def filter(self, record):
                    try:
                        return True
                    except (ValueError, OSError):
                        # 스트림 에러 무시
                        return False
            
            console_handler.addFilter(StreamErrorFilter())
            self.summary_logger.addHandler(console_handler)
            
            # 단계별 처리 로그 - 콘솔 출력
            self.step_logger = logging.getLogger("chatbot_step")
            self.step_logger.setLevel(logging.INFO)
            self.step_logger.handlers.clear()
            
            # 콘솔 핸들러 추가 (단계별 진행상황) - 안전한 스트림 처리
            step_console_handler = logging.StreamHandler(sys.stdout)
            step_console_handler.setLevel(logging.INFO)
            step_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s')
            step_console_handler.setFormatter(step_formatter)
            
            # 스트림 에러 무시를 위한 필터 추가
            step_console_handler.addFilter(StreamErrorFilter())
            self.step_logger.addHandler(step_console_handler)
            
        except Exception as e:
            try:
                logging.getLogger("chatbot_summary").error(f"로거 설정 실패: {e}")
            except (ValueError, OSError):
                # 로깅 에러 무시
                pass
            # 에러 로그에 기록
            log_error(e, "ChatbotLogger._setup_loggers")
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
            
            # 메타데이터는 콘솔에만 출력
                
        except Exception as e:
            try:
                logging.getLogger("chatbot_summary").error(f"단계별 로그 기록 실패: {e}")
            except (ValueError, OSError):
                # 로깅 에러 무시
                pass
    
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
                    additional_info: Optional[Dict[str, Any]] = None,
                    question_analysis: Optional[Dict[str, Any]] = None) -> str:
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
            
            # qa_log에만 JSON 형태로 기록 (가독성 개선)
            qa_log_entry = {
                "세션_ID": session_id,
                "타임스탬프": timestamp,
                "사용자_질문": user_question,
                "질문_유형": question_type.value,
                "의도": intent,
                "키워드": keywords,
                "처리_시간_초": round(processing_time, 3),
                "신뢰도": round(confidence_score, 3),
                "사용_모델": model_name,
                "생성된_SQL": generated_sql,
                "생성된_답변": generated_answer,
                "사용된_청크": used_chunks,
                "에러_메시지": error_message,
                "추가_정보": additional_info,
                "질문_분석": question_analysis
            }
            
            # qa_log.jsonl에 기록 (가독성 좋게, 파일 잠금 처리)
            try:
                # 파일 잠금을 위한 재시도 로직
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        with open(self.qa_log_path, 'a', encoding='utf-8') as f:
                            # 구분선과 질문 번호 추가
                            f.write('\n' + '='*100 + '\n')
                            f.write(f'질문 #{self.session_counter} - {timestamp}\n')
                            f.write('='*100 + '\n\n')
                            
                            # JSON을 들여쓰기와 함께 저장
                            formatted_json = json.dumps(qa_log_entry, ensure_ascii=False, indent=2)
                            f.write(formatted_json + '\n\n')
                            
                            # 하단 구분선
                            f.write('-'*100 + '\n')
                            f.flush()  # 즉시 디스크에 쓰기
                        break  # 성공 시 루프 종료
                    except (OSError, IOError) as e:
                        if attempt < max_retries - 1:
                            time.sleep(0.1)  # 100ms 대기 후 재시도
                            continue
                        else:
                            raise e
            except Exception as e:
                try:
                    logging.getLogger("chatbot_summary").error(f"qa_log 기록 실패: {e}")
                except (ValueError, OSError):
                    # 로깅 에러 무시
                    pass
                # 에러 로그에 기록
                log_error(e, "ChatbotLogger.log_question", {"session_id": session_id})
            
            # 요약 로그
            summary_msg = f"[{session_id}] {question_type.value.upper()} | {user_question[:50]}... | {processing_time:.2f}s | {confidence_score:.2f}"
            if error_message:
                summary_msg += f" | ERROR: {error_message}"
            self.summary_logger.info(summary_msg)
            
            # 유형별 로그는 qa_log에만 기록됨
            
            return session_id
            
        except Exception as e:
            try:
                logging.getLogger("chatbot_summary").error(f"질문 로깅 실패: {e}")
            except (ValueError, OSError):
                # 로깅 에러 무시
                pass
            return f"error_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
            
            # 메타데이터는 콘솔에만 출력
                
        except Exception as e:
            try:
                logging.getLogger("chatbot_summary").error(f"모듈 모드 로그 기록 실패: {e}")
            except (ValueError, OSError):
                # 로깅 에러 무시
                pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """로그 통계 조회"""
        try:
            stats = {
                "total_sessions": self.session_counter,
                "log_files": {
                    "qa_log": str(self.qa_log_path)
                }
            }
            
            # 파일 크기 정보
            if os.path.exists(self.qa_log_path):
                stats["qa_log_size"] = os.path.getsize(self.qa_log_path)
            else:
                stats["qa_log_size"] = 0
            
            return stats
            
        except Exception as e:
            try:
                logging.getLogger("chatbot_summary").error(f"로그 통계 조회 실패: {e}")
            except (ValueError, OSError):
                # 로깅 에러 무시
                pass
            return {
                "total_sessions": self.session_counter,
                "error": str(e)
            }
    
    def clear_logs(self):
        """로그 파일들 초기화"""
        try:
            if self.qa_log_path.exists():
                self.qa_log_path.unlink()
            
            self.session_counter = 0
            try:
                logging.getLogger("chatbot_summary").info("qa_log 파일이 초기화되었습니다.")
            except (ValueError, OSError):
                # 로깅 에러 무시
                pass
            
        except Exception as e:
            try:
                logging.getLogger("chatbot_summary").error(f"로그 초기화 실패: {e}")
            except (ValueError, OSError):
                # 로깅 에러 무시
                pass

# 기존 에러 로그 함수들은 새로운 error_logger로 대체됨

# 전역 로거 인스턴스
try:
    chatbot_logger = ChatbotLogger()
except Exception as e:
    try:
        logging.getLogger("chatbot_summary").error(f"전역 로거 생성 실패: {e}")
    except (ValueError, OSError):
        # 로깅 에러 무시
        pass
    # 에러 로그에 기록
    log_error(e, "전역_로거_생성")
    # 최소한의 로거 생성
    chatbot_logger = None
