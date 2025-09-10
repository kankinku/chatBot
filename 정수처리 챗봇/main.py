#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로컬 테스트용 챗봇 메인 스크립트

사용자가 질문을 입력하면 답변과 로그가 출력되는 방식으로 구현
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Optional, List
import subprocess
import requests

# 에러 로거 import
try:
    from utils.error_logger import log_error, log_system_error, catch_and_log_errors
except ImportError:
    # 폴백 함수들
    def log_error(error, context=None, additional_info=None):
        print(f"에러 로그 기록 실패: {error}", file=sys.stderr)
    
    def log_system_error(message, module="unknown", additional_info=None):
        print(f"시스템 에러: {message}", file=sys.stderr)
    
    def catch_and_log_errors(context=None):
        def decorator(func):
            return func
        return decorator

# Windows에서 한국어 인코딩 문제 해결
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 로그 디렉토리 생성 (로깅 설정 전에 수행)
log_dir = current_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# 로깅 설정 - 핵심 정보만 출력
logging.basicConfig(
    level=logging.WARNING,  # WARNING 이상만 출력
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    handlers=[
        logging.NullHandler()
    ]
)

# tqdm 진행률 표시줄 완전 비활성화
os.environ['TQDM_DISABLE'] = '1'
os.environ['TQDM_DISABLE_PROGRESS_BAR'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# tqdm 모듈 자체를 완전히 비활성화
try:
    import tqdm
    # tqdm 클래스 자체를 무효화
    tqdm.tqdm.__init__ = lambda self, *args, **kwargs: None
    tqdm.tqdm.__enter__ = lambda self: self
    tqdm.tqdm.__exit__ = lambda self, *args: None
    tqdm.tqdm.update = lambda self, n=1: None
    tqdm.tqdm.close = lambda self: None
    tqdm.tqdm.refresh = lambda self: None
    tqdm.tqdm.display = lambda self: None
    tqdm.tqdm.write = lambda self, s, file=None: None
except ImportError:
    pass

# sys.stdout을 가로채서 진행률 표시줄 제거
import io
class NoProgressStdout(io.StringIO):
    def write(self, s):
        # 진행률 표시줄 패턴 제거
        if 'Batches:' in s or '|' in s and '%' in s and 'it/s' in s:
            return
        return super().write(s)

# stdout을 임시로 교체 (진행률 표시줄만 필터링)
original_stdout = sys.stdout
sys.stdout = NoProgressStdout()

# 불필요한 경고/로그 레벨 조정
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\\.io\\.image")
logging.getLogger('faiss.loader').setLevel(logging.WARNING)
logging.getLogger('torchvision').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# 통합 시스템 임포트 (데코레이터 사용 전에)
try:
    from utils.unified_logger import unified_logger, log_info, log_error, LogCategory, log_execution_time
    from core.config.unified_config import get_config, config
    from core.utils.singleton_manager import singleton_manager
except ImportError as e:
    logger.warning(f"통합 시스템 임포트 실패, 기본 시스템 사용: {e}")
    # 폴백: 기본 함수들 정의
    def log_execution_time(category, operation):
        def decorator(func):
            return func
        return decorator
    
    def get_config(key, default=None):
        return os.getenv(key, default)
    
    class LogCategory:
        SYSTEM = "system"

@log_execution_time(LogCategory.SYSTEM, "Ollama 모델 확인")
def ensure_qwen_ollama_model() -> bool:
    """Ollama에서 Qwen 모델 사용 가능 여부 확인 (통합 설정 사용)"""
    base_url = get_config("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model_name = get_config("LLM_MODEL", "qwen2:1.5b-instruct-q4_K_M")
    try:
        # Ollama 서버 헬스 체크
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        tags = [t.get("name") for t in data.get("models", []) if t.get("name")]
        if model_name in tags:
            try:
                log_info(f"Qwen(Ollama) 모델 확인 성공: {model_name}", LogCategory.SYSTEM, "main")
            except:
                logger.info(f"Qwen(Ollama) 모델 확인 성공: {model_name}")
            return True
        try:
            log_error(f"Qwen 모델이 아직 없습니다: {model_name}. ollama pull 필요", LogCategory.SYSTEM, "main")
        except:
            logger.warning(f"Qwen 모델이 아직 없습니다: {model_name}. ollama pull 필요")
        return False
    except Exception as e:
        try:
            log_error(f"Ollama 연결 실패: {e}", LogCategory.SYSTEM, "main")
        except:
            logger.error(f"Ollama 연결 실패: {e}")
        return False

# 핵심 모듈들 임포트
try:
    from core.document.pdf_processor import PDFProcessor, TextChunk
    from core.document.vector_store import HybridVectorStore, VectorStoreInterface
    from core.query.question_analyzer import QuestionAnalyzer, AnalyzedQuestion
    from core.llm.answer_generator import AnswerGenerator, Answer
    from core.query.query_router import QueryRouter, QueryRoute
    from core.query.llm_greeting_handler import GreetingHandler
    from utils.chatbot_logger import chatbot_logger, QuestionType, ProcessingStep
    
    logger.info("[SUCCESS] 모든 핵심 모듈 import 성공")
    
except ImportError as e:
    logger.error(f"[ERROR] 모듈 import 실패: {e}")
    sys.exit(1)

# 시작 시 강제 데이터 초기화 유틸리티
def force_startup_cleanup():
    """시작 시 Chroma/FAISS/전처리 DB를 강제 초기화한다."""
    try:
        import shutil
        import time
        base_dir = Path(__file__).parent

        chroma_dir = base_dir / "chroma_db"
        faiss_dir = base_dir / "vector_store" / "faiss"
        pdf_db = base_dir / "data" / "pdf_database.db"
        error_log = base_dir / "logs" / "error.log"

        # ChromaDB 제거 (강화된 재시도 로직)
        if chroma_dir.exists():
            _safe_remove_directory(chroma_dir, "ChromaDB")
            logger.info(f"[CLEANUP] ChromaDB 제거: {chroma_dir}")

        # FAISS 제거 (강화된 재시도 로직)
        if faiss_dir.exists():
            _safe_remove_directory(faiss_dir, "FAISS")
            logger.info(f"[CLEANUP] FAISS 제거: {faiss_dir}")

        # PDF DB 제거 (강화된 재시도 로직)
        if pdf_db.exists():
            _safe_remove_file(pdf_db, "전처리 DB")
            logger.info(f"[CLEANUP] 전처리 DB 삭제: {pdf_db}")

        # error.log 초기화 (비활성화됨 - 로그 보존)
        # try:
        #     (base_dir / "logs").mkdir(parents=True, exist_ok=True)
        #     _safe_write_file(error_log, "", "error.log 초기화")
        #     logger.info(f"[CLEANUP] error.log 초기화: {error_log}")
        # except Exception as e:
        #     logger.warning(f"[CLEANUP] error.log 초기화 실패: {e}")
        logger.info(f"[CLEANUP] error.log 보존 (초기화 비활성화)")

    except Exception as e:
        logger.warning(f"[CLEANUP] 시작 시 초기화 실패(계속 진행): {e}")

def _safe_remove_directory(directory_path, description=""):
    """디렉토리를 안전하게 제거 (WinError 32 대응)"""
    max_retries = 3
    base_delay = 0.2
    
    for attempt in range(max_retries):
        try:
            shutil.rmtree(directory_path, ignore_errors=True)
            return  # 성공 시 종료
        except (OSError, PermissionError) as e:
            error_code = getattr(e, 'winerror', None) if hasattr(e, 'winerror') else None
            if error_code == 32 or isinstance(e, PermissionError):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(f"[CLEANUP] {description} 제거 실패 (파일 잠금): {directory_path}")
                    return
            else:
                logger.warning(f"[CLEANUP] {description} 제거 실패: {e}")
                return

def _safe_remove_file(file_path, description=""):
    """파일을 안전하게 제거 (WinError 32 대응)"""
    max_retries = 3
    base_delay = 0.2
    
    for attempt in range(max_retries):
        try:
            file_path.unlink()
            return  # 성공 시 종료
        except (OSError, PermissionError) as e:
            error_code = getattr(e, 'winerror', None) if hasattr(e, 'winerror') else None
            if error_code == 32 or isinstance(e, PermissionError):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    # 최종 시도 실패 시 os.remove로 재시도
                    try:
                        os.remove(str(file_path))
                        return
                    except Exception:
                        logger.warning(f"[CLEANUP] {description} 제거 실패 (파일 잠금): {file_path}")
                        return
            else:
                logger.warning(f"[CLEANUP] {description} 제거 실패: {e}")
                return

def _safe_write_file(file_path, content, description=""):
    """파일을 안전하게 쓰기 (WinError 32 대응)"""
    max_retries = 3
    base_delay = 0.2
    
    for attempt in range(max_retries):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            return  # 성공 시 종료
        except (OSError, PermissionError) as e:
            error_code = getattr(e, 'winerror', None) if hasattr(e, 'winerror') else None
            if error_code == 32 or isinstance(e, PermissionError):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(f"[CLEANUP] {description} 실패 (파일 잠금): {file_path}")
                    return
            else:
                logger.warning(f"[CLEANUP] {description} 실패: {e}")
                return

class LocalChatbot:
    """로컬 테스트용 챗봇 클래스"""
    
    def __init__(self):
        """초기화"""
        logger.info("로컬 챗봇 초기화 시작...")
        
        try:
            # QA 로그 초기화 (새로운 세션 시작)
            from utils.chatbot_logger import ChatbotLogger
            chatbot_logger = ChatbotLogger(reset_qa_log=True)
            logger.info("QA 로그 초기화 완료")
            
            # 컴포넌트들 초기화
            self.pdf_processor = PDFProcessor()
            self.vector_store = HybridVectorStore()
            self.question_analyzer = QuestionAnalyzer()
            # Ollama(Qwen)만 사용할 경우 기본은 프리로드 불필요이나, 질의 지연 방지 위해 경량 워밍업 수행
            self.answer_generator = AnswerGenerator(preload_models=False)
            self.query_router = QueryRouter()
            self.llm_greeting_handler = GreetingHandler(self.answer_generator)
            
            # 경량 워밍업: 최초 질의 지연 방지
            try:
                self.answer_generator.load_model()
                _ = self.answer_generator.generate_direct_answer("워밍업")
            except Exception as _warm_e:
                logger.warning(f"답변 생성기 워밍업 경고: {_warm_e}")

            logger.info("모든 컴포넌트 초기화 완료")
            
            # 기존 PDF 문서 로드
            try:
                existing_pdfs = self.vector_store.get_all_pdfs()
                total_chunks = self.vector_store.get_total_chunks()
                
                logger.info(f"기존 PDF 문서 {len(existing_pdfs)}개 로드됨 (총 청크 수: {total_chunks})")
                
                if existing_pdfs:
                    for pdf_info in existing_pdfs:
                        logger.info(f"  - {pdf_info['filename']} ({pdf_info['total_chunks']}개 청크)")
                else:
                    logger.info("  기존 PDF 문서가 없습니다.")

                # 자동 업로드: 청크가 0이면 data 경로에서 PDF를 재귀 스캔해 적재
                if total_chunks == 0:
                    logger.info("자동 업로드 시작: data 폴더에서 PDF를 스캔합니다...")
                    uploaded, failed = self._auto_upload_pdfs_from_disk()
                    logger.info(f"자동 업로드 완료: {uploaded}개 성공, {failed}개 실패")
                    logger.info(f"현재 총 청크: {self.vector_store.get_total_chunks()}개")
                    
            except Exception as e:
                logger.warning(f"기존 PDF 로드 실패: {e}")
                # 에러 로그에 기록
                log_error(e, "LocalChatbot.__init__", {"step": "기존_PDF_로드"})
            
        except Exception as e:
            logger.error(f"챗봇 초기화 실패: {e}")
            # 에러 로그에 기록
            log_error(e, "LocalChatbot.__init__", {"step": "챗봇_초기화"})
            raise
    
    def reset_and_regenerate_chunks(self):
        """청크 초기화 및 재생성"""
        try:
            print("1단계: 기존 청크 초기화...")
            
            # 기존 청크 수 확인
            total_chunks = self.vector_store.get_total_chunks()
            print(f"   기존 청크 수: {total_chunks}")
            
            # 벡터 저장소 초기화
            self.vector_store.clear()
            print("   벡터 저장소 초기화 완료")
            
            # 기존 PDF 파일들 찾기
            print("\n2단계: PDF 파일 스캔...")
            pdf_files = self._find_pdf_files()
            print(f"   발견된 PDF 파일: {len(pdf_files)}개")
            
            if not pdf_files:
                print("   PDF 파일이 없습니다. data/pdfs 폴더에 PDF 파일을 추가해주세요.")
                return
            
            # PDF 파일별로 청크 재생성
            print("\n3단계: 청크 재생성...")
            total_new_chunks = 0
            
            for pdf_file in pdf_files:
                try:
                    print(f"   처리 중: {pdf_file}")
                    
                    # PDF 처리
                    chunks = self.pdf_processor.process_pdf(pdf_file)
                    
                    if chunks:
                        # 벡터 저장소에 추가
                        self.vector_store.add_chunks(chunks)
                        total_new_chunks += len(chunks)
                        print(f"     → {len(chunks)}개 청크 생성 완료")
                    else:
                        print(f"     → 청크 생성 실패")
                        
                except Exception as e:
                    print(f"     → 오류: {e}")
                    continue
            
            print(f"\n청크 재생성 완료!")
            print(f"   총 생성된 청크: {total_new_chunks}개")
            print(f"   현재 총 청크: {self.vector_store.get_total_chunks()}개")
            
        except Exception as e:
            print(f"청크 초기화 및 재생성 실패: {e}")
            raise
    
    def _find_pdf_files(self) -> List[str]:
        """PDF 파일들 찾기"""
        pdf_files = []
        base_dir = Path(__file__).parent

        # 재귀적으로 data 및 data/pdfs 하위 모든 폴더에서 *.pdf 검색
        for search_dir in [base_dir / "data", base_dir / "data" / "pdfs"]:
            if search_dir.exists():
                pdf_files.extend([str(f) for f in search_dir.rglob("*.pdf")])

        return sorted(set(pdf_files))

    def _auto_upload_pdfs_from_disk(self) -> tuple:
        """디스크의 PDF를 찾아 처리하고 벡터 스토어에 적재"""
        pdf_files = self._find_pdf_files()
        if not pdf_files:
            logger.info("data 폴더에서 PDF 파일을 찾을 수 없습니다.")
            return (0, 0)

        logger.info(f"발견된 PDF 파일: {len(pdf_files)}개")
        uploaded = 0
        failed = 0
        for pdf_path in pdf_files:
            try:
                logger.info(f"PDF 처리 중: {pdf_path}")
                chunks = self.pdf_processor.process_pdf(pdf_path)[0] if isinstance(self.pdf_processor.process_pdf(pdf_path), tuple) else self.pdf_processor.process_pdf(pdf_path)
                # process_pdf가 (chunks, metadata) 반환하는 구현에 대응
                if isinstance(chunks, tuple):
                    chunks = chunks[0]
                self.vector_store.add_chunks(chunks)
                uploaded += 1
                logger.info(f"처리 완료: {pdf_path} ({len(chunks)}개 청크)")
            except Exception as e:
                logger.error(f"처리 실패: {pdf_path} - {e}")
                failed += 1
        return (uploaded, failed)
    def process_question(self, question: str) -> dict:
        """
        질문 처리
        
        Args:
            question: 사용자 질문
            
        Returns:
            처리 결과 딕셔너리
        """
        start_time = time.time()
        session_id = None
        
        try:
            logger.info(f"질문 처리 시작: {question}")
            
            # 단계별 로그 시작
            if chatbot_logger:
                session_id = chatbot_logger._generate_session_id()
                chatbot_logger.log_step(session_id, ProcessingStep.START, 0.0, f"질문: {question[:50]}...")
            
            # 1. SBERT 기반 쿼리 라우팅
            routing_start = time.time()
            route_result = self.query_router.route_query(question)
            routing_time = time.time() - routing_start
            
            logger.info(f"라우팅 결과: {route_result.route.value} (신뢰도: {route_result.confidence:.3f})")
            
            if chatbot_logger and session_id:
                chatbot_logger.log_step(
                    session_id, 
                    ProcessingStep.SBERT_ROUTING, 
                    routing_time, 
                    f"라우팅결과: {route_result.route.value} (신뢰도: {route_result.confidence:.3f})"
                )
            
            # 2. 인사말 처리
            if route_result.route == QueryRoute.GREETING:
                logger.info("인사말 처리 시작")
                
                if chatbot_logger and session_id:
                    chatbot_logger.log_step(session_id, ProcessingStep.GREETING_PIPELINE, 0.0, "인사말 처리 시작")
                
                greeting_response = self.llm_greeting_handler.get_greeting_response(question)
                
                logger.info(f"인사말 응답: {greeting_response['answer']}")
                
                if chatbot_logger and session_id:
                    chatbot_logger.log_step(session_id, ProcessingStep.COMPLETION, greeting_response["generation_time"], "인사말 처리 완료")
                
                return {
                    'success': True,
                    'answer': greeting_response['answer'],
                    'confidence_score': greeting_response['confidence_score'],
                    'question_type': 'greeting',
                    'processing_time': greeting_response['generation_time'],
                    'route': route_result.route.value,
                    'sql_query': None,
                    'used_chunks': []
                }
            
            # SQL 쿼리 처리는 제거되었습니다. PDF 검색으로 폴백합니다.
            # (SQL_QUERY enum이 제거되어 이 코드는 더 이상 실행되지 않음)
            
            # 4. PDF 검색 처리
            if route_result.route == QueryRoute.PDF_SEARCH:
                logger.info("PDF 검색 처리 시작")
                
                if chatbot_logger and session_id:
                    chatbot_logger.log_step(session_id, ProcessingStep.PDF_PIPELINE, 0.0, "PDF 파이프라인 시작")
                
                # 질문 분석 (다층적 Answer Target 추출 포함)
                analysis_start = time.time()
                analyzed_question = self.question_analyzer.analyze_question(question)
                analysis_time = time.time() - analysis_start
                
                logger.info(f"질문 분석 완료: {analyzed_question.question_type.value}")
                logger.info(f"답변 목표: {analyzed_question.answer_target} ({analyzed_question.target_type}, {analyzed_question.value_intent})")
                logger.info(f"추출 신뢰도: {analyzed_question.confidence_score:.3f}")
                
                if chatbot_logger and session_id:
                    chatbot_logger.log_step(session_id, ProcessingStep.QUESTION_ANALYSIS, analysis_time, 
                        f"질문분석: {analyzed_question.question_type.value}, 목표: {analyzed_question.answer_target} (신뢰도: {analyzed_question.confidence_score:.3f})")
                
                # 벡터 검색
                search_start = time.time()
                total_chunks = self.vector_store.get_total_chunks()
                logger.info(f"벡터 스토어 상태 - 총 청크 수: {total_chunks}")
                
                if total_chunks == 0:
                    logger.warning("벡터 스토어에 청크가 없습니다!")
                    return {
                        'success': False,
                        'answer': "죄송합니다. 현재 문서 데이터베이스가 비어있습니다. PDF 문서를 먼저 업로드해주세요.",
                        'confidence_score': 0.0,
                        'question_type': 'error',
                        'processing_time': time.time() - start_time,
                        'route': route_result.route.value,
                        'sql_query': None,
                        'used_chunks': []
                    }
                
                # 검색 범위 확대: 상위 10개, 임계값 상향
                relevant_chunks = self.vector_store.search(
                    analyzed_question.embedding,
                    top_k=10,
                    similarity_threshold=0.2
                )
                search_time = time.time() - search_start
                
                logger.info(f"검색된 청크 수: {len(relevant_chunks)}")
                
                if chatbot_logger and session_id:
                    chatbot_logger.log_step(session_id, ProcessingStep.VECTOR_SEARCH, search_time, f"벡터검색: {len(relevant_chunks)}개 청크 발견")
                
                # 답변 생성
                answer_start = time.time()
                if relevant_chunks:
                    chunks = [chunk for chunk, score in relevant_chunks]
                    logger.info(f"컨텍스트 기반 답변 생성 시작 ({len(chunks)}개 청크 사용)")
                    answer = self.answer_generator.generate_context_answer(question, chunks)
                else:
                    logger.info("관련 문서를 찾을 수 없어 직접 답변 생성 시작")
                    answer = self.answer_generator.generate_direct_answer(question)
                    
                    # 직접 답변도 실패한 경우 적절한 메시지 반환
                    if not answer or not answer.content or len(answer.content.strip()) < 10:
                        logger.warning("PDF에서 답변에 해당하는 내용을 찾을 수 없습니다.")
                        return {
                            'success': False,
                            'answer': "죄송합니다. PDF에서 답변에 해당하는 내용을 찾을 수 없습니다. 다른 질문을 시도해보시거나 더 구체적인 질문을 해주세요.",
                            'confidence_score': 0.0,
                            'question_type': 'pdf_no_content',
                            'processing_time': time.time() - start_time,
                            'route': route_result.route.value,
                            'sql_query': None,
                            'used_chunks': []
                        }
                
                answer_time = time.time() - answer_start
                
                logger.info(f"답변 생성 완료 (소요시간: {answer_time:.2f}초)")
                logger.info(f"생성된 답변: {answer.content[:100]}...")
                
                if chatbot_logger and session_id:
                    chatbot_logger.log_step(session_id, ProcessingStep.ANSWER_GENERATION, answer_time, "답변생성 완료")
                
                total_time = time.time() - start_time
                
                if chatbot_logger and session_id:
                    chatbot_logger.log_step(session_id, ProcessingStep.COMPLETION, total_time, "PDF 파이프라인 완료")
                
                # 질문 분석 정보 추출
                question_analysis = {
                    "question_type": analyzed_question.question_type.value,
                    "processed_question": analyzed_question.processed_question,
                    "keywords": analyzed_question.keywords,
                    "entities": analyzed_question.entities,
                    "intent": analyzed_question.intent,
                    "context_keywords": analyzed_question.context_keywords,
                    "answer_target": analyzed_question.answer_target,
                    "target_type": analyzed_question.target_type,
                    "value_intent": analyzed_question.value_intent,
                    "confidence_score": analyzed_question.confidence_score,
                    "enhanced_question": analyzed_question.enhanced_question,
                    "metadata": analyzed_question.metadata
                }
                
                # QA 로그 기록
                if chatbot_logger:
                    chatbot_logger.log_question(
                        user_question=question,
                        question_type=QuestionType.PDF,
                        intent=analyzed_question.intent,
                        keywords=analyzed_question.keywords,
                        processing_time=total_time,
                        confidence_score=answer.confidence,
                        generated_answer=answer.content,
                        used_chunks=[chunk.chunk_id for chunk in answer.sources] if answer.sources else [],
                        model_name=answer.model_name,
                        additional_info={
                            "pipeline_type": route_result.route.value,
                            "session_id": session_id
                        },
                        question_analysis=question_analysis
                    )
                
                return {
                    'success': True,
                    'answer': answer.content,
                    'confidence_score': answer.confidence,
                    'question_type': 'pdf',
                    'processing_time': total_time,
                    'route': route_result.route.value,
                    'sql_query': None,
                    'used_chunks': [chunk.chunk_id for chunk in answer.sources] if answer.sources else [],
                    # Answer Target 정보 추가
                    'answer_target': analyzed_question.answer_target,
                    'target_type': analyzed_question.target_type,
                    'value_intent': analyzed_question.value_intent,
                    'extraction_confidence': analyzed_question.confidence_score
                }
            
            # 5. 기본 처리 (PDF 검색으로 폴백)
            logger.info("기본 처리 (PDF 검색으로 폴백)")
            
            # 질문 분석 (다층적 Answer Target 추출 포함)
            analyzed_question = self.question_analyzer.analyze_question(question)
            logger.info(f"기본 처리 - 질문 분석 완료: {analyzed_question.question_type.value}")
            logger.info(f"답변 목표: {analyzed_question.answer_target} ({analyzed_question.target_type}, {analyzed_question.value_intent})")
            logger.info(f"추출 신뢰도: {analyzed_question.confidence_score:.3f}")
            
            # 벡터 검색
            # 기본 폴백 검색도 동일한 상한/임계값 적용
            relevant_chunks = self.vector_store.search(
                analyzed_question.embedding,
                top_k=10,
                similarity_threshold=0.2
            )
            
            # 답변 생성
            if relevant_chunks:
                chunks = [chunk for chunk, score in relevant_chunks]
                answer = self.answer_generator.generate_context_answer(question, chunks)
            else:
                answer = self.answer_generator.generate_direct_answer(question)
                
                # 직접 답변도 실패한 경우 적절한 메시지 반환
                if not answer or not answer.content or len(answer.content.strip()) < 10:
                    logger.warning("PDF에서 답변에 해당하는 내용을 찾을 수 없습니다.")
                    return {
                        'success': False,
                        'answer': "죄송합니다. PDF에서 답변에 해당하는 내용을 찾을 수 없습니다. 다른 질문을 시도해보시거나 더 구체적인 질문을 해주세요.",
                        'confidence_score': 0.0,
                        'question_type': 'pdf_no_content',
                        'processing_time': time.time() - start_time,
                        'route': 'fallback',
                        'sql_query': None,
                        'used_chunks': []
                    }
            
            total_time = time.time() - start_time
            
            # 질문 분석 정보 추출
            question_analysis = {
                "question_type": analyzed_question.question_type.value,
                "processed_question": analyzed_question.processed_question,
                "keywords": analyzed_question.keywords,
                "entities": analyzed_question.entities,
                "intent": analyzed_question.intent,
                "context_keywords": analyzed_question.context_keywords,
                "answer_target": analyzed_question.answer_target,
                "target_type": analyzed_question.target_type,
                "value_intent": analyzed_question.value_intent,
                "confidence_score": analyzed_question.confidence_score,
                "enhanced_question": analyzed_question.enhanced_question,
                "metadata": analyzed_question.metadata
            }
            
            # QA 로그 기록
            if chatbot_logger:
                chatbot_logger.log_question(
                    user_question=question,
                    question_type=QuestionType.PDF,
                    intent=analyzed_question.intent,
                    keywords=analyzed_question.keywords,
                    processing_time=total_time,
                    confidence_score=answer.confidence,
                    generated_answer=answer.content,
                    used_chunks=[chunk.chunk_id for chunk in answer.sources] if answer.sources else [],
                    model_name=answer.model_name,
                    additional_info={
                        "pipeline_type": "fallback",
                        "session_id": session_id
                    },
                    question_analysis=question_analysis
                )
            
            return {
                'success': True,
                'answer': answer.content,
                'confidence_score': answer.confidence,
                'question_type': 'fallback',
                'processing_time': total_time,
                'route': 'fallback',
                'sql_query': None,
                'used_chunks': [chunk.chunk_id for chunk in answer.sources] if answer.sources else [],
                # Answer Target 정보 추가
                'answer_target': analyzed_question.answer_target,
                'target_type': analyzed_question.target_type,
                'value_intent': analyzed_question.value_intent,
                'extraction_confidence': analyzed_question.confidence_score
            }
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            
            # 에러 로깅
            try:
                if chatbot_logger:
                    if session_id:
                        chatbot_logger.log_step(session_id, ProcessingStep.ERROR, 0.0, f"전체처리오류: {str(e)}")
                    
                    chatbot_logger.log_error(
                        user_question=question,
                        error_message=str(e),
                        question_type=QuestionType.UNKNOWN
                    )
            except Exception as log_error:
                logger.warning(f"에러 로깅 실패: {log_error}")
            
            return {
                'success': False,
                'answer': f"죄송합니다. 질문 처리 중 오류가 발생했습니다: {str(e)}",
                'confidence_score': 0.0,
                'question_type': 'error',
                'processing_time': time.time() - start_time,
                'route': 'error',
                'sql_query': None,
                'used_chunks': []
            }

def main():
    """메인 함수"""
    print("로컬 챗봇 테스트 시작")
    print("=" * 50)
    # 강제 데이터 초기화
    force_startup_cleanup()
    # Qwen(Ollama) 모델 확인
    print("Qwen(Ollama) 모델 확인 중...")
    if not ensure_qwen_ollama_model():
        print("Qwen 모델이 준비되지 않았습니다. 먼저 다음을 실행하세요:")
        print("  1) Ollama 실행 (포트 11434)\n  2) ollama pull qwen2:1.5b-instruct-q4_K_M")
        return 1
    print("=" * 50)
    
    # 청크 초기화 옵션 확인
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--reset-chunks":
        print("청크 초기화 모드로 시작합니다...")
        reset_chunks = True
    else:
        reset_chunks = False
    
    try:
        # 챗봇 초기화
        chatbot = LocalChatbot()
        print("챗봇 초기화 완료")
        
        # 청크 초기화가 요청된 경우
        if reset_chunks:
            print("\n청크 초기화를 시작합니다...")
            chatbot.reset_and_regenerate_chunks()
            print("청크 초기화 및 재생성 완료!")
            return 0
        
        print()
        
        # 대화 루프
        while True:
            try:
                # 사용자 입력 (한국어 인코딩 문제 해결)
                try:
                    question = input("질문을 입력하세요 (종료하려면 'quit' 또는 'exit' 입력): ").strip()
                except UnicodeDecodeError:
                    print("입력 인코딩 오류가 발생했습니다. 다시 시도해주세요.")
                    continue
                
                if question.lower() in ['quit', 'exit', '종료']:
                    print("챗봇을 종료합니다.")
                    break
                
                if not question:
                    print("질문을 입력해주세요.")
                    continue
                
                print()
                print("처리 중...")
                print("-" * 30)
                
                # 질문 처리
                result = chatbot.process_question(question)
                
                print()
                print("처리 결과:")
                print(f"  성공: {result['success']}")
                print(f"  질문 유형: {result['question_type']}")
                print(f"  라우팅: {result['route']}")
                print(f"  처리 시간: {result['processing_time']:.3f}초")
                print(f"  신뢰도: {result['confidence_score']:.3f}")
                
                # Answer Target 정보 표시 (새로운 기능)
                if hasattr(result, 'answer_target') and result.get('answer_target'):
                    print(f"  답변 목표: {result.get('answer_target', 'N/A')}")
                    print(f"  목표 유형: {result.get('target_type', 'N/A')}")
                    print(f"  값의 의도: {result.get('value_intent', 'N/A')}")
                    print(f"  추출 신뢰도: {result.get('confidence_score', 'N/A')}")
                
                if result.get('sql_query'):
                    print(f"  생성된 SQL: {result['sql_query']}")
                
                if result.get('used_chunks'):
                    print(f"  사용된 청크: {len(result['used_chunks'])}개")
                
                print()
                print("답변:")
                print(result['answer'])
                print()
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\nCtrl+C로 종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")
                print()
    
    except Exception as e:
        print(f"챗봇 초기화 실패: {e}")
        try:
            from utils.chatbot_logger import log_exception_to_error_log
            log_exception_to_error_log(f"INIT ERROR | {e}")
        except Exception:
            pass
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    finally:
        # 프로그램 종료 시 stdout 복원
        if 'original_stdout' in globals():
            sys.stdout = original_stdout
