#!/usr/bin/env python3
"""
범용 RAG 시스템 서버 (최적화 버전)

이 스크립트는 Dual Pipeline이 통합된 범용 RAG 시스템을 
FastAPI 서버 모드로 실행합니다.

주요 최적화 사항:
- CUDA GPU 지원 및 CPU 폴백
- 최적화된 캐싱 시스템
- 비동기 로깅
- 자동 메모리 관리
"""

import sys
import os
import logging
import time
import torch
from pathlib import Path
from typing import Optional

# 최적화 모듈 import
try:
    from core.config.optimization_config import get_optimization_config
    from core.utils.memory_optimizer import memory_optimizer, start_memory_monitoring
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logging.warning("최적화 모듈을 찾을 수 없습니다. 기본 모드로 실행됩니다.")

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 로그 디렉토리 생성 (권한 문제 방지)
log_dir = (Path(__file__).parent / "logs")
try:
    log_dir.mkdir(exist_ok=True)
except Exception:
    pass

# 로깅 설정 (import 전에 설정) - 콘솔 출력 제거, 파일로만 기록
logging.basicConfig(
    level=logging.CRITICAL,  # 로그 출력 최소화
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    handlers=[
        logging.NullHandler()
    ]
)

# 불필요한 로그 레벨 조정 - 더 많은 라이브러리 추가
logging.getLogger('faiss.loader').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('numpy').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)
logging.getLogger('uvicorn').setLevel(logging.WARNING)
logging.getLogger('fastapi').setLevel(logging.WARNING)

# 챗봇 관련 로거는 INFO 레벨로 설정 (중요한 메시지는 표시)
logging.getLogger('core.document.pdf_keyword_extractor').setLevel(logging.INFO)
logging.getLogger('core').setLevel(logging.INFO)
logging.getLogger('api').setLevel(logging.INFO)
logging.getLogger('utils').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# 시작 시 강제 데이터 초기화 유틸리티
def force_startup_cleanup():
    """시작 시 Chroma/FAISS/전처리 DB를 강제 초기화한다."""
    try:
        import shutil
        import time
        base_dir = current_dir

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

        # error.log 초기화 (강화된 재시도 로직)
        try:
            (base_dir / "logs").mkdir(exist_ok=True)
            _safe_write_file(error_log, "", "error.log 초기화")
            logger.info(f"[CLEANUP] error.log 초기화: {error_log}")
        except Exception as e:
            logger.warning(f"[CLEANUP] error.log 초기화 실패: {e}")

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

# 핵심 모듈 import 시도 (에러 처리 포함)
try:
    from api.endpoints import app, initialize_system
    from utils.chatbot_logger import chatbot_logger
    logger.info("[SUCCESS] API 모듈 import 성공")
    
    # 챗봇 로거 초기화 확인
    if chatbot_logger:
        logger.info("[SUCCESS] 챗봇 로거 초기화 완료")
    else:
        logger.warning("[WARNING] 챗봇 로거 초기화 실패, 기본 로깅 사용")
        
except ImportError as e:
    logger.error(f"[ERROR] API 모듈 import 실패: {e}")
    logger.error("현재 Python 경로:")
    for i, path in enumerate(sys.path):
        logger.error(f"  {i}: {path}")
    sys.exit(1)
except Exception as e:
    logger.error(f"[ERROR] 예상치 못한 import 오류: {e}")
    sys.exit(1)


def setup_device():
    """CUDA GPU 설정 및 CPU 폴백"""
    try:
        # CUDA 사용 가능 여부 확인
        if torch.cuda.is_available():
            # GPU 개수 확인
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logger.info(f"[GPU] CUDA 사용 가능: {gpu_count}개 GPU")
            logger.info(f"[GPU] 현재 GPU: {device_name}")
            logger.info(f"[GPU] GPU 메모리: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.1f}GB")
            
            # GPU 메모리 정보 출력
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"[GPU] GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
            
            device = torch.device("cuda")
            logger.info(f"[SUCCESS] CUDA GPU 사용: {device}")
            return device
            
        else:
            logger.warning("[WARNING] CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
            logger.info("[INFO] CPU 사용: torch.device('cpu')")
            return torch.device("cpu")
            
    except Exception as e:
        logger.error(f"[ERROR] 디바이스 설정 중 오류: {e}")
        logger.warning("[WARNING] CPU로 폴백합니다.")
        return torch.device("cpu")


class ChatbotServer:
    """최적화된 챗봇 서버 관리 클래스"""
    
    def __init__(self):
        # 통합 설정으로 치환
        try:
            from core.config.unified_config import get_config
        except Exception:
            def get_config(key, default=None):
                return os.getenv(key, default)
        self.model_type = str(get_config("MODEL_TYPE", "local"))
        self.model_name = str(get_config("MODEL_NAME", "beomi/KoAlpaca-Polyglot-5.8B"))
        self.embedding_model = str(get_config("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask"))
        
        # 디바이스 설정
        self.device = setup_device()
        
        # 최적화 설정 로드
        if OPTIMIZATION_AVAILABLE:
            self.config = get_optimization_config()
            logger.info("최적화 모드로 서버 초기화")
        else:
            self.config = None
            
        logger.info(f"서버 설정: MODEL_TYPE={self.model_type}, MODEL_NAME={self.model_name}")
        logger.info(f"사용 디바이스: {self.device}")
        
        # 메모리 모니터링 시작
        if OPTIMIZATION_AVAILABLE and self.config.performance.memory_monitoring:
            memory_optimizer.warning_threshold = self.config.performance.memory_warning_threshold
            memory_optimizer.critical_threshold = self.config.performance.memory_critical_threshold
            start_memory_monitoring()
        
    def check_local_model_available(self) -> bool:
        """로컬 모델이 사용 가능한지 확인 (개선된 버전)"""
        try:
            logger.info(f"로컬 모델 확인 중: {self.model_name}")
            
            # transformers 라이브러리 확인
            try:
                from transformers import AutoTokenizer, AutoModel
            except ImportError:
                logger.error("transformers 라이브러리가 설치되지 않았습니다.")
                return False
            
            # 모델 파일 존재 여부 확인 (실제 다운로드 확인)
            from transformers import AutoConfig
            try:
                config = AutoConfig.from_pretrained(self.model_name, cache_dir="./models")
                logger.info(f"모델 설정 확인 완료: {config.model_type}")
                return True
            except Exception as e:
                logger.warning(f"모델 설정 확인 실패: {e}")
                return False
                
        except Exception as e:
            logger.warning(f"로컬 모델 확인 실패: {e}")
            return False
    
    def download_and_load_models(self) -> bool:
        """필요한 모델들을 완전히 다운로드하고 메모리에 로드 (GPU/CPU 지원)"""
        try:
            logger.info("[AI] 모든 AI 모델을 다운로드하고 메모리에 로드합니다...")
            logger.info(f"[DEVICE] 사용 디바이스: {self.device}")
            
            models_to_load = [
                {
                    "id": "beomi/KoAlpaca-Polyglot-5.8B",
                    "name": "KoAlpaca-Polyglot-5.8B (답변 생성)",
                    "type": "causal"
                },
                {
                    "id": "jhgan/ko-sroberta-multitask", 
                    "name": "Ko-SRoBERTa (임베딩/SBERT)",
                    "type": "sbert"
                },
            ]
            
            success_count = 0
            total_count = len(models_to_load)
            
            for model_info in models_to_load:
                try:
                    logger.info(f"[DOWNLOAD] {model_info['name']} 다운로드 및 로드 중: {model_info['id']}")
                    
                    if model_info['type'] == 'sbert':
                        from sentence_transformers import SentenceTransformer
                        logger.info(f"[LOADING] {model_info['name']} SBERT 모델 로드 중...")
                        
                        # GPU 사용 가능한 경우 GPU로 로드
                        if self.device.type == "cuda":
                            model = SentenceTransformer(
                                model_info['id'], 
                                cache_folder="./models",
                                device=str(self.device)
                            )
                        else:
                            model = SentenceTransformer(
                                model_info['id'], 
                                cache_folder="./models"
                            )
                        
                        # 테스트 임베딩 생성으로 모델이 제대로 로드되었는지 확인
                        test_embedding = model.encode("테스트 문장", convert_to_tensor=True)
                        if self.device.type == "cuda":
                            test_embedding = test_embedding.to(self.device)
                        
                        logger.info(f"[SUCCESS] {model_info['name']} SBERT 모델 로드 완료 (임베딩 차원: {test_embedding.shape})")
                        
                    elif model_info['type'] == 'sqlcoder':
                        from transformers import AutoTokenizer, AutoModelForCausalLM
                        logger.info(f"[LOADING] {model_info['name']} SQLCoder 모델 로드 중...")
                        
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_info['id'],
                            cache_dir="./models",
                            trust_remote_code=True
                        )
                        
                        # GPU 사용 가능한 경우 GPU로 로드
                        if self.device.type == "cuda":
                            model = AutoModelForCausalLM.from_pretrained(
                                model_info['id'],
                                cache_dir="./models",
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                torch_dtype=torch.float16,  # GPU 메모리 절약
                                device_map="auto"  # 자동 GPU 매핑
                            )
                        else:
                            model = AutoModelForCausalLM.from_pretrained(
                                model_info['id'],
                                cache_dir="./models",
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                torch_dtype=torch.float32  # CPU는 float32 사용
                            )
                        
                        # 테스트 생성으로 모델이 제대로 로드되었는지 확인
                        test_input = tokenizer("SELECT", return_tensors="pt")
                        if self.device.type == "cuda":
                            test_input = {k: v.to(self.device) for k, v in test_input.items()}
                        
                        with torch.no_grad():
                            test_output = model.generate(**test_input, max_length=10)
                        
                        logger.info(f"[SUCCESS] {model_info['name']} SQLCoder 모델 로드 완료")
                    
                    elif model_info['type'] == 'causal':
                        from transformers import AutoTokenizer, AutoModelForCausalLM
                        logger.info(f"[LOADING] {model_info['name']} Causal 모델 로드 중...")
                        
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_info['id'],
                            cache_dir="./models",
                            trust_remote_code=True
                        )
                        
                        # GPU 사용 가능한 경우 GPU로 로드
                        if self.device.type == "cuda":
                            model = AutoModelForCausalLM.from_pretrained(
                                model_info['id'],
                                cache_dir="./models",
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                torch_dtype=torch.float16,  # GPU 메모리 절약
                                device_map="auto"  # 자동 GPU 매핑
                            )
                        else:
                            model = AutoModelForCausalLM.from_pretrained(
                                model_info['id'],
                                cache_dir="./models",
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                torch_dtype=torch.float32  # CPU는 float32 사용
                            )
                        
                        # 테스트 생성으로 모델이 제대로 로드되었는지 확인
                        test_input = tokenizer("안녕하세요", return_tensors="pt")
                        if self.device.type == "cuda":
                            test_input = {k: v.to(self.device) for k, v in test_input.items()}
                        
                        with torch.no_grad():
                            test_output = model.generate(**test_input, max_length=10)
                        
                        logger.info(f"[SUCCESS] {model_info['name']} Causal 모델 로드 완료")
                    
                    success_count += 1
                    logger.info(f"[PROGRESS] {success_count}/{total_count} 모델 로드 완료")
                    
                except Exception as e:
                    logger.error(f"[ERROR] {model_info['name']} 로드 실패: {e}")
                    logger.error("상세 오류 정보:")
                    import traceback
                    traceback.print_exc()
                    continue
            
            logger.info(f"[STATS] 모델 로드 결과: {success_count}/{total_count} 성공")
            
            if success_count == total_count:
                logger.info("[SUCCESS] 모든 모델이 성공적으로 로드되었습니다!")
                return True
            elif success_count > 0:
                logger.warning(f"[WARNING] 일부 모델만 로드되었습니다 ({success_count}/{total_count})")
                return True
            else:
                logger.error("[ERROR] 모든 모델 로드에 실패했습니다.")
                return False
            
        except Exception as e:
            logger.error(f"[ERROR] 모델 다운로드 및 로드 중 오류 발생: {e}")
            logger.error("상세 오류 정보:")
            import traceback
            traceback.print_exc()
            return False
    
    def print_startup_banner(self):
        """시작 배너 출력 (콘솔 소음 방지를 위해 파일 로그만 기록)"""
        logger.info("[START] 범용 RAG 시스템 서버 시작")
        logger.info("[AI] 최적화된 버전 - 빠른 응답!")
        logger.info(f"[DEVICE] 사용 디바이스: {self.device}")
    
    def initialize_system(self) -> bool:
        """시스템 초기화 (에러 처리 개선)"""
        try:
            logger.info("시스템 초기화 및 자동 PDF 업로드 시작...")
            
            # 시스템 초기화 (PDF 자동 업로드 포함)
            initialize_system()
            
            logger.info("[SUCCESS] 시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] 시스템 초기화 실패: {e}")
            logger.error("상세 오류 정보:")
            import traceback
            traceback.print_exc()
            
            # 초기화 실패 시에도 계속 진행할지 확인
            logger.warning("[WARNING] 시스템 초기화에 실패했지만 서버는 시작됩니다.")
            logger.warning("[WARNING] 일부 기능이 제한될 수 있습니다.")
            return False
    
    def start_server(self):
        """FastAPI 서버 시작 (에러 처리 개선)"""
        try:
            logger.info("[WEB] API 서버를 시작합니다...")
            logger.info(f"서버 주소: http://0.0.0.0:8008")
            
            # uvicorn 서버 실행
            import uvicorn
            uvicorn.run(
                app, 
                host="0.0.0.0", 
                port=8008,
                log_level="warning",
                access_log=False,
                reload=False  # 프로덕션 환경에서는 False
            )
            
        except KeyboardInterrupt:
            logger.info("서버가 중단되었습니다.")
        except Exception as e:
            logger.error(f"❌ 서버 실행 중 오류: {e}")
            logger.error("상세 오류 정보:")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def run(self):
        """메인 실행 로직 (에러 처리 개선)"""
        try:
            # 시작 배너는 파일 로그로만 기록하고, 콘솔에는 1회만 알림
            self.print_startup_banner()
            print("서버 시작 준비 완료.")
            
            # 로컬 모델 준비 확인 및 완전 로드
            if self.model_type == "local":
                logger.info("[CHECK] 모든 AI 모델을 다운로드하고 메모리에 로드합니다...")
                
                # 모든 모델을 완전히 다운로드하고 메모리에 로드
                if not self.download_and_load_models():
                    logger.error("모델 로드에 실패했습니다.")
                    logger.error("해결 방법:")
                    logger.error("1. 인터넷 연결 확인")
                    logger.error("2. 충분한 디스크 공간 확인")
                    logger.error("3. 충분한 메모리 확인")
                    logger.error("4. python setup_sbert.py 실행")
                    
                    # 사용자에게 계속 진행할지 묻기
                    logger.warning("모델 로드에 실패했지만 서버를 시작할까요? (제한된 기능)")
                    logger.warning("계속하려면 Ctrl+C를 누르고 다시 실행하세요.")
                    
                    # 10초 대기 후 종료
                    for i in range(10, 0, -1):
                        logger.info(f"[EXIT] {i}초 후 서버가 종료됩니다...")
                        time.sleep(1)
                    sys.exit(1)
                else:
                    logger.info("[SUCCESS] 모든 AI 모델이 성공적으로 로드되었습니다!")
                    logger.info("[READY] 이제 질문을 받을 준비가 완료되었습니다!")
            
            # 시스템 초기화
            logger.info("[INIT] 시스템 초기화 중...")
            if not self.initialize_system():
                logger.warning("시스템 초기화에 실패했지만 계속 진행합니다.")
            
            # 서버 시작
            logger.info("[START] 서버 시작 중...")
            self.start_server()
            
        except Exception as e:
            logger.error(f"서버 실행 중 예상치 못한 오류: {e}")
            logger.error("상세 오류 정보:")
            import traceback
            traceback.print_exc()
            try:
                from utils.chatbot_logger import log_exception_to_error_log
                log_exception_to_error_log(f"SERVER RUN ERROR | {e}")
            except Exception:
                pass
            sys.exit(1)


def main():
    """메인 함수 (에러 처리 개선)"""
    try:
        logger.info("범용 RAG 시스템 서버 시작")
        # 강제 데이터 초기화
        force_startup_cleanup()
        
        # 서버 인스턴스 생성 및 실행
        server = ChatbotServer()
        server.run()
        
    except Exception as e:
        logger.error(f"메인 함수 실행 중 오류: {e}")
        logger.error("상세 오류 정보:")
        import traceback
        traceback.print_exc()
        try:
            from utils.chatbot_logger import log_exception_to_error_log
            log_exception_to_error_log(f"SERVER MAIN ERROR | {e}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
