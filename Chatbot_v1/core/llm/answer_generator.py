"""
KoAlpaca 기반 답변 생성 모듈

메모리 효율성과 빠른 응답을 위한 최적화된 모듈
"""

import time
import logging
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os
import re

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# 로컬 LLM 라이브러리 (지연 import로 최적화)
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch를 찾을 수 없습니다.")

# 상대 경로 import (필요시에만 로딩)
from core.document.text_chunk import TextChunk
from core.query.question_analyzer import AnalyzedQuestion
from core.cache.fast_cache import get_question_cache
from core.utils.memory_optimizer import model_memory_manager, memory_profiler
from core.document.units import normalize_unit
from core.document.defined_value_detector import DefinedValueDetector, DefinedValue
from core.llm.hallucination_prevention import HallucinationPrevention, create_hallucination_prevention
from utils.qa_logger import log_question_answer

# 통합 설정 import (폴백 포함)
try:
    from core.config.unified_config import get_config
except ImportError:
    def get_config(key, default=None):
        return os.getenv(key, default)

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """모델 타입 열거형"""
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

@dataclass
class GenerationConfig:
    """경량화된 생성 설정 (RAG 최적화)"""
    max_length: int = 1024
    temperature: float = 0.3  # 창의성보다 정확성 우선
    top_p: float = 0.8  # 더 집중된 토큰 선택
    do_sample: bool = True
    num_beams: int = 3
    early_stopping: bool = True

@dataclass
class Answer:
    """답변 데이터 클래스"""
    content: str
    confidence: float
    sources: List[TextChunk]
    generation_time: float
    model_name: str
    metadata: Optional[Dict] = None

class ModelManager:
    """싱글톤 패턴을 사용한 모델 관리자 (메모리 최적화 적용)"""
    _instance = None
    _models = {}
    _tokenizers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str, model_type: str = "seq2seq"):
        """지연 로딩을 통한 모델 획득 (메모리 매니저 통합)"""
        # 먼저 메모리 매니저에서 확인
        model_obj = model_memory_manager.get_model(model_name)
        if model_obj is not None:
            # 모델과 토크나이저 분리 (model_obj는 튜플로 저장됨)
            if isinstance(model_obj, tuple) and len(model_obj) == 2:
                return model_obj[0], model_obj[1]
            return model_obj, self._tokenizers.get(model_name)
        
        # 메모리 매니저에 없으면 새로 로드
        logger.info(f"모델 지연 로딩 중: {model_name}")
        return self._load_model(model_name, model_type)
    
    def _load_model(self, model_name: str, model_type: str):
        """실제 모델 로딩 로직 (메모리 매니저 통합) - GPU/CPU 지원"""
        try:
            with memory_profiler(f"모델 로딩: {model_name}"):
                # 필요시에만 transformers import
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
                
                # CUDA 사용 가능 여부 확인
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"[DEVICE] 모델 로딩 디바이스: {device}")
                
                # 토크나이저 로드 (캐시 디렉토리 최적화)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir="./models",
                    use_fast=True,  # 빠른 토크나이저 사용
                    trust_remote_code=True
                )
                
                # 모델 타입에 따른 로딩 (GPU/CPU 지원)
                if model_type == "causal":
                    if device.type == "cuda":
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,  # GPU 메모리 절약
                            cache_dir="./models",
                            low_cpu_mem_usage=True,
                            device_map="auto",  # 자동 GPU 매핑
                            trust_remote_code=True
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,  # CPU는 float32 사용
                            cache_dir="./models",
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                elif model_type == "seq2seq":
                    if device.type == "cuda":
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,  # GPU 메모리 절약
                            cache_dir="./models",
                            low_cpu_mem_usage=True,
                            device_map="auto",  # 자동 GPU 매핑
                            trust_remote_code=True
                        )
                    else:
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,  # CPU는 float32 사용
                            cache_dir="./models",
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                
                # 메모리 매니저에 등록
                model_tuple = (model, tokenizer)
                estimated_memory = self._estimate_model_memory(model_name)
                model_memory_manager.register_model(model_name, model_tuple, estimated_memory)
                
                # 로컬 캐시에도 저장 (호환성)
                self._models[model_name] = model
                self._tokenizers[model_name] = tokenizer
                
                logger.info(f"모델 로딩 완료: {model_name} (예상 메모리: {estimated_memory:.2f}GB)")
                return model, tokenizer
                
        except Exception as e:
            logger.error(f"모델 로딩 실패 {model_name}: {e}")
            # 오프라인 모드 시도
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
                
                # CUDA 사용 가능 여부 확인
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"[DEVICE] 오프라인 모델 로딩 디바이스: {device}")
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir="./models",
                    use_fast=True,
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                # 모델 타입에 따른 로딩 (오프라인 모드, GPU/CPU 지원)
                if "alpaca" in model_name.lower() or "polyglot" in model_name.lower():
                    if device.type == "cuda":
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            cache_dir="./models",
                            low_cpu_mem_usage=True,
                            device_map="auto",
                            local_files_only=True,
                            trust_remote_code=True
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,
                            cache_dir="./models",
                            low_cpu_mem_usage=True,
                            local_files_only=True,
                            trust_remote_code=True
                        )
                else:
                    if device.type == "cuda":
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            cache_dir="./models",
                            low_cpu_mem_usage=True,
                            device_map="auto",
                            local_files_only=True,
                            trust_remote_code=True
                        )
                    else:
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,
                            cache_dir="./models",
                            low_cpu_mem_usage=True,
                            local_files_only=True,
                            trust_remote_code=True
                        )
                
                # 메모리 매니저에 등록
                model_tuple = (model, tokenizer)
                estimated_memory = self._estimate_model_memory(model_name)
                model_memory_manager.register_model(model_name, model_tuple, estimated_memory)
                
                self._models[model_name] = model
                self._tokenizers[model_name] = tokenizer
                
                logger.info(f"오프라인 모델 로딩 완료: {model_name}")
                return model, tokenizer
                
            except Exception as offline_error:
                logger.error(f"오프라인 모델 로딩도 실패 {model_name}: {offline_error}")
                raise
    
    def _estimate_model_memory(self, model_name: str) -> float:
        """모델별 예상 메모리 사용량 추정 (GB)"""
        model_memory_map = {
            "beomi/KoAlpaca-Polyglot-5.8B": 3.0,
            "jhgan/ko-sroberta-multitask": 0.5,
            # SQL 관련 모델 제거됨
        }
        return model_memory_map.get(model_name, 1.0)
    
    def clear_models(self):
        """메모리 정리를 위한 모델 해제"""
        # 메모리 매니저의 모든 모델 언로드
        model_status = model_memory_manager.get_model_status()
        for model_name in model_status["loaded_models"]:
            model_memory_manager.unload_model(model_name)
        
        # 로컬 캐시 정리
        for model_name in list(self._models.keys()):
            del self._models[model_name]
            del self._tokenizers[model_name]
        self._models.clear()
        self._tokenizers.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("모든 모델 메모리 해제 완료")

# 전역 모델 매니저 인스턴스
model_manager = ModelManager()

class OllamaLLMInterface:
    """Ollama 로컬 서버를 통한 LLM 인터페이스 (Qwen 등)"""
    def __init__(self, model_name: str = "qwen2.5:latest", base_url: str = None, config: GenerationConfig = None):
        self.model_name = model_name
        # 기본 URL 정규화 (말단 슬래시 제거)
        self.base_url = (base_url or get_config("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.config = config or GenerationConfig()
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests 라이브러리가 필요합니다. pip install requests")
    
    def generate(self, prompt: str) -> str:
        try:
            # Qwen은 대화형이지만, 간단히 단일 프롬프트로 generate 사용
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_beams": self.config.num_beams,
                    "num_predict": self.config.max_length
                }
            }
            url = f"{self.base_url}/api/generate"
            last_err = None
            retries = int(get_config("LLM_RETRY", 2))
            backoff = float(get_config("LLM_BACKOFF_S", 0.5))
            for attempt in range(retries + 1):
                try:
                    resp = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=120)
                    break
                except Exception as e:
                    last_err = e
                    if attempt < retries:
                        time.sleep(backoff * (2 ** attempt))
                        continue
                    raise
            if resp.status_code == 404:
                # 일부 환경에서 /api/generate 미지원 → /api/chat로 폴백
                chat_payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "num_beams": self.config.num_beams,
                        "num_predict": self.config.max_length
                    }
                }
                chat_url = f"{self.base_url}/api/chat"
                chat_resp = requests.post(chat_url, data=json.dumps(chat_payload), headers={"Content-Type": "application/json"}, timeout=120)
                chat_resp.raise_for_status()
                chat_data = chat_resp.json()
                # /api/chat는 'message' 혹은 'response' 사용
                if isinstance(chat_data, dict):
                    if "message" in chat_data and isinstance(chat_data["message"], dict):
                        return chat_data["message"].get("content", "")
                    return chat_data.get("response", "")
                return ""
            resp.raise_for_status()
            data = resp.json()
            # Ollama /api/generate 는 'response' 필드에 텍스트를 담음
            return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama 생성 실패: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

class LocalLLMInterface:
    """최적화된 KoAlpaca-Polyglot-5.8B 인터페이스 (메모리 최적화 적용)"""
    
    def __init__(self, model_name: str = "beomi/KoAlpaca-Polyglot-5.8B", config: GenerationConfig = None):
        self.model_name = model_name
        self.config = config or GenerationConfig()
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
    def _ensure_model_loaded(self):
        """모델이 로드되지 않았다면 로드"""
        if not self._model_loaded:
            logger.info(f"지연 로딩: {self.model_name}")
            self.model, self.tokenizer = model_manager.get_model(self.model_name, "causal")
            self._model_loaded = True
        
    def generate(self, prompt: str) -> str:
        """KoAlpaca-Polyglot-1.3B 기반 실제 텍스트 생성 (메모리 프로파일링 적용)"""
        try:
            with memory_profiler(f"텍스트 생성: {self.model_name}"):
                # 모델 지연 로딩 확인
                self._ensure_model_loaded()
                
                # 입력 텍스트 전처리
                input_text = f"### 질문: {prompt}\n\n### 답변:"
                
                # 토크나이징 (causal 모델용) - token_type_ids 제거
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                
                # token_type_ids가 있는 경우 제거 (causal 모델에서는 불필요)
                if 'token_type_ids' in inputs:
                    inputs.pop('token_type_ids')
                
                # GPU 사용 가능시 GPU로 이동 (개선된 버전)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if device.type == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    self.model = self.model.to(device)
                    logger.debug(f"[GPU] 모델과 입력을 GPU로 이동: {device}")
                else:
                    logger.debug(f"[CPU] CPU에서 실행: {device}")
                
                # 생성 (causal 모델용)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_length,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                        num_beams=self.config.num_beams,
                        early_stopping=self.config.early_stopping,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # 디코딩
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 입력 프롬프트 제거
                if "### 답변:" in generated_text:
                    answer = generated_text.split("### 답변:")[-1].strip()
                else:
                    answer = generated_text.strip()
                
                return answer
                
        except Exception as e:
            logger.error(f"텍스트 생성 실패: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

class AnswerGenerator:
    """경량화된 답변 생성기 (메모리 최적화 적용)"""
    
    def __init__(self, model_name: str = "qwen2:1.5b-instruct-q4_K_M", cache_enabled: bool = True, preload_models: bool = False):
        # Qwen(Ollama) 고정
        self.model_type = ModelType.OLLAMA.value
        # SSOT로 모델명 조회 (환경변수/JSON/기본 순)
        self.model_name = str(get_config("MODEL_NAME", model_name))
        self.cache_enabled = cache_enabled
        
        # LLM 인터페이스 초기화: Ollama(Qwen)만 사용
        logger.info(f"Ollama(Qwen) 모드로 초기화: MODEL_NAME={self.model_name}")
        self.llm = OllamaLLMInterface(self.model_name)
        
        # 프리로드는 전부 비활성화
        # (Ollama는 필요 시 자동 제공, 로컬/다른 모델 사용 안 함)
        
        # 회사 컨텍스트 설정
        # 통합 설정 사용
        self.company_name = get_config('COMPANY_NAME')
        self.project_name = get_config('PROJECT_NAME')
        
        # 엄격한 프롬프트 템플릿 사용 (환상 방지 강화)
        try:
            from core.config.strict_prompt_templates import get_strict_templates
            self.prompt_template = get_strict_templates('wastewater', variant='A', version='1')
            logger.info("엄격한 프롬프트 템플릿 적용 (환상 방지 강화)")
        except Exception:
            # 폴백: 기본 템플릿
            domain = str(get_config('DOMAIN_TEMPLATE', 'general'))
            variant = str(get_config('PROMPT_VARIANT', 'A'))
            version = str(get_config('PROMPT_VERSION', '1'))
            try:
                from core.config.prompt_catalog import get_templates
                self.prompt_template = get_templates(domain, variant=variant, version=version)
            except Exception:
                # 최종 폴백: general A v1
                from core.config.prompt_catalog import get_templates
                self.prompt_template = get_templates('general', variant='A', version='1')
        
        # 정의된 값 감지기 초기화
        self.defined_value_detector = DefinedValueDetector()
        
        # 환상 방지 시스템 초기화
        self.hallucination_prevention = create_hallucination_prevention()
        
        logger.info(f"경량화된 답변 생성기 초기화 완료: {model_name}")
    
    # ==== 가독성 포맷터 (모듈 내부 유틸) ====
    @staticmethod
    def _format_answer_for_readability(original_text: str) -> str:
        """답변을 자연스러운 문단 형태로 정리 (번호 목록 제거)
        규칙:
        - 번호 목록(1. 2. 3.)이나 불릿 포인트(-, *) 제거
        - 강조 표시(***) 제거
        - 자연스러운 문단 형태로 변환
        """
        try:
            if not original_text:
                return original_text
            
            text = original_text.strip()
            import re as _re
            
            # 번호 목록 제거 (1. 2. 3. 등)
            text = _re.sub(r'^\d+\.\s*', '', text, flags=_re.MULTILINE)
            
            # 불릿 포인트 제거 (-, *, • 등)
            text = _re.sub(r'^[-*•]\s*', '', text, flags=_re.MULTILINE)
            
            # 강조 표시 제거 (***, **, * 등)
            text = _re.sub(r'\*{1,3}', '', text)
            
            # 줄바꿈을 공백으로 변환하여 자연스러운 문단으로 만들기
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            
            # 빈 줄로 구분된 문단들을 하나의 문단으로 합치기
            paragraphs = []
            current_paragraph = []
            
            for line in lines:
                if line:
                    current_paragraph.append(line)
                else:
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
            
            # 마지막 문단 처리
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            # 문단들을 자연스럽게 연결
            result = ' '.join(paragraphs)
            
            # 연속된 공백 정리
            result = _re.sub(r'\s+', ' ', result).strip()
            
            return result
            
        except Exception:
            return original_text

    @staticmethod
    def _limit_sentences(answer_text: str, max_sentences: int = 3) -> str:
        """답변을 최대 N문장으로 제한한다."""
        try:
            if not answer_text:
                return answer_text
            import re as _re
            text = answer_text.strip()
            # 문장 분리 (간단 규칙: 마침표/물음표/느낌표)
            sentences = _re.split(r"(?<=[\.\?\!])\s+", text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) <= max_sentences:
                return text
            return " ".join(sentences[:max_sentences]).strip()
        except Exception:
            return answer_text

    def _preload_models(self):
        """필요한 모델들을 미리 로딩"""
        # 완전 비활성화 (Qwen 고정)
        logger.info("프리로드 비활성화: Qwen(Ollama)만 사용")
    
    def load_model(self) -> bool:
        """모델 로드 (로컬 LLM은 이미 로드되어 있음)"""
        try:
            # 로컬 LLM은 이미 초기화 시 로드되므로 성공으로 간주
            logger.info(f"로컬 LLM 모델 {self.model_name} 로드 완료")
            return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False
    
    def unload_model(self):
        """경량화된 모델 언로드"""
        try:
            if hasattr(self.llm, 'model'):
                del self.llm.model
            if hasattr(self.llm, 'tokenizer'):
                del self.llm.tokenizer
            import gc
            gc.collect()
            logger.info(f"경량화된 모델 {self.model_name} 언로드 완료")
        except Exception as e:
            logger.error(f"모델 언로드 실패: {e}")
    
    def generate_direct_answer(self, question: str) -> Answer:
        """문서/데이터 없이 회사 컨텍스트 기반 직접 답변 생성"""
        start_time = time.time()
        
        try:
            # 회사 컨텍스트 기반 프롬프트 생성
            context_prompt = self.prompt_template["no_context"].format(question=question)
            
            # 답변 생성
            raw_text = self.llm.generate(context_prompt)
            answer_text = self._postprocess_answer(raw_text)
            answer_text = self._format_answer_for_readability(answer_text)
            
            generation_time = time.time() - start_time
            
            answer = Answer(
                content=answer_text,
                confidence=0.8,  # 직접 답변은 신뢰도가 낮음
                sources=[],
                generation_time=generation_time,
                model_name=self.model_name,
                metadata={"answer_type": "direct", "context_used": False}
            )
            
            # QA 로그 기록
            try:
                log_question_answer(
                    question=question,
                    answer=answer_text,
                    chunks_used=[],
                    generation_time_ms=generation_time * 1000,
                    confidence_score=0.8,
                    model_name=self.model_name,
                    pipeline_type="direct",
                    session_id=answer.metadata.get("session_id") if answer.metadata else None
                )
            except Exception as e:
                logger.debug(f"QA 로그 기록 실패: {e}")
            
            return answer
            
        except Exception as e:
            logger.error(f"직접 답변 생성 실패: {e}")
            return Answer(
                content=f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
                confidence=0.0,
                sources=[],
                generation_time=time.time() - start_time,
                model_name=self.model_name,
                metadata={"error": str(e)}
            )
    
    def generate_context_answer(self, question: str, context_chunks: List[TextChunk], 
                              analyzed_question=None) -> Answer:
        """컨텍스트 기반 답변 생성 (목표 기반 프롬프트 강화)"""
        start_time = time.time()
        
        try:
            # 컨텍스트 텍스트 생성 (가독성 위해 상위 2개로 제한)
            trimmed_chunks = context_chunks[:6]
            
            # 질문 분석 결과에서 목표 정보 추출
            answer_target = ""
            target_type = ""
            value_intent = ""
            if analyzed_question and hasattr(analyzed_question, 'answer_target'):
                answer_target = analyzed_question.answer_target
                target_type = analyzed_question.target_type
                value_intent = analyzed_question.value_intent
            # 사용된 청크의 출처 정보를 로깅으로 출력
            try:
                logger.debug("[PDF 출처] 답변 생성에 사용된 컨텍스트 청크:")
                for idx, ch in enumerate(trimmed_chunks, start=1):
                    filename = getattr(ch, 'filename', None)
                    pdf_id = getattr(ch, 'pdf_id', None)
                    page_number = getattr(ch, 'page_number', None)
                    chunk_id = getattr(ch, 'chunk_id', None)
                    # 메타데이터에 파일명/페이지가 있을 수 있음
                    if not filename and getattr(ch, 'metadata', None):
                        filename = ch.metadata.get('filename') or ch.metadata.get('source') or ch.metadata.get('title')
                    if page_number is None and getattr(ch, 'metadata', None):
                        page_number = ch.metadata.get('page') or ch.metadata.get('page_number')
                    parts = []
                    if filename:
                        parts.append(f"파일: {filename}")
                    if pdf_id and not filename:
                        parts.append(f"PDF ID: {pdf_id}")
                    if page_number is not None:
                        parts.append(f"페이지: {page_number}")
                    if chunk_id:
                        parts.append(f"청크ID: {chunk_id}")
                    # 짧은 미리보기
                    preview = (getattr(ch, 'content', '') or '')[:80].replace("\n", " ")
                    info_line = " | ".join(parts) if parts else "출처 정보 없음"
                    logger.debug(f"  - #{idx}: {info_line}")
                    if preview:
                        logger.debug(f"      내용: {preview}...")
            except Exception as _e:
                # print 단계 오류는 무시 (답변 생성에는 영향 주지 않음)
                logger.debug(f"출처 print 중 오류 무시: {_e}")
            
            # 정의된 값 감지 및 가중치 적용
            all_defined_values = []
            enhanced_context_parts = []
            
            for chunk in trimmed_chunks:
                chunk_content = chunk.content
                page_number = getattr(chunk, 'page_number', None)
                source = getattr(chunk, 'filename', None) or getattr(chunk, 'pdf_id', None)
                
                # 청크에서 정의된 값들 감지
                defined_values = self.defined_value_detector.detect_defined_values(
                    chunk_content, page_number, source
                )
                all_defined_values.extend(defined_values)
                
                # 정의된 값이 있는 청크는 가중치를 높여서 컨텍스트에 포함
                if defined_values:
                    # 높은 가중치의 정의된 값들을 강조
                    high_priority_values = self.defined_value_detector.get_high_priority_values(defined_values)
                    if high_priority_values:
                        # 정의된 값들을 강조 표시
                        enhanced_content = self._enhance_context_with_defined_values(chunk_content, high_priority_values)
                        enhanced_context_parts.append(enhanced_content)
                    else:
                        enhanced_context_parts.append(chunk_content)
                else:
                    enhanced_context_parts.append(chunk_content)
            
            # 질문과 관련된 정의된 값들 추출
            relevant_defined_values = self.defined_value_detector.extract_values_for_question(all_defined_values, question)
            
            # 컨텍스트 텍스트 생성 (정의된 값이 강조된 버전)
            context_text = "\n\n".join(enhanced_context_parts)
            
            # 사용된 키워드 추출 (로깅 최소화)
            used_keywords = self._extract_used_keywords(trimmed_chunks)
            if used_keywords:
                preview = ", ".join(used_keywords[:5])
                logger.debug(f"키워드 사용 수: {len(used_keywords)} | 미리보기: {preview}")
            
            # 정의된 값 정보 로깅
            if relevant_defined_values:
                logger.debug(f"질문 관련 정의된 값 {len(relevant_defined_values)}개 감지:")
                for idx, value in enumerate(relevant_defined_values[:3], 1):
                    logger.debug(f"  - #{idx}: {value.normalized_value} (가중치: {value.weight:.1f}, 신뢰도: {value.confidence:.1f})")
            
            # 수치 질의 감지: 간단 정규식
            import re as _re
            _qnum = _re.search(r"(\d+(?:[\.,]\d+)?)(?:\s*(?:~|\-|–|to)\s*(\d+(?:[\.,]\d+)?))?\s*(%|mg\s*/\s*L|mg/L|m³/h|m3/h|㎥/h|분|시간|초)?", question)
            numeric_guard = _qnum is not None

            # 목표 기반 프롬프트 생성 (프롬프트 템플릿만 사용)
            prompt = self._create_target_based_prompt(
                context_text, question, answer_target, target_type, value_intent
            )
            
            # 답변 생성
            raw_text = self.llm.generate(prompt)
            answer_text = self._postprocess_answer(raw_text)
            
            # 답변 검증: 문서 기반 답변인지 확인
            answer_text = self._validate_document_based_answer(answer_text, context_text, question)
            
            # 환상 방지 검증 (강화된 검증)
            answer_text, was_corrected = self.hallucination_prevention.validate_answer_strictly(
                answer_text, context_text, question
            )
            
            if was_corrected:
                logger.warning("환상 방지 시스템에 의해 답변이 수정되었습니다.")

            # 가독성 포맷은 인용(출처) 추가 전 단계에서만 적용하여
            # 불릿/줄바꿈 제거가 인용 레이아웃을 깨뜨리지 않도록 한다
            answer_text = self._format_answer_for_readability(answer_text)
            # 문장 수 제한 (최대 2문장)
            answer_text = self._limit_sentences(answer_text, max_sentences=3)

            # 인용(출처) 표기를 답변에서 제거: 메타데이터에만 유지
            citations: List[Dict[str, str]] = []
            for ch in trimmed_chunks[:2]:
                page_no = getattr(ch, 'page_number', None) or (ch.metadata.get('page_number') if getattr(ch, 'metadata', None) else None)
                section = None
                if getattr(ch, 'metadata', None):
                    section = ch.metadata.get('section') or ch.metadata.get('subsection')
                filename = getattr(ch, 'filename', None) or (ch.metadata.get('title') if getattr(ch, 'metadata', None) else None)
                span_src = (getattr(ch, 'content', '') or '').strip().replace('\n', ' ')
                snippet = span_src[:90]
                citations.append({
                    'filename': str(filename) if filename else '',
                    'page': str(page_no) if page_no is not None else '',
                    'section': str(section) if section else '',
                    'snippet': snippet
                })
            
            # 수치 답변 후처리: 단위 누락 시 보정 시도(문맥에서 가장 빈도 높은 단위 부착은 위험하므로 스킵)
            
            generation_time = time.time() - start_time
            
            # 답변 길이와 포맷 간결화: 불필요한 리스트/중복 제거는 프롬프트 설계로 유도됨
            answer = Answer(
                content=answer_text,
                confidence=0.9,  # 컨텍스트 기반 답변은 신뢰도가 높음
                sources=trimmed_chunks,
                generation_time=generation_time,
                model_name=self.model_name,
                metadata={
                    "answer_type": "context", 
                    "context_used": True, 
                    "chunks_used": len(trimmed_chunks), 
                    "used_keywords": used_keywords,
                    "defined_values_detected": len(all_defined_values),
                    "relevant_defined_values": len(relevant_defined_values),
                    "high_priority_values": len([v for v in relevant_defined_values if v.weight >= 2.0]),
                    "citations": citations
                }
            )
            
            # QA 로그 기록
            try:
                log_question_answer(
                    question=question,
                    answer=answer_text,
                    chunks_used=trimmed_chunks,
                    generation_time_ms=generation_time * 1000,
                    confidence_score=0.9,
                    model_name=self.model_name,
                    pipeline_type="context",
                    session_id=answer.metadata.get("session_id") if answer.metadata else None
                )
            except Exception as e:
                logger.debug(f"QA 로그 기록 실패: {e}")
            
            return answer
            
        except Exception as e:
            logger.error(f"컨텍스트 답변 생성 실패: {e}")
            return Answer(
                content=f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
                confidence=0.0,
                sources=context_chunks,
                generation_time=time.time() - start_time,
                model_name=self.model_name,
                metadata={"error": str(e)}
            )
    
    def _enhance_context_with_defined_values(self, content: str, defined_values: List[DefinedValue]) -> str:
        """정의된 값들을 강조하여 컨텍스트를 향상시킴"""
        enhanced_content = content
        
        # 가중치가 높은 정의된 값들을 강조 표시
        for value in defined_values:
            if value.weight >= 2.0:  # 높은 가중치의 값들만 강조
                # 원본 값을 **강조** 형태로 변경
                enhanced_content = enhanced_content.replace(
                    value.value, 
                    f"**{value.normalized_value}**"
                )
        
        return enhanced_content
    
    def _extract_used_keywords(self, context_chunks: List[TextChunk]) -> List[str]:
        """컨텍스트 청크에서 사용된 키워드 추출"""
        keywords = set()
        
        for chunk in context_chunks:
            # 메타데이터에서 키워드 추출
            if chunk.metadata and "keywords" in chunk.metadata:
                chunk_keywords = chunk.metadata["keywords"]
                if isinstance(chunk_keywords, list):
                    keywords.update(chunk_keywords)
            
            # PDF ID에서 파일명 추출
            if chunk.pdf_id:
                keywords.add(f"문서: {chunk.pdf_id}")
            
            # 청크 내용에서 주요 용어 추출 (간단한 방식)
            content = chunk.content.lower()
            important_terms = [
                "교통사고", "교통량", "사고", "통계", "분석", "보고서", "데이터",
                "교통", "안전", "정책", "법규", "규정", "시스템", "관리"
            ]
            
            for term in important_terms:
                if term in content:
                    keywords.add(term)
        
        return sorted(list(keywords))
    
    def generate_answer(self, analyzed_question, context_chunks: List[TextChunk], 
                       conversation_history=None, pdf_id=None, session_id=None) -> Answer:
        """통합 답변 생성 메서드 (API 호출용) - 목표 기반 답변 생성 지원"""
        question = analyzed_question.question if hasattr(analyzed_question, 'question') else str(analyzed_question)
        
        if context_chunks:
            # 분석된 질문 정보를 전달하여 목표 기반 답변 생성
            answer = self.generate_context_answer(question, context_chunks, analyzed_question)
            # 세션 ID를 메타데이터에 추가
            if session_id and answer.metadata:
                answer.metadata["session_id"] = session_id
            return answer
        else:
            answer = self.generate_direct_answer(question)
            # 세션 ID를 메타데이터에 추가
            if session_id and answer.metadata:
                answer.metadata["session_id"] = session_id
            return answer
    
    def generate_summary(self, chunks: List[TextChunk]) -> Answer:
        start_time = time.time()
        
        try:
            # 요약용 컨텍스트 생성
            context_text = "\n\n".join([chunk.content for chunk in chunks])
            
            # 사용된 키워드 추출 (로깅 최소화)
            used_keywords = self._extract_used_keywords(chunks)
            if used_keywords:
                preview = ", ".join(used_keywords[:5])
                logger.debug(f"요약 키워드 사용 수: {len(used_keywords)} | 미리보기: {preview}")
            
            # 요약 프롬프트 생성
            prompt = self.prompt_template["summary"].format(context=context_text)
            
            # 요약 생성
            summary_text = self.llm.generate(prompt)
            
            generation_time = time.time() - start_time
            
            answer = Answer(
                content=self._format_answer_for_readability(summary_text),
                confidence=0.85,
                sources=chunks,
                generation_time=generation_time,
                model_name=self.model_name,
                metadata={"answer_type": "summary", "chunks_summarized": len(chunks), "used_keywords": used_keywords}
            )
            
            # QA 로그 기록
            try:
                log_question_answer(
                    question="[요약 요청]",
                    answer=self._format_answer_for_readability(summary_text),
                    chunks_used=chunks,
                    generation_time_ms=generation_time * 1000,
                    confidence_score=0.85,
                    model_name=self.model_name,
                    pipeline_type="summary",
                    session_id=None
                )
            except Exception as e:
                logger.debug(f"QA 로그 기록 실패: {e}")
            
            return answer
            
        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            return Answer(
                content=f"죄송합니다. 요약 생성 중 오류가 발생했습니다: {str(e)}",
                confidence=0.0,
                sources=chunks,
                generation_time=time.time() - start_time,
                model_name=self.model_name,
                metadata={"error": str(e)}
            )

    def _validate_document_based_answer(self, answer: str, context: str, question: str) -> str:
        """답변이 문서 기반인지 검증하고 필요시 수정 (강화된 검증)
        - 극단적 환상 패턴 차단
        - 최소 길이 보장
        - 문서-답변 토큰 겹침 비율이 임계치 미만이면 차단
        - 질문과 무관한 서술 길이 제한 시 보조 차단
        """
        try:
            if not answer or not context:
                return answer
            
            # 문서에 없는 정보가 포함되었는지 검사
            answer_lower = answer.lower()
            context_lower = context.lower()
            question_lower = question.lower()
            
            # 매우 명백한 환상 패턴들만 검사 (극도로 완화)
            extreme_hallucination_patterns = [
                "일반적으로 알려진", "보통의 경우", "대부분의 경우", "일반적인 원칙"
            ]
            
            # 매우 명백한 환상 패턴이 발견되면 문서 기반 답변으로 교체
            for pattern in extreme_hallucination_patterns:
                if pattern in answer_lower:
                    logger.warning(f"극도로 명백한 환상 패턴 감지: {pattern}")
                    return "제공된 문서에서 해당 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
            
            # 답변이 너무 짧거나 의미가 없는 경우만 검사
            if len(answer.strip()) < 10:
                logger.warning("답변이 너무 짧음")
                return "제공된 문서에서 해당 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
            
            # 문서-답변 토큰 겹침 비율 기반 차단 (간단 토크나이저)
            import re as _re
            token_pattern = _re.compile(r"[가-힣A-Za-z0-9]+")
            context_tokens = [t for t in token_pattern.findall(context_lower) if len(t) >= 2]
            answer_tokens = [t for t in token_pattern.findall(answer_lower) if len(t) >= 2]
            if context_tokens and answer_tokens:
                context_vocab = set(context_tokens)
                answer_vocab = set(answer_tokens)
                overlap = len(context_vocab.intersection(answer_vocab))
                denom = max(1, len(answer_vocab))
                overlap_ratio = overlap / denom
                # 임계치: 답변 토큰의 20% 미만이 문서와 겹치면 환상 가능성 높음
                if overlap_ratio < 0.12:
                    logger.warning(f"문서-답변 토큰 겹침 비율 낮음: {overlap_ratio:.3f}")
                    return "제공된 문서에서 해당 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
            
            # 문장별 스니펫 매칭 가드 (완화 기준)
            try:
                sentences = [s.strip() for s in re.split(r"[\.!?]\s+|\n+", answer) if s.strip()]
                context_text = context_lower
                kept_sentences = []
                for s in sentences:
                    s_low = s.lower()
                    if len(s_low) < 8:
                        kept_sentences.append(s)  # 너무 짧은 문장은 패스
                        continue
                    # 간단 부분 문자열/토큰 기반 매칭
                    token_hits = sum(1 for t in token_pattern.findall(s_low) if len(t) >= 3 and t in context_text)
                    unique_tokens = len(set(t for t in token_pattern.findall(s_low) if len(t) >= 3)) or 1
                    ratio = token_hits / unique_tokens
                    # 임계: 문장 토큰의 0.12 미만이면 경고만 남기고 유지
                    if ratio < 0.12:
                        logger.warning(f"문장 스니펫 매칭 약함(ratio={ratio:.2f}): {s[:50]}")
                    kept_sentences.append(s)
                # 최소 1문장 보장
                if kept_sentences:
                    answer = ". ".join(kept_sentences)
            except Exception as _e:
                logger.debug(f"문장별 스니펫 매칭 가드 스킵: {_e}")

            # 핵심 키워드 강제 검증 (질문-답변 매칭)
            try:
                # 질문에서 핵심 키워드 추출 (간단 규칙)
                q_tokens = [t for t in token_pattern.findall(question_lower) if len(t) >= 3]
                a_tokens = [t for t in token_pattern.findall(answer_lower) if len(t) >= 3]
                context_tokens = [t for t in token_pattern.findall(context_lower) if len(t) >= 3]
                
                # 질문 키워드가 컨텍스트에 있고 답변에 없으면 문제
                missing_keywords = []
                for q_token in q_tokens:
                    if q_token in context_tokens and q_token not in a_tokens:
                        missing_keywords.append(q_token)
                
                # 핵심 키워드 누락이 심하면 차단
                if len(missing_keywords) >= 2:
                    logger.warning(f"핵심 키워드 누락: {missing_keywords}")
                    return "제공된 문서에서 해당 정보를 찾을 수 없습니다. 문서에 명시된 내용만을 기반으로 답변드릴 수 있습니다."
            except Exception as _e:
                logger.debug(f"핵심 키워드 검증 스킵: {_e}")

            return answer
            
        except Exception as e:
            logger.error(f"답변 검증 중 오류: {e}")
            return answer
    
    def _postprocess_answer(self, text: str) -> str:
        """생성 텍스트를 간결하고 자연스러운 한국어 문장으로 정제한다.
        규칙:
        - 프롬프트 위반 패턴 제거 (질문 재출력, AI 비고, 따옴표 등)
        - 문서명/기관명으로 시작하는 과도한 주어 제거 시도
        - URL/경로 표기를 백틱으로 감싸고 중복 제거
        - 불필요한 접속사/군더더기 정리
        - 중국어/일본어 등 외국어 혼입 제거
        - 끝 마침표 보정
        """
        try:
            if not text:
                return text

            cleaned = text.strip()

            # 0) 프롬프트 위반 패턴 제거
            # "질문: ... 답변:" 패턴 제거
            cleaned = re.sub(r"^질문\s*:\s*[^\n]*\n\s*답변\s*:\s*", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"^사용자\s*질문\s*:\s*[^\n]*\n\s*답변\s*작성\s*지침\s*:\s*", "", cleaned, flags=re.MULTILINE)
            
            # "AI 비고:", "참고:", "주의:" 레이블 제거
            cleaned = re.sub(r"^(AI\s*비고|참고|주의)\s*:\s*", "", cleaned, flags=re.MULTILINE)
            
            # 불필요한 따옴표 제거 (문장 시작/끝의 따옴표)
            cleaned = re.sub(r'^["\']|["\']$', "", cleaned)
            
            # 중국어/일본어 혼입 제거 및 정규화
            chinese_patterns = [
                (r"이分别是", "이는"),
                (r"적度过치", "적도 과치"),
                (r"过치", "과치"),
                (r"过", "과"),
                (r"적도", "적도"),
                (r"是", "는"),
                (r"的", "의"),
                (r"了", ""),
                (r"在", "에서"),
                (r"和", "와"),
                (r"或", "또는"),
                (r"与", "와"),
                (r"为", "를 위해"),
                (r"对", "에 대해"),
                (r"从", "에서부터"),
                (r"到", "까지"),
                (r"通过", "통해"),
                (r"根据", "에 따르면"),
                (r"按照", "에 따라"),
                (r"由于", "때문에"),
                (r"因此", "따라서"),
                (r"所以", "그래서"),
                (r"但是", "하지만"),
                (r"然而", "그러나"),
                (r"而且", "또한"),
                (r"并且", "그리고"),
                (r"同时", "동시에"),
                (r"另外", "또한"),
                (r"此外", "게다가"),
                (r"总之", "요약하면"),
                (r"总之", "결론적으로"),
            ]
            
            for pattern, replacement in chinese_patterns:
                cleaned = re.sub(pattern, replacement, cleaned)
            
            # 일본어 패턴 제거
            japanese_patterns = [
                (r"です", ""),
                (r"である", ""),
                (r"だ", ""),
                (r"である", ""),
                (r"ですから", "따라서"),
                (r"しかし", "하지만"),
                (r"そして", "그리고"),
                (r"また", "또한"),
                (r"さらに", "더욱이"),
                (r"つまり", "즉"),
                (r"例えば", "예를 들어"),
                (r"特に", "특히"),
                (r"一般的に", "일반적으로"),
            ]
            
            for pattern, replacement in japanese_patterns:
                cleaned = re.sub(pattern, replacement, cleaned)
            
            # 숫자로 시작하는 불완전한 답변 제거
            cleaned = re.sub(r"^,\s*", "", cleaned)
            cleaned = re.sub(r"^\d+\.\s*$", "", cleaned)
            
            # 문장 수에 따른 숫자 매기기 제거 (1. 2. 3. 등)
            cleaned = re.sub(r"^\d+\.\s*", "", cleaned, flags=re.MULTILINE)
            
            # 강조 표시 제거 (***, **, * 등)
            cleaned = re.sub(r'\*{1,3}', '', cleaned)
            
            # 불릿 포인트 제거 (-, *, • 등)
            cleaned = re.sub(r'^[-*•]\s*', '', cleaned, flags=re.MULTILINE)
            
            # 빈 괄호 제거 ( ) 또는 (  ) 등
            cleaned = re.sub(r"\(\s*\)", "", cleaned)
            
            # 내용이 없는 괄호 패턴 제거
            cleaned = re.sub(r"\(\s*\)", "", cleaned)  # ( )
            cleaned = re.sub(r"\(\s*\)", "", cleaned)  # (  )
            cleaned = re.sub(r"\(\s*\)", "", cleaned)  # (   )
            
            # 1) 앞부분 불필요한 문서명/기관명 서두 제거 패턴(간단 휴리스틱)
            # 예: "금강 유역 남부 ... 사용자 설명서 ... 에 따르면" 류를 잘라냄
            lead_patterns = [
                r"^(?:[\w\s\-·\(\)\/]+?설명서|[\w\s\-·\(\)\/]+?매뉴얼|[\w\s\-·\(\)\/]+?문서|[\w\s\-·\(\)\/]+?가이드)[^\n\.!?]{0,80}?(?:에\s*따르면|에\s*의하면|에\s*따라|에서\s*제공하는|에\s*기재된)\s*",
                r"^[\w\s\-·\(\)\/]+?컨소시엄[^\n\.!?]{0,80}?(?:에서\s*제공하는|에\s*따르면)\s*",
                r"^문서에\s*따르면[,\s]*",
                r"^제공된\s*문서에\s*따르면[,\s]*",
                r"^금강유역\s*남부\s*스마트\s*정수장[^\n\.!?]{0,100}?문서에서[^\n\.!?]{0,50}?",
                r"^[\w\s\-·\(\)\/]+?문서에서[^\n\.!?]{0,50}?",
                r"^[\w\s\-·\(\)\/]+?에셈블\s*컨소시엄[^\n\.!?]{0,50}?",
                r"^[\w\s\-·\(\)\/]+?시스템[^\n\.!?]{0,50}?(?:에서\s*제공하는|에\s*따르면)\s*",
            ]
            for pat in lead_patterns:
                cleaned = re.sub(pat, "", cleaned)

            # 2) URL 정규화: 공백 옆 URL 감지 → 백틱 감싸기, 중복 제거
            url_regex = r"(https?://[\w\-\.:/#?%&=]+)"
            urls = re.findall(url_regex, cleaned)
            unique_urls = []
            for u in urls:
                if u not in unique_urls:
                    unique_urls.append(u)
            # 백틱으로 감싸기
            for u in unique_urls:
                cleaned = cleaned.replace(u, f"`{u}`")

            # 2-1) 자격증명/시간/지표 정규화
            # 아이디/비밀번호 패턴 (KWATER 고정 우선 교정)
            cleaned = re.sub(r"(아이디|ID)\s*[:：]?\s*kwater", "아이디: KWATER", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"(비밀번호|PW|Password)\s*[:：]?\s*kwater", "비밀번호: KWATER", cleaned, flags=re.IGNORECASE)
            # 시간 표현 보정: "10분 내" → "10분"
            cleaned = re.sub(r"(\d+)\s*분\s*내", r"\\1분", cleaned)
            # 성능 지표 표기 통일: MAE/MSE/RMSE/R²: 값
            def _metric_norm(m):
                key = m.group(1).upper()
                key = 'R²' if key in ['R2', 'R²'] else key
                return f"{key}: {m.group(2)}"
            cleaned = re.sub(r"\b(mae|mse|rmse|r2|r²)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", _metric_norm, cleaned, flags=re.IGNORECASE)

            # 3) 문장 분리(간단): 마침표/물음표/느낌표 기준
            sentences = re.split(r"(?<=[\.?\!])\s+", cleaned)
            sentences = [s.strip() for s in sentences if s.strip()]

            # 4) 군더더기 접두사 정리
            def tidy_sentence(s: str) -> str:
                s = re.sub(r"^(그리고|또는|또|그러나|하지만|그런데|즉|그러므로|따라서)\s+", "", s)
                s = re.sub(r"\s+\.$", ".", s)
                # 끝 마침표 보정
                if not re.search(r"[\.?\!]$", s):
                    s += "."
                return s

            sentences = [tidy_sentence(s) for s in sentences]

            # 5) 3문장 이내로 제한 (강제)
            if len(sentences) > 3:
                sentences = sentences[:3]

            # 6) 동일 URL이 문장에 반복되면 두 번째부터 제거
            if unique_urls and len(sentences) >= 2:
                u0 = unique_urls[0]
                for i in range(1, len(sentences)):
                    sentences[i] = sentences[i].replace(f"`{u0}`", "").strip()
                    if sentences[i] and sentences[i][-1] not in ".?!":
                        sentences[i] += "."

            final_text = " ".join(sentences)
            # 여분 공백 정리
            final_text = re.sub(r"\s{2,}", " ", final_text).strip()
            return final_text
        except Exception:
            return text
    
    def _create_target_based_prompt(self, context_text: str, question: str, 
                                  answer_target: str, target_type: str, value_intent: str) -> str:
        """목표 기반 프롬프트 생성"""
        
        # 기본 프롬프트 템플릿
        base_prompt = self.prompt_template["context"].format(
            context=context_text,
            question=question
        )
        
        # 목표 정보가 있는 경우 프롬프트 강화
        if answer_target and target_type:
            target_instructions = self._get_target_instructions(target_type, answer_target, value_intent)
            
            # 목표 기반 지침 추가
            enhanced_prompt = base_prompt + "\n\n" + target_instructions
            
            logger.debug(f"목표 기반 프롬프트 생성: {answer_target} ({target_type})")
            return enhanced_prompt
        
        return base_prompt
    
    def _get_target_instructions(self, target_type: str, answer_target: str, value_intent: str) -> str:
        """목표 타입별 답변 지침 생성"""
        
        if target_type == "quantitative_value":
            return f"""
답변 목표: {answer_target}
답변 유형: 정량적 수치/값

답변 지침:
문서에서 {answer_target}에 대한 구체적인 수치, 범위, 단위를 찾아 답변하세요. MAE, MSE, RMSE, R² 등의 성능 지표가 있으면 반드시 포함하세요. 수치와 단위를 정확히 표기하세요 (예: 0.68 mg/L, 1.54 RPM). 문서에 해당 수치 정보가 없으면 '문서에 {answer_target}에 대한 구체적인 수치 정보가 명시되어 있지 않습니다'라고 답하세요. 추측이나 일반적인 정보는 포함하지 마세요.
"""
        
        elif target_type == "qualitative_definition":
            return f"""
답변 목표: {answer_target}
답변 유형: 정의/개념/목적

답변 지침:
문서에서 {answer_target}에 대한 명확한 정의, 목적, 기능을 찾아 답변하세요. AI 모델의 목표, 시스템의 기능, 역할 등을 구체적으로 설명하세요. 문서에 명시된 내용만을 기반으로 답변하세요. 추상적이거나 일반적인 설명은 피하고 구체적인 내용을 우선하세요.
"""
        
        elif target_type == "procedural":
            return f"""
답변 목표: {answer_target}
답변 유형: 절차/과정/방법

답변 지침:
문서에서 {answer_target}에 대한 구체적인 절차, 과정, 방법을 찾아 답변하세요. 단계별 과정이나 계산 방법이 있으면 순서대로 설명하세요. 사용되는 공식, 알고리즘, 모델이 있으면 구체적으로 언급하세요. 문서에 해당 절차 정보가 없으면 '문서에 {answer_target}에 대한 구체적인 절차가 명시되어 있지 않습니다'라고 답하세요.
"""
        
        elif target_type == "comparative":
            return f"""
답변 목표: {answer_target}
답변 유형: 비교/관계/상관관계

답변 지침:
문서에서 {answer_target}에 대한 구체적인 비교 결과, 상관관계, 영향도를 찾아 답변하세요. 상관계수, 영향도 수치가 있으면 반드시 포함하세요. 어떤 변수가 가장 높은/낮은 관계를 가지는지 구체적으로 명시하세요. 문서에 해당 관계 정보가 없으면 '문서에 {answer_target}에 대한 구체적인 관계 정보가 명시되어 있지 않습니다'라고 답하세요.
"""
        
        elif target_type == "verification":
            return f"""
답변 목표: {answer_target}
답변 유형: 확인/가능성/유효성

답변 지침:
문서에서 {answer_target}에 대한 구체적인 확인 가능 여부, 접근 가능성, 유효성을 찾아 답변하세요. 가능/불가능, 유효/무효를 명확히 답변하세요. 구체적으로 어떤 정보를 확인할 수 있는지, 어떤 기능에 접근할 수 있는지 설명하세요. 문서에 해당 확인 정보가 없으면 '문서에 {answer_target}에 대한 구체적인 확인 정보가 명시되어 있지 않습니다'라고 답하세요.
"""
        
        else:
            return f"""
답변 목표: {answer_target}
답변 유형: 일반

답변 지침:
문서에서 {answer_target}에 대한 관련 정보를 찾아 답변하세요. 구체적이고 정확한 정보를 우선적으로 제공하세요. 문서에 명시된 내용만을 기반으로 답변하세요.
"""
