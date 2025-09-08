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
from core.document.pdf_processor import TextChunk
from core.query.question_analyzer import AnalyzedQuestion
from core.cache.fast_cache import get_question_cache
from core.utils.memory_optimizer import model_memory_manager, memory_profiler

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
    """경량화된 생성 설정"""
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
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
        
        # 기본 프롬프트 템플릿 (도메인 SSOT 주입)
        domain = str(get_config('DOMAIN_TEMPLATE', 'general'))
        variant = str(get_config('PROMPT_VARIANT', 'A'))
        version = str(get_config('PROMPT_VERSION', '1'))
        try:
            from core.config.prompt_catalog import get_templates
            self.prompt_template = get_templates(domain, variant=variant, version=version)
        except Exception:
            # 폴백: general A v1
            from core.config.prompt_catalog import get_templates
            self.prompt_template = get_templates('general', variant='A', version='1')
        
        logger.info(f"경량화된 답변 생성기 초기화 완료: {model_name}")
    
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
            answer_text = self.llm.generate(context_prompt)
            
            generation_time = time.time() - start_time
            
            return Answer(
                content=answer_text,
                confidence=0.8,  # 직접 답변은 신뢰도가 낮음
                sources=[],
                generation_time=generation_time,
                model_name=self.model_name,
                metadata={"answer_type": "direct", "context_used": False}
            )
            
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
    
    def generate_context_answer(self, question: str, context_chunks: List[TextChunk]) -> Answer:
        """컨텍스트 기반 답변 생성"""
        start_time = time.time()
        
        try:
            # 컨텍스트 텍스트 생성 (가독성 위해 상위 3개로 제한)
            trimmed_chunks = context_chunks[:3]
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
            context_text = "\n\n".join([chunk.content for chunk in trimmed_chunks])
            
            # 사용된 키워드 추출 (로깅 최소화)
            used_keywords = self._extract_used_keywords(trimmed_chunks)
            if used_keywords:
                preview = ", ".join(used_keywords[:5])
                logger.debug(f"키워드 사용 수: {len(used_keywords)} | 미리보기: {preview}")
            
            # 프롬프트 생성
            prompt = self.prompt_template["context"].format(
                context=context_text,
                question=question
            )
            
            # 답변 생성
            answer_text = self.llm.generate(prompt)
            
            generation_time = time.time() - start_time
            
            # 답변 길이와 포맷 간결화: 불필요한 리스트/중복 제거는 프롬프트 설계로 유도됨
            return Answer(
                content=answer_text,
                confidence=0.9,  # 컨텍스트 기반 답변은 신뢰도가 높음
                sources=trimmed_chunks,
                generation_time=generation_time,
                model_name=self.model_name,
                metadata={"answer_type": "context", "context_used": True, "chunks_used": len(trimmed_chunks), "used_keywords": used_keywords}
            )
            
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
            
            return Answer(
                content=summary_text,
                confidence=0.85,
                sources=chunks,
                generation_time=generation_time,
                model_name=self.model_name,
                metadata={"answer_type": "summary", "chunks_summarized": len(chunks), "used_keywords": used_keywords}
            )
            
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
