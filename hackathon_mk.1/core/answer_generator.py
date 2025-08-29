"""
로컬 LLM을 사용한 답변 생성 모듈

이 모듈은 로컬에서 실행되는 LLM/SLM을 사용하여 PDF 내용을 바탕으로
자연스러운 답변을 생성합니다. API 의존성을 최소화하고 개인정보 보호를 보장합니다.
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

# 로컬 LLM 라이브러리들
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama 라이브러리를 찾을 수 없습니다. pip install ollama")

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama-cpp-python 라이브러리를 찾을 수 없습니다.")

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        pipeline, GenerationConfig
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers 라이브러리를 찾을 수 없습니다.")

from .pdf_processor import TextChunk
from .question_analyzer import AnalyzedQuestion, ConversationItem
from .conversation_logger import ConversationLogger

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """지원하는 모델 타입"""
    OLLAMA = "ollama"           # Ollama 서비스
    LLAMA_CPP = "llama_cpp"     # llama.cpp 기반
    HUGGINGFACE = "huggingface" # HuggingFace transformers

@dataclass
class GenerationConfig:
    """텍스트 생성 설정"""
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1

@dataclass
class Answer:
    """생성된 답변 데이터 클래스"""
    content: str
    confidence_score: float
    used_chunks: List[str]  # 사용된 청크 ID들
    generation_time: float
    model_name: str
    metadata: Optional[Dict] = None

class LocalLLMInterface:
    """로컬 LLM 인터페이스 추상 클래스"""
    
    def __init__(self, model_name: str, config: GenerationConfig):
        self.model_name = model_name
        self.config = config
        self.is_loaded = False
    
    def load_model(self) -> bool:
        """모델 로드"""
        raise NotImplementedError
    
    def generate(self, prompt: str) -> str:
        """텍스트 생성"""
        raise NotImplementedError
    
    def unload_model(self):
        """모델 언로드"""
        pass

class OllamaInterface(LocalLLMInterface):
    """
    Ollama 인터페이스
    
    Ollama는 로컬에서 LLM을 쉽게 실행할 수 있는 도구입니다.
    설치: https://ollama.ai/
    
    장점:
    - 설치와 사용이 간단
    - 다양한 모델 지원 (Llama2, Mistral, CodeLlama 등)
    - 자동 리소스 관리
    
    사용 예시:
    ollama pull llama2:7b
    ollama pull mistral:7b
    ollama pull codellama:13b
    """
    
    def __init__(self, model_name: str = "llama2:7b", config: GenerationConfig = None):
        super().__init__(model_name, config or GenerationConfig())
        self.client = None
    
    def load_model(self) -> bool:
        """Ollama 모델 확인 및 로드"""
        if not OLLAMA_AVAILABLE:
            logger.error("Ollama 라이브러리가 설치되지 않았습니다.")
            return False
        
        try:
            self.client = ollama.Client()
            
            # 모델이 설치되어 있는지 확인
            models = self.client.list()
            logger.info(f"Ollama 응답: {models}")
            
            model_names = []
            if 'models' in models:
                for model in models['models']:
                    if 'name' in model:
                        model_names.append(model['name'])
                    elif 'model' in model:
                        model_names.append(model['model'])
            
            logger.info(f"사용 가능한 모델들: {model_names}")
            
            if self.model_name not in model_names:
                logger.warning(f"모델 {self.model_name}이 설치되지 않았습니다.")
                logger.info(f"다음 명령어로 설치하세요: ollama pull {self.model_name}")
                return False
            
            self.is_loaded = True
            logger.info(f"Ollama 모델 로드 완료: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Ollama 모델 로드 실패: {e}")
            return False
    
    def generate(self, prompt: str) -> str:
        """Ollama를 사용한 텍스트 생성"""
        if not self.is_loaded or not self.client:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': self.config.temperature,
                    'top_p': self.config.top_p,
                    'top_k': self.config.top_k,
                    'repeat_penalty': self.config.repetition_penalty,
                    'num_predict': self.config.max_length
                }
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"Ollama 텍스트 생성 실패: {e}")
            return ""

class LlamaCppInterface(LocalLLMInterface):
    """
    llama.cpp 인터페이스
    
    llama.cpp는 Meta의 LLaMA 모델을 CPU에서 효율적으로 실행하는 C++ 구현체입니다.
    
    장점:
    - CPU에서도 빠른 실행
    - 메모리 사용량 최적화
    - 양자화 모델 지원
    
    모델 다운로드:
    - HuggingFace에서 GGUF 포맷 모델 다운로드
    - 예: microsoft/DialoGPT-medium-gguf
    """
    
    def __init__(self, model_path: str, config: GenerationConfig = None, **kwargs):
        super().__init__(model_path, config or GenerationConfig())
        self.model_path = model_path
        self.model = None
        self.llm_kwargs = kwargs
    
    def load_model(self) -> bool:
        """llama.cpp 모델 로드"""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python 라이브러리가 설치되지 않았습니다.")
            return False
        
        if not os.path.exists(self.model_path):
            logger.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            return False
        
        try:
            # llama.cpp 모델 로드
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,  # 컨텍스트 길이
                n_threads=4,  # 쓰레드 수
                verbose=False,
                **self.llm_kwargs
            )
            
            self.is_loaded = True
            logger.info(f"llama.cpp 모델 로드 완료: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"llama.cpp 모델 로드 실패: {e}")
            return False
    
    def generate(self, prompt: str) -> str:
        """llama.cpp를 사용한 텍스트 생성"""
        if not self.is_loaded or not self.model:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        try:
            response = self.model(
                prompt,
                max_tokens=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repetition_penalty,
                stop=["</s>", "\n\n"]  # 적절한 중단점
            )
            
            return response['choices'][0]['text']
            
        except Exception as e:
            logger.error(f"llama.cpp 텍스트 생성 실패: {e}")
            return ""

class HuggingFaceInterface(LocalLLMInterface):
    """
    HuggingFace Transformers 인터페이스
    
    HuggingFace에서 제공하는 다양한 한국어 모델들을 사용할 수 있습니다.
    
    추천 한국어 모델:
    - beomi/KoAlpaca-Polyglot-5.8B: 한국어 대화형 모델
    - nlpai-lab/kullm-polyglot-5.8b-v2: 한국어 instruction following
    - EleutherAI/polyglot-ko-5.8b: 한국어 기본 모델
    
    주의: GPU 메모리가 충분해야 합니다 (최소 8GB 권장)
    """
    
    def __init__(self, model_name: str = "beomi/KoAlpaca-Polyglot-5.8B", 
                 config: GenerationConfig = None, use_gpu: bool = True):
        super().__init__(model_name, config or GenerationConfig())
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.tokenizer = None
        self.model = None
        self.pipeline = None
    
    def load_model(self) -> bool:
        """HuggingFace 모델 로드"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers 라이브러리가 설치되지 않았습니다.")
            return False
        
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 모델 로드
            device = "cuda" if self.use_gpu else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 파이프라인 생성
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.use_gpu else -1
            )
            
            self.is_loaded = True
            logger.info(f"HuggingFace 모델 로드 완료: {self.model_name} ({device})")
            return True
            
        except Exception as e:
            logger.error(f"HuggingFace 모델 로드 실패: {e}")
            return False
    
    def generate(self, prompt: str) -> str:
        """HuggingFace를 사용한 텍스트 생성"""
        if not self.is_loaded or not self.pipeline:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        try:
            # 생성 설정
            generation_config = {
                'max_length': len(self.tokenizer.encode(prompt)) + self.config.max_length,
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'top_k': self.config.top_k,
                'repetition_penalty': self.config.repetition_penalty,
                'do_sample': self.config.do_sample,
                'pad_token_id': self.tokenizer.eos_token_id,
                'num_return_sequences': 1
            }
            
            # 텍스트 생성
            response = self.pipeline(
                prompt,
                **generation_config
            )
            
            # 프롬프트 제거하고 생성된 텍스트만 반환
            generated_text = response[0]['generated_text']
            output = generated_text[len(prompt):].strip()
            
            return output
            
        except Exception as e:
            logger.error(f"HuggingFace 텍스트 생성 실패: {e}")
            return ""
    
    def unload_model(self):
        """모델 언로드 (메모리 해제)"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("HuggingFace 모델 언로드 완료")

class AnswerGenerator:
    """
    PDF 기반 질문 답변 생성기
    
    주요 기능:
    1. 관련 문서 청크를 바탕으로 답변 생성
    2. 이전 대화 컨텍스트 고려
    3. 다양한 로컬 LLM 지원
    4. 답변 품질 평가
    """
    
    def __init__(self, model_type: ModelType = ModelType.OLLAMA, 
                 model_name: str = "llama2:7b",
                 generation_config: GenerationConfig = None,
                 enable_conversation_cache: bool = True):
        """
        AnswerGenerator 초기화
        
        Args:
            model_type: 사용할 모델 타입
            model_name: 모델 이름 또는 경로
            generation_config: 생성 설정
            enable_conversation_cache: 대화 이력 캐시 사용 여부
        """
        self.model_type = model_type
        self.enable_conversation_cache = enable_conversation_cache
        
        # 한국어 생성에 최적화된 기본 설정
        if generation_config is None:
            self.generation_config = GenerationConfig(
                temperature=0.1,  # 더 결정적이고 일관된 답변
                top_p=0.9,       # 상위 확률 토큰 고려
                top_k=50,        # 상위 50개 토큰 고려
                max_length=256,  # 적절한 답변 길이
                repetition_penalty=1.2  # 반복 방지
            )
        else:
            self.generation_config = generation_config
        
        # 대화 이력 로거 초기화
        if enable_conversation_cache:
            self.conversation_logger = ConversationLogger()
        else:
            self.conversation_logger = None
        
        # 모델 인터페이스 초기화
        if model_type == ModelType.OLLAMA:
            self.llm = OllamaInterface(model_name, self.generation_config)
        elif model_type == ModelType.LLAMA_CPP:
            self.llm = LlamaCppInterface(model_name, self.generation_config)
        elif model_type == ModelType.HUGGINGFACE:
            self.llm = HuggingFaceInterface(model_name, self.generation_config)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        # 프롬프트 템플릿
        self.prompt_templates = self._load_prompt_templates()
        
        logger.info(f"AnswerGenerator 초기화: {model_type.value}, 캐시: {enable_conversation_cache}")
    
    def load_model(self) -> bool:
        """모델 로드"""
        return self.llm.load_model()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        프롬프트 템플릿 로드
        
        각 질문 유형과 상황에 맞는 프롬프트 템플릿을 정의합니다.
        한국어에 최적화된 프롬프트를 사용합니다.
        """
        return {
            "basic": """당신은 한국어 문서를 기반으로 질문에 답변하는 전문가입니다.

주어진 문서 내용을 정확히 읽고 질문에 답변해주세요.

문서 내용:
{context}

질문: {question}

답변 규칙:
1. 반드시 한국어로만 답변하세요
2. 문서에 명시된 정보만을 사용하세요
3. 구체적인 숫자, 이름, 날짜 등이 있다면 정확히 포함하세요
4. 문서에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요
5. 2-4문장으로 간결하고 명확하게 답변하세요
6. 반드시 완전한 문장으로 끝내세요

답변:""",

            "with_context": """당신은 한국어 문서를 기반으로 질문에 답변하는 전문가입니다.

이전 대화 내용:
{conversation_history}

현재 문서 내용:
{context}

질문: {question}

답변 규칙:
1. 반드시 한국어로만 답변하세요
2. 문서에 명시된 정보만을 사용하세요
3. 이전 대화와 자연스럽게 연결되도록 답변하세요
4. 구체적인 정보가 있다면 정확히 포함하세요
5. 2-4문장으로 간결하게 답변하세요

답변:""",

            "comparative": """당신은 한국어 문서를 기반으로 비교 분석을 수행하는 전문가입니다.

문서 내용:
{context}

질문: {question}

답변 규칙:
1. 반드시 한국어로만 답변하세요
2. 문서 내용을 바탕으로 비교 분석을 수행하세요
3. 차이점과 유사점을 명확히 구분하여 설명하세요
4. 구체적인 근거를 제시하세요
5. 3-5문장으로 자세히 답변하세요

답변:""",

            "procedural": """당신은 한국어 문서를 기반으로 절차를 설명하는 전문가입니다.

문서 내용:
{context}

질문: {question}

답변 규칙:
1. 반드시 한국어로만 답변하세요
2. 문서 내용을 바탕으로 단계별로 설명하세요
3. 각 단계를 명확히 구분하고 순서대로 정리하세요
4. 구체적인 방법이나 절차를 포함하세요
5. 3-5문장으로 자세히 답변하세요

답변:""",

            "clarification": """당신은 한국어 문서를 기반으로 추가 설명을 제공하는 전문가입니다.

문서 내용:
{context}

질문: {question}

답변 규칙:
1. 반드시 한국어로만 답변하세요
2. 문서 내용을 바탕으로 더 구체적이고 자세한 설명을 제공하세요
3. 기술적인 용어가 있다면 쉽게 풀어서 설명하세요
4. 관련된 추가 정보나 배경 지식을 포함하세요
5. 3-5문장으로 자세히 답변하세요

답변:"""
        }
    
    def generate_answer(self, 
                       analyzed_question: AnalyzedQuestion,
                       relevant_chunks: List[Tuple[TextChunk, float]],
                       conversation_history: List[ConversationItem] = None,
                       pdf_id: Optional[str] = None) -> Answer:
        """
        질문에 대한 답변 생성
        
        Args:
            analyzed_question: 분석된 질문
            relevant_chunks: 관련 문서 청크들 (유사도 포함)
            conversation_history: 이전 대화 기록
            pdf_id: PDF 문서 ID (대화 이력 검색용)
            
        Returns:
            생성된 답변
        """
        start_time = time.time()
        question = analyzed_question.original_question
        
        # 1. 대화 이력에서 빠른 답변 확인 (캐시가 활성화된 경우)
        if self.enable_conversation_cache and self.conversation_logger:
            cached_answer = self._check_conversation_cache(question, pdf_id)
            if cached_answer:
                logger.info(f"캐시된 답변 사용: {time.time() - start_time:.3f}초")
                return cached_answer
        
        # 2. 컨텍스트 구성
        context = self._build_context(relevant_chunks)
        
        # 3. 프롬프트 선택 및 구성
        prompt = self._build_prompt(
            analyzed_question, 
            context, 
            conversation_history
        )
        
        # 4. 답변 생성
        try:
            generated_text = self.llm.generate(prompt)
            
            # 5. 후처리
            processed_answer = self._post_process_answer(generated_text)
            
            # 6. 신뢰도 계산
            confidence_score = self._calculate_confidence(
                analyzed_question,
                relevant_chunks,
                processed_answer
            )
            
            generation_time = time.time() - start_time
            
            answer = Answer(
                content=processed_answer,
                confidence_score=confidence_score,
                used_chunks=[chunk.chunk_id for chunk, _ in relevant_chunks],
                generation_time=generation_time,
                model_name=self.llm.model_name,
                metadata={
                    "question_type": analyzed_question.question_type.value,
                    "num_chunks_used": len(relevant_chunks),
                    "prompt_length": len(prompt),
                    "from_cache": False
                }
            )
            
            # 7. 대화 이력에 저장 (캐시가 활성화된 경우)
            if self.enable_conversation_cache and self.conversation_logger:
                self._save_to_conversation_cache(
                    question, processed_answer, analyzed_question.question_type.value,
                    confidence_score, pdf_id, [chunk.chunk_id for chunk, _ in relevant_chunks]
                )
            
            logger.info(f"답변 생성 완료: {generation_time:.2f}초, 신뢰도: {confidence_score:.2f}")
            return answer
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            
            # 실패 시 기본 답변 반환
            return Answer(
                content="죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.",
                confidence_score=0.0,
                used_chunks=[],
                generation_time=time.time() - start_time,
                model_name=self.llm.model_name,
                metadata={"error": str(e)}
            )
    
    def _build_context(self, relevant_chunks: List[Tuple[TextChunk, float]], 
                      max_context_length: int = 2000) -> str:
        """
        관련 청크들로부터 컨텍스트 구성
        
        Args:
            relevant_chunks: 관련 청크들
            max_context_length: 최대 컨텍스트 길이
            
        Returns:
            구성된 컨텍스트 문자열
        """
        if not relevant_chunks:
            return "관련 정보를 찾을 수 없습니다."
        
        context_parts = []
        current_length = 0
        
        # 유사도 순으로 정렬된 청크들을 순서대로 추가 (최대 5개까지)
        for i, (chunk, similarity) in enumerate(relevant_chunks[:5]):
            # 청크 내용을 정리하고 페이지 정보 포함
            chunk_content = chunk.content.strip()
            
            # 너무 긴 내용은 자르기
            if len(chunk_content) > 800:
                chunk_content = chunk_content[:800] + "..."
            
            # 페이지 정보 포함
            chunk_text = f"[페이지 {chunk.page_number}] {chunk_content}"
            
            # 길이 제한 확인
            if current_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, 
                     analyzed_question: AnalyzedQuestion,
                     context: str,
                     conversation_history: List[ConversationItem] = None) -> str:
        """
        질문 유형에 맞는 프롬프트 구성
        
        Args:
            analyzed_question: 분석된 질문
            context: 문서 컨텍스트
            conversation_history: 대화 기록
            
        Returns:
            구성된 프롬프트
        """
        question_type = analyzed_question.question_type
        question = analyzed_question.original_question
        
        # 대화 기록이 있고 후속 질문인 경우
        if (conversation_history and 
            question_type.value in ["follow_up", "clarification"]):
            
            # 최근 3개 대화만 포함
            recent_history = conversation_history[-3:]
            history_text = "\n".join([
                f"Q: {item.question}\nA: {item.answer}"
                for item in recent_history
            ])
            
            template_key = "with_context"
            return self.prompt_templates[template_key].format(
                conversation_history=history_text,
                context=context,
                question=question
            )
        
        # 질문 유형별 템플릿 선택
        template_mapping = {
            "comparative": "comparative",
            "procedural": "procedural", 
            "clarification": "clarification"
        }
        
        template_key = template_mapping.get(question_type.value, "basic")
        
        return self.prompt_templates[template_key].format(
            context=context,
            question=question
        )
    
    def _post_process_answer(self, generated_text: str) -> str:
        """
        생성된 답변 후처리 (한국어 강제)
        
        Args:
            generated_text: 생성된 원본 텍스트
            
        Returns:
            후처리된 답변
        """
        import re
        
        # 불필요한 공백 제거
        answer = generated_text.strip()
        
        # 영어 문장이나 영어 단어가 주를 이루는 부분 제거
        lines = answer.split('\n')
        korean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 한글이 50% 이상인 라인만 유지
            korean_chars = len(re.findall(r'[가-힣]', line))
            total_chars = len(re.findall(r'[a-zA-Z가-힣]', line))
            
            if total_chars == 0:
                korean_lines.append(line)  # 숫자나 기호만 있는 경우
            elif korean_chars / total_chars >= 0.3:  # 한글이 30% 이상
                korean_lines.append(line)
        
        answer = '\n'.join(korean_lines)
        
        # 영어로 시작하는 문장 제거
        sentences = answer.split('.')
        korean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 영어로 시작하는 문장 제거
            if re.match(r'^[a-zA-Z]', sentence):
                continue
                
            # 반복 문장 제거
            if sentence not in korean_sentences:
                korean_sentences.append(sentence)
        
        answer = '. '.join(korean_sentences)
        
        # 한국어 문장 종결어미로 끝나지 않으면 정리
        korean_endings = ['다.', '습니다.', '요.', '음.', '네.', '죠.', '다!', '습니다!', '요!', '다?', '습니다?', '요?']
        proper_endings = ['.', '!', '?']
        
        # 적절한 종결어미로 끝나는지 확인
        ends_properly = any(answer.endswith(ending) for ending in korean_endings + proper_endings)
        
        if answer and not ends_properly:
            # 마지막 완전한 한국어 종결을 찾기
            last_positions = []
            
            # 다양한 종결어미 위치 찾기
            for ending in ['다.', '습니다.', '요.', '다!', '습니다!', '요!', '다?', '습니다?', '요?']:
                pos = answer.rfind(ending)
                if pos >= 0:
                    last_positions.append(pos + len(ending))
            
            # 일반적인 문장 부호도 확인
            for punct in ['.', '!', '?']:
                pos = answer.rfind(punct)
                if pos >= 0:
                    # 문장 부호 앞에 한글이 있는지 확인
                    if pos > 0 and answer[pos-1] in '다요음네':
                        last_positions.append(pos + 1)
            
            if last_positions:
                # 가장 뒤에 있는 완전한 종결점 사용
                last_end = max(last_positions)
                if last_end > len(answer) * 0.3:  # 전체 길이의 30% 이상인 경우만
                    answer = answer[:last_end]
            else:
                # 완전한 종결을 찾지 못한 경우, 자연스럽게 마무리
                answer = answer.rstrip('.,!?') + '.'
        
        # 최종 정리: 자연스러운 종결 보장
        if answer.strip():
            answer = answer.strip()
            
            # 마지막 문자가 한글인 경우 적절한 종결어미 추가
            if answer and answer[-1] in '가나다라마바사아자차카타파하':
                answer += '입니다.'
            elif answer and not answer.endswith(('.', '!', '?')):
                # 어떤 종결도 없는 경우 마침표 추가
                answer += '.'
                
            # 불완전한 문장 제거 (한 글자짜리 단어나 의미없는 끝부분)
            if len(answer) > 1 and answer.endswith(('다.', '요.', '습니다.')):
                # 이미 적절히 끝난 경우 그대로 유지
                pass
            elif not answer.endswith(('.', '!', '?')):
                # 여전히 완전하지 않은 경우 강제로 마무리
                answer = answer.rstrip('.,!?가나다라마바사아자차카타파하') + '입니다.'
        else:
            answer = "죄송합니다. 해당 질문에 대한 적절한 답변을 생성할 수 없습니다."
        
        return answer
    
    def _calculate_confidence(self, 
                            analyzed_question: AnalyzedQuestion,
                            relevant_chunks: List[Tuple[TextChunk, float]],
                            generated_answer: str) -> float:
        """
        답변의 신뢰도 계산 (개선된 버전)
        
        Args:
            analyzed_question: 분석된 질문
            relevant_chunks: 사용된 청크들
            generated_answer: 생성된 답변
            
        Returns:
            신뢰도 점수 (0.0 ~ 1.0)
        """
        confidence = 0.0
        
        # 1. 관련 청크의 유사도 기반 신뢰도 (30%)
        if relevant_chunks:
            avg_similarity = sum(sim for _, sim in relevant_chunks) / len(relevant_chunks)
            confidence += avg_similarity * 0.3
        
        # 2. 키워드 매칭 점수 (25%)
        keyword_score = self._calculate_keyword_matching_score(
            analyzed_question, generated_answer
        )
        confidence += keyword_score * 0.25
        
        # 3. 답변 품질 점수 (20%)
        quality_score = self._calculate_answer_quality_score(generated_answer)
        confidence += quality_score * 0.2
        
        # 4. 컨텍스트 활용도 점수 (15%)
        context_score = self._calculate_context_utilization_score(
            relevant_chunks, generated_answer
        )
        confidence += context_score * 0.15
        
        # 5. 답변 완성도 점수 (10%)
        completeness_score = self._calculate_completeness_score(generated_answer)
        confidence += completeness_score * 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_keyword_matching_score(self, 
                                        analyzed_question: AnalyzedQuestion,
                                        generated_answer: str) -> float:
        """
        질문-답변 키워드 매칭 점수 계산
        
        Args:
            analyzed_question: 분석된 질문
            generated_answer: 생성된 답변
            
        Returns:
            키워드 매칭 점수 (0.0 ~ 1.0)
        """
        question_keywords = set(analyzed_question.keywords)
        answer_words = set(generated_answer.lower().split())
        
        if not question_keywords:
            return 0.5  # 키워드가 없는 경우 중간 점수
        
        # 1. 정확한 키워드 매칭
        exact_matches = len(question_keywords & answer_words)
        exact_score = exact_matches / len(question_keywords)
        
        # 2. 부분 키워드 매칭 (한국어 특성 고려)
        partial_score = 0.0
        for q_keyword in question_keywords:
            for answer_word in answer_words:
                # 부분 문자열 매칭
                if len(q_keyword) >= 3 and len(answer_word) >= 3:
                    if q_keyword in answer_word or answer_word in q_keyword:
                        partial_score += 0.5
                        break
        
        partial_score = min(partial_score / len(question_keywords), 1.0)
        
        # 3. 동의어 매칭
        synonym_score = self._calculate_synonym_matching_score(
            question_keywords, answer_words
        )
        
        # 가중 평균
        total_score = (exact_score * 0.5 + partial_score * 0.3 + synonym_score * 0.2)
        
        return min(total_score, 1.0)
    
    def _calculate_synonym_matching_score(self, 
                                        question_keywords: set,
                                        answer_words: set) -> float:
        """
        동의어 매칭 점수 계산
        
        Args:
            question_keywords: 질문 키워드들
            answer_words: 답변 단어들
            
        Returns:
            동의어 매칭 점수 (0.0 ~ 1.0)
        """
        # 한국어 동의어 사전
        synonyms = {
            '방법': ['방식', '기법', '기술', '수단'],
            '과정': ['절차', '단계', '순서', '진행'],
            '결과': ['성과', '효과', '결과물', '산출물'],
            '문제': ['이슈', '과제', '해결사항'],
            '개선': ['향상', '발전', '고도화', '최적화'],
            '분석': ['검토', '조사', '연구', '평가'],
            '시스템': ['플랫폼', '솔루션', '도구'],
            '데이터': ['정보', '자료', '내용'],
            '관리': ['운영', '유지보수'],
            '보안': ['안전', '보호', '안전성'],
            '성능': ['효율', '속도', '품질'],
            '비용': ['금액', '가격', '지출'],
            '시간': ['기간', '소요시간', '기한'],
            '사용자': ['고객', '이용자'],
            '기능': ['특성', '역할', '작용'],
            '구조': ['체계', '구성', '설계'],
            '환경': ['조건', '상황', '배경'],
            '요구사항': ['필요사항', '요구', '필요'],
            '정책': ['규정', '지침', '방침'],
            '절차': ['순서', '과정', '단계']
        }
        
        synonym_matches = 0
        total_keywords = len(question_keywords)
        
        for keyword in question_keywords:
            if keyword in synonyms:
                keyword_synonyms = synonyms[keyword]
                for synonym in keyword_synonyms:
                    if synonym in answer_words:
                        synonym_matches += 1
                        break
        
        return synonym_matches / total_keywords if total_keywords > 0 else 0.0
    
    def _calculate_answer_quality_score(self, generated_answer: str) -> float:
        """
        답변 품질 점수 계산
        
        Args:
            generated_answer: 생성된 답변
            
        Returns:
            품질 점수 (0.0 ~ 1.0)
        """
        score = 0.0
        
        # 1. 답변 길이 적절성
        answer_length = len(generated_answer)
        if 50 <= answer_length <= 500:
            score += 0.3
        elif 30 <= answer_length <= 800:
            score += 0.2
        else:
            score += 0.1
        
        # 2. 문장 완성도
        if generated_answer.endswith(('.', '!', '?', '다', '습니다', '요')):
            score += 0.2
        
        # 3. 한국어 비율
        korean_chars = len(re.findall(r'[가-힣]', generated_answer))
        total_chars = len(re.findall(r'[a-zA-Z가-힣]', generated_answer))
        
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            if korean_ratio >= 0.7:
                score += 0.3
            elif korean_ratio >= 0.5:
                score += 0.2
            else:
                score += 0.1
        
        # 4. 부정적 표현 제거
        negative_phrases = [
            "찾을 수 없습니다", "모르겠습니다", "알 수 없습니다",
            "정보가 없습니다", "답변할 수 없습니다"
        ]
        
        has_negative = any(phrase in generated_answer for phrase in negative_phrases)
        if not has_negative:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_context_utilization_score(self,
                                           relevant_chunks: List[Tuple[TextChunk, float]],
                                           generated_answer: str) -> float:
        """
        컨텍스트 활용도 점수 계산
        
        Args:
            relevant_chunks: 관련 청크들
            generated_answer: 생성된 답변
            
        Returns:
            컨텍스트 활용도 점수 (0.0 ~ 1.0)
        """
        if not relevant_chunks:
            return 0.0
        
        # 청크 내용에서 중요한 정보 추출
        chunk_keywords = set()
        for chunk, _ in relevant_chunks:
            # 청크에서 중요한 단어들 추출
            important_words = re.findall(r'\b\w{2,}\b', chunk.content.lower())
            chunk_keywords.update(important_words[:10])  # 상위 10개만
        
        # 답변에서 청크 키워드 사용 여부 확인
        answer_words = set(generated_answer.lower().split())
        
        if chunk_keywords:
            utilization_ratio = len(chunk_keywords & answer_words) / len(chunk_keywords)
            return min(utilization_ratio, 1.0)
        
        return 0.5  # 기본 점수
    
    def _calculate_completeness_score(self, generated_answer: str) -> float:
        """
        답변 완성도 점수 계산
        
        Args:
            generated_answer: 생성된 답변
            
        Returns:
            완성도 점수 (0.0 ~ 1.0)
        """
        score = 0.0
        
        # 1. 문장 완성도
        sentences = generated_answer.split('.')
        complete_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 의미있는 문장 길이
                complete_sentences += 1
        
        if complete_sentences > 0:
            score += min(complete_sentences * 0.2, 0.6)
        
        # 2. 구체적 정보 포함 여부
        specific_patterns = [
            r'\d+',  # 숫자
            r'\d{4}년',  # 연도
            r'\d+월',  # 월
            r'\d+일',  # 일
            r'[가-힣]{2,}시스템',  # 시스템명
            r'[가-힣]{2,}프로그램',  # 프로그램명
            r'[가-힣]{2,}기능',  # 기능명
        ]
        
        specific_info_count = 0
        for pattern in specific_patterns:
            if re.search(pattern, generated_answer):
                specific_info_count += 1
        
        score += min(specific_info_count * 0.1, 0.4)
        
        return min(score, 1.0)
    
    def batch_generate(self, 
                      questions_and_chunks: List[Tuple[AnalyzedQuestion, List[Tuple[TextChunk, float]]]],
                      conversation_histories: List[List[ConversationItem]] = None) -> List[Answer]:
        """
        여러 질문에 대한 배치 답변 생성
        
        Args:
            questions_and_chunks: (분석된 질문, 관련 청크) 튜플 리스트
            conversation_histories: 각 질문에 대한 대화 기록들
            
        Returns:
            생성된 답변 리스트
        """
        answers = []
        
        if conversation_histories is None:
            conversation_histories = [None] * len(questions_and_chunks)
        
        for i, (analyzed_q, chunks) in enumerate(questions_and_chunks):
            conv_history = conversation_histories[i] if i < len(conversation_histories) else None
            
            answer = self.generate_answer(analyzed_q, chunks, conv_history)
            answers.append(answer)
        
        return answers
    
    def _check_conversation_cache(self, question: str, pdf_id: Optional[str] = None) -> Optional[Answer]:
        """
        대화 이력에서 빠른 답변 확인
        
        Args:
            question: 사용자 질문
            pdf_id: PDF 문서 ID
            
        Returns:
            캐시된 답변 또는 None
        """
        if not self.conversation_logger:
            return None
        
        try:
            # 1. 정확히 일치하는 질문 검색
            exact_match = self.conversation_logger.find_exact_match(question)
            if exact_match:
                logger.info(f"정확한 일치 발견: {exact_match.id}")
                return Answer(
                    content=exact_match.answer,
                    confidence_score=exact_match.confidence_score,
                    used_chunks=exact_match.used_chunks or [],
                    generation_time=0.001,  # 캐시 사용으로 매우 빠름
                    model_name="conversation_cache",
                    metadata={
                        "from_cache": True,
                        "cache_id": exact_match.id,
                        "question_type": exact_match.question_type,
                        "timestamp": exact_match.timestamp
                    }
                )
            
            # 2. 유사한 질문 검색 (신뢰도가 높은 경우만)
            similar_questions = self.conversation_logger.find_similar_questions(
                question, threshold=0.8, limit=3
            )
            
            for log_entry, similarity in similar_questions:
                # 신뢰도가 높고 유사도가 높은 경우만 사용
                if log_entry.confidence_score >= 0.7 and similarity >= 0.85:
                    logger.info(f"유사 질문 사용: {log_entry.id} (유사도: {similarity:.2f})")
                    return Answer(
                        content=log_entry.answer,
                        confidence_score=log_entry.confidence_score * similarity,  # 유사도로 가중치 적용
                        used_chunks=log_entry.used_chunks or [],
                        generation_time=0.002,  # 캐시 사용으로 매우 빠름
                        model_name="conversation_cache",
                        metadata={
                            "from_cache": True,
                            "cache_id": log_entry.id,
                            "question_type": log_entry.question_type,
                            "similarity": similarity,
                            "timestamp": log_entry.timestamp
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"대화 이력 캐시 확인 중 오류: {e}")
            return None
    
    def _save_to_conversation_cache(self, 
                                  question: str, 
                                  answer: str, 
                                  question_type: str,
                                  confidence_score: float,
                                  pdf_id: Optional[str] = None,
                                  used_chunks: Optional[List[str]] = None):
        """
        대화 이력에 답변 저장
        
        Args:
            question: 사용자 질문
            answer: 생성된 답변
            question_type: 질문 유형
            confidence_score: 답변 신뢰도
            pdf_id: PDF 문서 ID
            used_chunks: 사용된 청크 ID들
        """
        if not self.conversation_logger:
            return
        
        try:
            # 신뢰도가 낮은 답변은 저장하지 않음
            if confidence_score < 0.5:
                logger.debug(f"신뢰도가 낮아 캐시에 저장하지 않음: {confidence_score:.2f}")
                return
            
            # 답변이 너무 짧거나 긴 경우 저장하지 않음
            if len(answer) < 10 or len(answer) > 2000:
                logger.debug(f"답변 길이가 부적절하여 캐시에 저장하지 않음: {len(answer)}자")
                return
            
            # 에러 메시지나 기본 답변은 저장하지 않음
            error_patterns = [
                "죄송합니다", "오류가 발생했습니다", "문서에서 해당 정보를 찾을 수 없습니다",
                "관련 정보를 찾을 수 없습니다", "답변을 생성할 수 없습니다"
            ]
            
            if any(pattern in answer for pattern in error_patterns):
                logger.debug("에러 메시지로 판단되어 캐시에 저장하지 않음")
                return
            
            # 대화 이력에 저장
            log_id = self.conversation_logger.add_conversation(
                question=question,
                answer=answer,
                question_type=question_type,
                confidence_score=confidence_score,
                pdf_id=pdf_id,
                used_chunks=used_chunks,
                metadata={
                    "model_name": self.llm.model_name,
                    "generation_time": time.time()
                }
            )
            
            logger.debug(f"대화 이력에 저장: {log_id}")
            
        except Exception as e:
            logger.warning(f"대화 이력 저장 중 오류: {e}")
    
    def get_conversation_statistics(self) -> Dict[str, Any]:
        """
        대화 이력 통계 조회
        
        Returns:
            통계 정보
        """
        if not self.conversation_logger:
            return {"error": "대화 이력 로거가 비활성화되어 있습니다."}
        
        try:
            return self.conversation_logger.get_statistics()
        except Exception as e:
            logger.error(f"대화 이력 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    def clear_conversation_cache(self, pdf_id: Optional[str] = None) -> int:
        """
        대화 이력 캐시 삭제
        
        Args:
            pdf_id: 특정 PDF의 대화 이력만 삭제 (None이면 전체)
            
        Returns:
            삭제된 로그 수
        """
        if not self.conversation_logger:
            return 0
        
        try:
            return self.conversation_logger.clear_history(pdf_id)
        except Exception as e:
            logger.error(f"대화 이력 삭제 실패: {e}")
            return 0
    
    def unload_model(self):
        """모델 언로드"""
        self.llm.unload_model()

# 파인튜닝 관련 함수들
def prepare_finetuning_data(qa_pairs: List[Dict], 
                          pdf_chunks: List[TextChunk]) -> Dict:
    """
    로컬 LLM 파인튜닝을 위한 데이터 준비
    
    Args:
        qa_pairs: 질문-답변 쌍들
        pdf_chunks: PDF 청크들
        
    Returns:
        훈련 데이터
        
    파인튜닝 이유:
    1. 도메인 특화 지식 학습: 특정 분야의 전문 용어와 개념 이해
    2. 답변 스타일 조정: 원하는 답변 형식과 톤에 맞춤
    3. 한국어 성능 향상: 한국어 문맥 이해와 자연스러운 표현
    4. 컨텍스트 활용 개선: PDF 내용을 효과적으로 활용한 답변 생성
    
    필요한 데이터:
    - 최소 100-500개의 고품질 QA 쌍
    - 도메인 특화 용어집
    - 다양한 질문 유형 커버
    """
    
    chunk_dict = {chunk.chunk_id: chunk for chunk in pdf_chunks}
    training_examples = []
    
    for qa_pair in qa_pairs:
        question = qa_pair['question']
        answer = qa_pair['answer']
        relevant_chunk_ids = qa_pair.get('relevant_chunks', [])
        
        # 관련 청크들로 컨텍스트 구성
        context_parts = []
        for chunk_id in relevant_chunk_ids:
            if chunk_id in chunk_dict:
                context_parts.append(chunk_dict[chunk_id].content)
        
        context = "\n\n".join(context_parts)
        
        # 훈련 예시 생성 (instruction-following 형태)
        training_example = {
            "instruction": "주어진 문서 내용을 바탕으로 질문에 답변해주세요.",
            "input": f"문서 내용:\n{context}\n\n질문: {question}",
            "output": answer
        }
        
        training_examples.append(training_example)
    
    return {
        "training_data": training_examples,
        "statistics": {
            "total_examples": len(training_examples),
            "avg_context_length": sum(len(ex["input"]) for ex in training_examples) / len(training_examples),
            "avg_answer_length": sum(len(ex["output"]) for ex in training_examples) / len(training_examples)
        }
    }

if __name__ == "__main__":
    # 테스트 코드
    print("AnswerGenerator 모듈이 정상적으로 로드되었습니다.")
    
    # 사용 예시
    config = GenerationConfig(
        max_length=512,
        temperature=0.7,
        top_p=0.9
    )
    
    # Ollama 사용 예시
    if OLLAMA_AVAILABLE:
        generator = AnswerGenerator(
            model_type=ModelType.OLLAMA,
            model_name="llama2:7b",
            generation_config=config
        )
        print("Ollama 인터페이스 초기화 완료")
    
    # HuggingFace 사용 예시 (GPU가 있는 경우)
    if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
        hf_generator = AnswerGenerator(
            model_type=ModelType.HUGGINGFACE,
            model_name="beomi/KoAlpaca-Polyglot-5.8B",
            generation_config=config
        )
        print("HuggingFace 인터페이스 초기화 완료")
