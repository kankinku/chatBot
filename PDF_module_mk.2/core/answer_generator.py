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
                 generation_config: GenerationConfig = None):
        """
        AnswerGenerator 초기화
        
        Args:
            model_type: 사용할 모델 타입
            model_name: 모델 이름 또는 경로
            generation_config: 생성 설정
        """
        self.model_type = model_type
        # 한국어 생성에 최적화된 기본 설정
        if generation_config is None:
            self.generation_config = GenerationConfig(
                temperature=0.3,  # 더 결정적인 생성으로 일관된 한국어
                top_p=0.8,       # 상위 확률 토큰만 고려
                top_k=30         # 상위 30개 토큰만 고려
            )
        else:
            self.generation_config = generation_config
        
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
        
        logger.info(f"AnswerGenerator 초기화: {model_type.value}")
    
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
            "basic": """다음은 문서의 내용 중 일부입니다:

{context}

사용자 질문: {question}

지시사항:
1. 반드시 한국어로만 답변하세요.
2. 2-4문장으로 간결하고 핵심적인 답변을 제공하세요.
3. 문서에 없는 내용은 추측하지 말고 "문서에서 해당 정보를 찾을 수 없습니다"라고 말하세요.
4. 반드시 완전한 문장으로 끝내세요 (예: ~습니다. ~입니다. ~됩니다.)
5. 영어나 다른 언어를 사용하지 마세요.

완전한 한국어 답변:""",

            "with_context": """다음은 이전 대화 내용입니다:
{conversation_history}

현재 문서의 관련 내용:
{context}

사용자 질문: {question}

지시사항:
1. 반드시 한국어로만 답변하세요.
2. 2-4문장으로 간결하게 답변하세요.
3. 반드시 완전한 문장으로 끝내세요 (예: ~습니다. ~입니다. ~됩니다.)
4. 이전 대화와 자연스럽게 연결되도록 답변하세요.
5. 영어나 다른 언어를 절대 사용하지 마세요.

완전한 한국어 답변:""",

            "comparative": """다음은 문서의 관련 내용들입니다:

{context}

사용자 질문: {question}

지시사항:
1. 반드시 한국어로만 답변하세요.
2. 위 내용을 바탕으로 비교 분석하여 답변하세요.
3. 차이점과 유사점을 명확히 구분하여 설명하고, 구체적인 근거를 제시하세요.
4. 정중하고 자연스러운 한국어로 작성하세요.
5. 영어나 다른 언어를 절대 사용하지 마세요.

한국어 답변:""",

            "procedural": """다음은 문서의 관련 내용입니다:

{context}

사용자 질문: {question}

지시사항:
1. 반드시 한국어로만 답변하세요.
2. 위 내용을 바탕으로 단계별로 설명하세요.
3. 각 단계를 명확히 구분하고, 순서대로 정리하여 답변하세요.
4. 정중하고 자연스러운 한국어로 작성하세요.
5. 영어나 다른 언어를 절대 사용하지 마세요.

한국어 답변:""",

            "clarification": """다음은 문서의 관련 내용입니다:

{context}

이전 답변이나 설명에 대한 추가 질문: {question}

지시사항:
1. 반드시 한국어로만 답변하세요.
2. 위 내용을 바탕으로 더 구체적이고 자세한 설명을 제공하세요.
3. 기술적인 용어가 있다면 쉽게 풀어서 설명하세요.
4. 정중하고 자연스러운 한국어로 작성하세요.
5. 영어나 다른 언어를 절대 사용하지 마세요.

한국어 답변:"""
        }
    
    def generate_answer(self, 
                       analyzed_question: AnalyzedQuestion,
                       relevant_chunks: List[Tuple[TextChunk, float]],
                       conversation_history: List[ConversationItem] = None) -> Answer:
        """
        질문에 대한 답변 생성
        
        Args:
            analyzed_question: 분석된 질문
            relevant_chunks: 관련 문서 청크들 (유사도 포함)
            conversation_history: 이전 대화 기록
            
        Returns:
            생성된 답변
        """
        start_time = time.time()
        
        # 1. 컨텍스트 구성
        context = self._build_context(relevant_chunks)
        
        # 2. 프롬프트 선택 및 구성
        prompt = self._build_prompt(
            analyzed_question, 
            context, 
            conversation_history
        )
        
        # 3. 답변 생성
        try:
            generated_text = self.llm.generate(prompt)
            
            # 4. 후처리
            processed_answer = self._post_process_answer(generated_text)
            
            # 5. 신뢰도 계산
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
                    "prompt_length": len(prompt)
                }
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
        
        # 유사도 순으로 정렬된 청크들을 순서대로 추가
        for i, (chunk, similarity) in enumerate(relevant_chunks):
            chunk_text = f"[문서 {i+1}] {chunk.content}"
            
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
        답변의 신뢰도 계산
        
        Args:
            analyzed_question: 분석된 질문
            relevant_chunks: 사용된 청크들
            generated_answer: 생성된 답변
            
        Returns:
            신뢰도 점수 (0.0 ~ 1.0)
        """
        confidence = 0.0
        
        # 1. 관련 청크의 유사도 기반 신뢰도
        if relevant_chunks:
            avg_similarity = sum(sim for _, sim in relevant_chunks) / len(relevant_chunks)
            confidence += avg_similarity * 0.4
        
        # 2. 답변 길이 기반 신뢰도 (너무 짧거나 길면 감점)
        answer_length = len(generated_answer)
        if 50 <= answer_length <= 500:
            length_score = 1.0
        elif answer_length < 50:
            length_score = answer_length / 50.0
        else:
            length_score = max(0.5, 1.0 - (answer_length - 500) / 1000.0)
        
        confidence += length_score * 0.2
        
        # 3. 질문-답변 키워드 매칭
        question_keywords = set(analyzed_question.keywords)
        answer_words = set(generated_answer.lower().split())
        
        if question_keywords:
            keyword_overlap = len(question_keywords & answer_words) / len(question_keywords)
            confidence += keyword_overlap * 0.2
        
        # 4. 답변 완성도 (문장 부호, 문법 등)
        completeness_score = 0.0
        if generated_answer.endswith(('.', '!', '?', '다', '습니다', '요')):
            completeness_score += 0.5
        if not any(phrase in generated_answer for phrase in ["찾을 수 없습니다", "모르겠습니다"]):
            completeness_score += 0.5
        
        confidence += completeness_score * 0.2
        
        return min(confidence, 1.0)
    
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
