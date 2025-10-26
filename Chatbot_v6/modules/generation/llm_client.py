"""
Ollama Client

Ollama API 통신 클라이언트.
텍스트 생성 및 연결 상태 확인.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional, Dict, Any

from config.model_config import LLMModelConfig
from modules.core.exceptions import LLMConnectionError, LLMTimeoutError, LLMResponseEmptyError
from modules.core.logger import get_logger
from .ollama_manager import ollama_manager

logger = get_logger(__name__)


class OllamaClient:
    """Ollama API 통신 클라이언트"""
    
    def __init__(self, config: Optional[LLMModelConfig] = None):
        """
        Args:
            config: LLM 모델 설정
        """
        self.config = config or LLMModelConfig()
        self.base_url = self.config.get_ollama_url()
        
        logger.info("OllamaClient initialized",
                   base_url=self.base_url,
                   model=self.config.model_name)
        
        # 모델 자동 설치 확인
        self._ensure_model_available()
    
    def generate(
        self,
        prompt: str,
        timeout: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 프롬프트
            timeout: 타임아웃 (초)
            **kwargs: 추가 생성 옵션
            
        Returns:
            생성된 텍스트
            
        Raises:
            LLMConnectionError: 연결 실패
            LLMTimeoutError: 타임아웃
            LLMResponseEmptyError: 빈 응답
        """
        url = f"{self.base_url}/api/generate"
        timeout = timeout or self.config.timeout
        
        # keep_alive 설정
        keep_alive = (
            f"{self.config.keep_alive_minutes}m"
            if self.config.keep_alive_minutes > 0
            else "0"
        )
        
        # 요청 데이터
        data = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                **self.config.get_generation_options(),
                **kwargs,  # 추가 옵션 오버라이드
            }
        }
        
        logger.debug("Generating text",
                    prompt_length=len(prompt),
                    model=self.config.model_name,
                    timeout=timeout)
        
        # HTTP 요청
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="ignore"))
                response_text = body.get("response", "")
                
                if not response_text:
                    raise LLMResponseEmptyError(prompt[:100])
                
                logger.debug("Text generated",
                            response_length=len(response_text))
                
                return response_text
        
        except urllib.error.URLError as e:
            raise LLMConnectionError(
                url=url,
                cause=e,
            ) from e
        
        except TimeoutError as e:
            raise LLMTimeoutError(
                timeout=timeout,
                prompt_length=len(prompt),
            ) from e
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise LLMConnectionError(
                url=url,
                cause=e,
            ) from e
    
    def is_available(self) -> bool:
        """Ollama 서버가 사용 가능한지 확인"""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url)
            
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        
        except Exception:
            return False
    
    def _ensure_model_available(self) -> None:
        """모델이 사용 가능한지 확인하고 필요시 자동 설치"""
        try:
            # OllamaManager를 사용하여 모델 확인 및 자동 설치
            if not ollama_manager.ensure_model_available(self.config.model_name):
                logger.warning(f"Model {self.config.model_name} is not available")
                logger.info("You may need to install it manually: ollama pull " + self.config.model_name)
        except Exception as e:
            logger.error(f"Failed to ensure model availability: {e}")
            logger.info("Continuing without automatic model installation...")

