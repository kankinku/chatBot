"""
Ollama Manager

Ollama 모델 자동 설치 및 관리 기능을 제공합니다.
"""

from __future__ import annotations

import subprocess
import json
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from modules.core.logger import get_logger
from modules.core.exceptions import LLMConnectionError

logger = get_logger(__name__)


class OllamaManager:
    """Ollama 모델 관리자"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Args:
            base_url: Ollama 서버 URL
        """
        self.base_url = base_url
        self._available_models: Optional[List[str]] = None
        
        logger.info("OllamaManager initialized", base_url=base_url)
    
    def check_ollama_running(self) -> bool:
        """Ollama 서버가 실행 중인지 확인"""
        try:
            import urllib.request
            import urllib.error
            
            # Ollama 서버 상태 확인
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        
        except Exception as e:
            logger.debug(f"Ollama server check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """설치된 모델 목록 가져오기"""
        if self._available_models is not None:
            return self._available_models
        
        try:
            import urllib.request
            
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                models = []
                for model_info in data.get("models", []):
                    model_name = model_info.get("name", "")
                    if model_name:
                        models.append(model_name)
                
                self._available_models = models
                logger.info(f"Available models: {models}")
                return models
        
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def is_model_installed(self, model_name: str) -> bool:
        """모델이 설치되어 있는지 확인"""
        available_models = self.get_available_models()
        
        # 정확한 이름 매칭
        if model_name in available_models:
            return True
        
        # 태그 없는 버전으로도 확인 (예: qwen2.5:3b-instruct-q4_K_M -> qwen2.5)
        base_name = model_name.split(':')[0]
        for model in available_models:
            if model.startswith(base_name):
                return True
        
        return False
    
    def install_model(self, model_name: str, timeout: int = 300) -> bool:
        """
        모델 자동 설치
        
        Args:
            model_name: 설치할 모델 이름
            timeout: 타임아웃 (초)
            
        Returns:
            설치 성공 여부
        """
        if self.is_model_installed(model_name):
            logger.info(f"Model already installed: {model_name}")
            return True
        
        logger.info(f"Installing model: {model_name}")
        
        try:
            import urllib.request
            import json
            
            # 모델 풀 요청
            data = {
                "name": model_name,
                "stream": False
            }
            
            req = urllib.request.Request(
                f"{self.base_url}/api/pull",
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode())
                
                if result.get("status") == "success":
                    logger.info(f"Model installed successfully: {model_name}")
                    self._available_models = None  # 캐시 무효화
                    return True
                else:
                    logger.error(f"Model installation failed: {result}")
                    return False
        
        except Exception as e:
            logger.error(f"Failed to install model {model_name}: {e}", exc_info=True)
            return False
    
    def ensure_model_available(self, model_name: str) -> bool:
        """
        모델이 사용 가능한지 확인하고 필요시 설치
        
        Args:
            model_name: 확인할 모델 이름
            
        Returns:
            모델 사용 가능 여부
        """
        # Ollama 서버 실행 확인
        if not self.check_ollama_running():
            logger.error("Ollama server is not running")
            logger.info("Please start Ollama server: ollama serve")
            return False
        
        # 모델 설치 확인 및 자동 설치
        if not self.is_model_installed(model_name):
            logger.warning(f"Model not found: {model_name}")
            logger.info(f"Attempting to install model: {model_name}")
            
            if not self.install_model(model_name):
                logger.error(f"Failed to install model: {model_name}")
                return False
        
        logger.info(f"Model is available: {model_name}")
        return True
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """모델 정보 가져오기"""
        try:
            import urllib.request
            
            req = urllib.request.Request(f"{self.base_url}/api/show")
            data = {"name": model_name}
            
            req = urllib.request.Request(
                f"{self.base_url}/api/show",
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode())
        
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
    
    def start_ollama_if_needed(self) -> bool:
        """필요시 Ollama 서버 시작 (Windows)"""
        if self.check_ollama_running():
            return True
        
        logger.info("Attempting to start Ollama server...")
        
        try:
            # Windows에서 Ollama 서버 시작
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            
            # 서버 시작 대기
            for _ in range(30):  # 최대 30초 대기
                time.sleep(1)
                if self.check_ollama_running():
                    logger.info("Ollama server started successfully")
                    return True
            
            logger.error("Failed to start Ollama server")
            return False
        
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False


# 전역 인스턴스
ollama_manager = OllamaManager()

