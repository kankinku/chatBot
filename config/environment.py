"""
Environment Configuration

환경별 설정을 관리합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_VECTOR_STORE_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_FORMAT,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_PORT,
)


@dataclass
class EnvironmentConfig:
    """환경 설정"""
    
    # Environment
    env_name: str = "development"  # development, staging, production
    debug: bool = False
    
    # Paths
    data_dir: str = DEFAULT_DATA_DIR
    vector_store_dir: str = DEFAULT_VECTOR_STORE_DIR
    log_dir: str = DEFAULT_LOG_DIR
    
    # Logging
    log_level: str = DEFAULT_LOG_LEVEL
    log_format: str = DEFAULT_LOG_FORMAT
    
    # Ollama
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_port: int = DEFAULT_OLLAMA_PORT
    
    # Performance
    max_workers: int = 4
    enable_gpu: bool = False
    
    def __post_init__(self):
        """환경 변수에서 설정 오버라이드"""
        # Environment
        self.env_name = os.getenv("ENV_NAME", self.env_name)
        self.debug = os.getenv("DEBUG", str(self.debug)).lower() in {"true", "1", "yes"}
        
        # Paths
        self.data_dir = os.getenv("DATA_DIR", self.data_dir)
        self.vector_store_dir = os.getenv("VECTOR_STORE_DIR", self.vector_store_dir)
        self.log_dir = os.getenv("LOG_DIR", self.log_dir)
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.log_format = os.getenv("LOG_FORMAT", self.log_format)
        
        # Ollama
        self.ollama_host = os.getenv("OLLAMA_HOST", self.ollama_host)
        self.ollama_port = int(os.getenv("OLLAMA_PORT", str(self.ollama_port)))
        
        # Performance
        self.max_workers = int(os.getenv("MAX_WORKERS", str(self.max_workers)))
        self.enable_gpu = os.getenv("ENABLE_GPU", str(self.enable_gpu)).lower() in {"true", "1", "yes"}
        
        # 디렉토리 생성
        self.ensure_directories()
    
    def ensure_directories(self) -> None:
        """필요한 디렉토리 생성"""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    def is_production(self) -> bool:
        """프로덕션 환경인지 확인"""
        return self.env_name == "production"
    
    def is_development(self) -> bool:
        """개발 환경인지 확인"""
        return self.env_name == "development"


# Singleton instance
_env_config: Optional[EnvironmentConfig] = None


def get_env_config() -> EnvironmentConfig:
    """환경 설정 싱글톤 인스턴스 반환"""
    global _env_config
    if _env_config is None:
        _env_config = EnvironmentConfig()
    return _env_config


def reset_env_config() -> None:
    """환경 설정 초기화 (테스트용)"""
    global _env_config
    _env_config = None

