# 메모리 최적화 설정 파일
# memory_config.py

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MemoryConfig:
    """메모리 최적화 설정"""
    
    # 시스템 메모리 설정
    max_model_memory_gb: float = 8.0
    memory_warning_threshold: float = 75.0
    memory_critical_threshold: float = 85.0
    
    # PDF 처리 메모리 설정
    pdf_max_memory_gb: float = 2.0
    pdf_chunk_size: int = 256
    pdf_chunk_overlap: int = 30
    pdf_batch_size: int = 16
    
    # 모델별 메모리 사용량 (GB)
    model_memory_usage: Dict[str, float] = None
    
    # 모델별 우선순위 (낮을수록 높은 우선순위)
    model_priorities: Dict[str, int] = None
    
    # 모니터링 설정
    monitoring_interval: float = 30.0
    enable_auto_cleanup: bool = True
    
    def __post_init__(self):
        if self.model_memory_usage is None:
            self.model_memory_usage = {
                "beomi/KoAlpaca-Polyglot-5.8B": 3.0,
                "jhgan/ko-sroberta-multitask": 0.5,
                # "defog/sqlcoder-7b-2": 4.0  # SQL 모델 제거됨
            }
        
        if self.model_priorities is None:
            self.model_priorities = {
                "beomi/KoAlpaca-Polyglot-5.8B": 1,  # 핵심 모델
                "jhgan/ko-sroberta-multitask": 1,  # 핵심 모델
                # "defog/sqlcoder-7b-2": 2  # SQL 모델 제거됨
            }
    
    @classmethod
    def from_env(cls):
        """환경 변수에서 설정 로드"""
        return cls(
            max_model_memory_gb=float(os.getenv('MAX_MODEL_MEMORY_GB', '8.0')),
            memory_warning_threshold=float(os.getenv('MEMORY_WARNING_THRESHOLD', '75.0')),
            memory_critical_threshold=float(os.getenv('MEMORY_CRITICAL_THRESHOLD', '85.0')),
            pdf_max_memory_gb=float(os.getenv('PDF_MAX_MEMORY_GB', '2.0')),
            monitoring_interval=float(os.getenv('MEMORY_MONITORING_INTERVAL', '30.0')),
            enable_auto_cleanup=os.getenv('ENABLE_AUTO_CLEANUP', 'true').lower() == 'true'
        )

# 전역 설정 인스턴스
memory_config = MemoryConfig.from_env()

def get_memory_config() -> MemoryConfig:
    """메모리 설정 반환"""
    return memory_config

def update_memory_config(**kwargs):
    """메모리 설정 업데이트"""
    global memory_config
    for key, value in kwargs.items():
        if hasattr(memory_config, key):
            setattr(memory_config, key, value)
    
    return memory_config
