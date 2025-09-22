"""
메모리 최적화 유틸리티

AI 모델과 대용량 데이터 처리에서 메모리 사용량을 최적화하는 도구들을 제공합니다.
"""

import gc
import psutil
import os
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from contextlib import contextmanager
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

@dataclass
class MemoryInfo:
    """메모리 정보 데이터 클래스"""
    total: float  # GB
    available: float  # GB
    used: float  # GB
    percent: float  # %
    process_memory: float  # GB

@dataclass
class ModelMemoryInfo:
    """모델별 메모리 정보"""
    model_name: str
    memory_usage_gb: float
    last_used: float
    priority: int  # 1: 높음, 2: 중간, 3: 낮음
    is_loaded: bool = True

class MemoryOptimizer:
    """메모리 최적화 관리자"""
    
    def __init__(self, warning_threshold: float = 75.0, critical_threshold: float = 85.0):
        """
        MemoryOptimizer 초기화
        
        Args:
            warning_threshold: 경고 임계치 (%)
            critical_threshold: 위험 임계치 (%)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring_enabled = False
        self.monitor_thread = None
        self.cleanup_callbacks = []
        
    def get_memory_info(self) -> MemoryInfo:
        """현재 메모리 정보 반환"""
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        
        return MemoryInfo(
            total=memory.total / (1024**3),
            available=memory.available / (1024**3),
            used=memory.used / (1024**3),
            percent=memory.percent,
            process_memory=process.memory_info().rss / (1024**3)
        )
    
    def force_garbage_collection(self) -> int:
        """강제 가비지 컬렉션 실행"""
        before = len(gc.get_objects())
        
        # 3세대 모두 정리
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        after = len(gc.get_objects())
        freed = before - after
        
        logger.info(f"가비지 컬렉션 완료: {collected}개 객체 수집, {freed}개 객체 해제")
        return collected
    
    def clear_torch_cache(self):
        """PyTorch GPU 캐시 정리"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("PyTorch GPU 캐시 정리 완료")
        except ImportError:
            pass
    
    def optimize_memory(self, aggressive: bool = False) -> MemoryInfo:
        """
        메모리 최적화 실행
        
        Args:
            aggressive: 적극적 정리 모드
            
        Returns:
            최적화 후 메모리 정보
        """
        before_info = self.get_memory_info()
        logger.info(f"메모리 최적화 시작 - 사용량: {before_info.percent:.1f}%")
        
        # 1. 가비지 컬렉션
        self.force_garbage_collection()
        
        # 2. PyTorch 캐시 정리
        self.clear_torch_cache()
        
        # 3. 등록된 정리 콜백 실행
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"정리 콜백 실행 실패: {e}")
        
        # 4. 적극적 모드인 경우 추가 정리
        if aggressive:
            self._aggressive_cleanup()
        
        after_info = self.get_memory_info()
        saved = before_info.used - after_info.used
        
        logger.info(f"메모리 최적화 완료 - 절약: {saved:.2f}GB, 현재 사용량: {after_info.percent:.1f}%")
        return after_info
    
    def _aggressive_cleanup(self):
        """적극적 메모리 정리"""
        # 모든 세대 가비지 컬렉션을 여러 번 실행
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)
        
        # 시스템 캐시 정리 힌트
        try:
            if hasattr(os, 'sync'):
                os.sync()
        except:
            pass
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """메모리 정리 콜백 등록"""
        self.cleanup_callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 30.0):
        """메모리 모니터링 시작 (간격 단축)"""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitor_thread = threading.Thread(
            target=self._memory_monitor,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"메모리 모니터링 시작 (간격: {interval}초)")
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.monitoring_enabled = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("메모리 모니터링 중지")
    
    def _memory_monitor(self, interval: float):
        """메모리 모니터링 루프"""
        while self.monitoring_enabled:
            try:
                memory_info = self.get_memory_info()
                
                if memory_info.percent >= self.critical_threshold:
                    logger.warning(f"메모리 사용량 위험: {memory_info.percent:.1f}% - 자동 정리 실행")
                    self.optimize_memory(aggressive=True)
                    # 모델 메모리 매니저에게 알림
                    model_memory_manager.handle_memory_pressure()
                elif memory_info.percent >= self.warning_threshold:
                    logger.warning(f"메모리 사용량 경고: {memory_info.percent:.1f}% - 기본 정리 실행")
                    self.optimize_memory(aggressive=False)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"메모리 모니터링 오류: {e}")
                time.sleep(interval)

@contextmanager
def memory_limit(max_memory_gb: float):
    """메모리 사용량 제한 컨텍스트 매니저"""
    optimizer = MemoryOptimizer()
    initial_info = optimizer.get_memory_info()
    
    try:
        yield optimizer
    finally:
        final_info = optimizer.get_memory_info()
        
        # 메모리 사용량이 제한을 초과한 경우 정리
        if final_info.process_memory > max_memory_gb:
            logger.warning(f"메모리 제한 초과: {final_info.process_memory:.2f}GB > {max_memory_gb}GB")
            optimizer.optimize_memory(aggressive=True)

@contextmanager
def memory_profiler(operation_name: str):
    """메모리 사용량 프로파일링 컨텍스트 매니저"""
    optimizer = MemoryOptimizer()
    
    before_info = optimizer.get_memory_info()
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        after_info = optimizer.get_memory_info()
        
        memory_diff = after_info.process_memory - before_info.process_memory
        duration = end_time - start_time
        
        # 중요한 메모리 변화나 긴 실행 시간일 때만 로깅
        if abs(memory_diff) >= 0.1 or duration >= 5.0:
            logger.info(f"메모리 프로파일 [{operation_name}]: "
                       f"사용량 변화: {memory_diff:+.3f}GB, "
                       f"실행 시간: {duration:.3f}초, "
                       f"현재 사용량: {after_info.process_memory:.3f}GB")

class ModelMemoryManager:
    """AI 모델 메모리 관리자 (LRU 캐시 기반)"""
    
    def __init__(self, max_total_memory_gb: float = 8.0):
        self.loaded_models: OrderedDict[str, ModelMemoryInfo] = OrderedDict()
        self.model_objects: Dict[str, Any] = {}
        self.optimizer = MemoryOptimizer()
        self.max_total_memory_gb = max_total_memory_gb
        self.lock = threading.Lock()
        
        # 모델별 기본 메모리 사용량 (GB)
        self.default_memory_usage = {
            "beomi/KoAlpaca-Polyglot-5.8B": 3.0,
            "jhgan/ko-sroberta-multitask": 0.5,
            "defog/sqlcoder-7b-2": 4.0
        }
        
        # 모델별 우선순위 (낮을수록 높은 우선순위)
        self.model_priorities = {
            "beomi/KoAlpaca-Polyglot-5.8B": 1,  # 핵심 모델
            "jhgan/ko-sroberta-multitask": 1,  # 핵심 모델
            "defog/sqlcoder-7b-2": 2  # 선택적 모델
        }
    
    def register_model(self, model_name: str, model_obj: Any, estimated_memory_gb: float = None):
        """모델 등록"""
        with self.lock:
            if estimated_memory_gb is None:
                estimated_memory_gb = self.default_memory_usage.get(model_name, 1.0)
            
            priority = self.model_priorities.get(model_name, 3)
            
            model_info = ModelMemoryInfo(
                model_name=model_name,
                memory_usage_gb=estimated_memory_gb,
                last_used=time.time(),
                priority=priority,
                is_loaded=True
            )
            
            self.loaded_models[model_name] = model_info
            self.model_objects[model_name] = model_obj
            
            logger.info(f"모델 등록: {model_name} (예상 메모리: {estimated_memory_gb:.2f}GB, 우선순위: {priority})")
            
            # 메모리 사용량 체크
            self._check_memory_limits()
    
    def get_model(self, model_name: str) -> Any:
        """모델 획득 (LRU 업데이트)"""
        with self.lock:
            if model_name in self.loaded_models:
                # LRU 업데이트
                model_info = self.loaded_models.pop(model_name)
                model_info.last_used = time.time()
                self.loaded_models[model_name] = model_info
                
                logger.debug(f"모델 사용: {model_name}")
                return self.model_objects[model_name]
            else:
                logger.warning(f"모델이 로드되지 않음: {model_name}")
                return None
    
    def unload_model(self, model_name: str):
        """모델 언로드"""
        with self.lock:
            if model_name in self.loaded_models:
                model_info = self.loaded_models.pop(model_name)
                model_obj = self.model_objects.pop(model_name, None)
                
                # 모델 객체 정리
                if model_obj is not None:
                    try:
                        del model_obj
                    except:
                        pass
                
                # 메모리 정리
                self.optimizer.force_garbage_collection()
                self.optimizer.clear_torch_cache()
                
                logger.info(f"모델 언로드: {model_name} (해제된 메모리: {model_info.memory_usage_gb:.2f}GB)")
    
    def handle_memory_pressure(self):
        """메모리 압박 상황 처리"""
        with self.lock:
            current_usage = self.get_total_model_memory()
            
            if current_usage <= self.max_total_memory_gb:
                return
            
            logger.warning(f"메모리 압박 감지: {current_usage:.2f}GB > {self.max_total_memory_gb}GB")
            
            # 우선순위가 낮은 모델부터 언로드
            models_to_unload = []
            for model_name, model_info in self.loaded_models.items():
                models_to_unload.append((model_name, model_info.priority, model_info.last_used))
            
            # 우선순위, 마지막 사용 시간 순으로 정렬
            models_to_unload.sort(key=lambda x: (x[1], -x[2]))
            
            for model_name, _, _ in models_to_unload:
                self.unload_model(model_name)
                current_usage = self.get_total_model_memory()
                
                if current_usage <= self.max_total_memory_gb * 0.8:  # 20% 여유 확보
                    break
    
    def _check_memory_limits(self):
        """메모리 제한 체크"""
        current_usage = self.get_total_model_memory()
        
        if current_usage > self.max_total_memory_gb:
            logger.warning(f"모델 메모리 제한 초과: {current_usage:.2f}GB > {self.max_total_memory_gb}GB")
            self.handle_memory_pressure()
    
    def get_total_model_memory(self) -> float:
        """총 모델 메모리 사용량 반환"""
        return sum(model_info.memory_usage_gb for model_info in self.loaded_models.values())
    
    def get_model_status(self) -> Dict[str, Any]:
        """모델 상태 정보 반환"""
        with self.lock:
            return {
                "loaded_models": list(self.loaded_models.keys()),
                "total_memory_gb": self.get_total_model_memory(),
                "max_memory_gb": self.max_total_memory_gb,
                "model_details": {
                    name: {
                        "memory_gb": info.memory_usage_gb,
                        "priority": info.priority,
                        "last_used": info.last_used
                    }
                    for name, info in self.loaded_models.items()
                }
            }

# 전역 인스턴스들
memory_optimizer = MemoryOptimizer()
model_memory_manager = ModelMemoryManager()

def optimize_system_memory():
    """시스템 메모리 최적화 (편의 함수)"""
    return memory_optimizer.optimize_memory()

def start_memory_monitoring():
    """메모리 모니터링 시작 (편의 함수)"""
    memory_optimizer.start_monitoring()

def stop_memory_monitoring():
    """메모리 모니터링 중지 (편의 함수)"""
    memory_optimizer.stop_monitoring()

if __name__ == "__main__":
    # 테스트 코드
    optimizer = MemoryOptimizer()
    
    # 메모리 정보 출력
    info = optimizer.get_memory_info()
    print(f"메모리 사용량: {info.percent:.1f}% ({info.used:.2f}GB / {info.total:.2f}GB)")
    
    # 메모리 최적화 테스트
    with memory_profiler("테스트 작업"):
        # 큰 리스트 생성
        big_list = [i for i in range(1000000)]
        del big_list
    
    # 메모리 최적화
    optimizer.optimize_memory()
    
    print("메모리 최적화 테스트 완료")
