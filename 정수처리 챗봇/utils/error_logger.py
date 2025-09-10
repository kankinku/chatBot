#!/usr/bin/env python3
"""
에러 로그 전용 수집 시스템

에러 발생 시 별도 파일로 수집하여 가독성 좋게 저장
- 에러 날짜
- 에러 위치 (파일명, 함수명, 라인 번호)
- 에러 내용
- 스택 트레이스
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import queue

class ErrorLogger:
    """에러 로그 전용 수집기"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # 로그 디렉토리 설정
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # 에러 로그 파일 경로
        self.error_log_path = self.log_dir / "error_log.jsonl"
        
        # 큐와 워커 스레드 설정
        self._error_queue = queue.Queue()
        self._worker_thread = None
        self._shutdown_event = threading.Event()
        
        # 워커 스레드 시작
        self._start_worker()
        
        self._initialized = True
    
    def _start_worker(self):
        """워커 스레드 시작"""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def _worker_loop(self):
        """워커 스레드 루프"""
        while not self._shutdown_event.is_set():
            try:
                # 타임아웃으로 주기적으로 체크
                error_data = self._error_queue.get(timeout=1.0)
                self._write_error_to_file(error_data)
                self._error_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                # 에러 로거 자체의 에러는 콘솔에 출력
                print(f"에러 로거 워커 오류: {e}", file=sys.stderr)
    
    def _write_error_to_file(self, error_data: Dict[str, Any]):
        """에러 데이터를 파일에 기록 (새로운 에러를 맨 앞에 추가)"""
        try:
            # JSON 형태로 가독성 좋게 저장
            formatted_error = json.dumps(error_data, ensure_ascii=False, indent=2)
            new_entry = formatted_error + '\n' + '='*80 + '\n'
            
            # Windows 파일 잠금 문제 해결을 위한 강화된 재시도 로직
            max_retries = 5
            base_delay = 0.1
            
            for attempt in range(max_retries):
                try:
                    # 기존 파일 내용 읽기
                    existing_content = ""
                    if self.error_log_path.exists():
                        try:
                            with open(self.error_log_path, 'r', encoding='utf-8') as f:
                                existing_content = f.read()
                        except (OSError, IOError, PermissionError):
                            # 파일 읽기 실패 시 빈 내용으로 처리
                            existing_content = ""
                    
                    # 새로운 에러를 맨 앞에 추가
                    with open(self.error_log_path, 'w', encoding='utf-8') as f:
                        f.write(new_entry)
                        if existing_content:
                            f.write(existing_content)
                        f.flush()  # 즉시 디스크에 쓰기
                        os.fsync(f.fileno())  # 강제 동기화
                    
                    break  # 성공 시 루프 종료
                    
                except (OSError, IOError, PermissionError) as e:
                    error_code = getattr(e, 'winerror', None) if hasattr(e, 'winerror') else None
                    
                    # WinError 32 (파일이 다른 프로세스에서 사용 중) 또는 PermissionError
                    if error_code == 32 or isinstance(e, PermissionError):
                        if attempt < max_retries - 1:
                            # 지수 백오프로 대기 시간 증가
                            delay = base_delay * (2 ** attempt)
                            time.sleep(delay)
                            continue
                        else:
                            # 최종 시도 실패 시 임시 파일로 기록
                            self._write_to_temp_file(new_entry)
                            break
                    else:
                        # 다른 종류의 에러는 즉시 재시도
                        if attempt < max_retries - 1:
                            time.sleep(base_delay)
                            continue
                        else:
                            raise e
                
        except Exception as e:
            print(f"에러 로그 파일 쓰기 실패: {e}", file=sys.stderr)
    
    def _write_to_temp_file(self, content: str):
        """임시 파일에 에러 로그 기록 (메인 파일이 잠겨있을 때 사용)"""
        try:
            temp_path = self.error_log_path.with_suffix('.tmp')
            with open(temp_path, 'a', encoding='utf-8') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            print(f"에러 로그를 임시 파일에 기록: {temp_path}", file=sys.stderr)
        except Exception as e:
            print(f"임시 파일 기록도 실패: {e}", file=sys.stderr)
    
    def log_error(self, 
                  error: Exception, 
                  context: Optional[str] = None,
                  additional_info: Optional[Dict[str, Any]] = None):
        """
        에러 로그 기록
        
        Args:
            error: 발생한 예외 객체
            context: 에러 발생 컨텍스트 (선택사항)
            additional_info: 추가 정보 (선택사항)
        """
        try:
            # 현재 시간
            timestamp = datetime.now().isoformat()
            
            # 스택 트레이스 수집
            tb_lines = traceback.format_exc().strip().split('\n')
            
            # 에러 위치 정보 추출
            error_location = self._extract_error_location(tb_lines)
            
            # 에러 데이터 구성
            error_data = {
                "에러_날짜": timestamp,
                "에러_위치": error_location,
                "에러_내용": {
                    "예외_타입": type(error).__name__,
                    "에러_메시지": str(error),
                    "컨텍스트": context
                },
                "스택_트레이스": tb_lines,
                "추가_정보": additional_info or {}
            }
            
            # 비동기 큐에 추가
            try:
                self._error_queue.put_nowait(error_data)
            except queue.Full:
                # 큐가 가득 찬 경우 직접 처리
                self._write_error_to_file(error_data)
                
        except Exception as e:
            print(f"에러 로그 기록 실패: {e}", file=sys.stderr)
    
    def _extract_error_location(self, tb_lines: list) -> Dict[str, str]:
        """스택 트레이스에서 에러 위치 정보 추출"""
        location = {
            "파일명": "알 수 없음",
            "함수명": "알 수 없음", 
            "라인_번호": "알 수 없음"
        }
        
        try:
            # 마지막 traceback line에서 위치 정보 추출
            for line in reversed(tb_lines):
                if 'File "' in line and 'line ' in line:
                    # "File "path", line X, in function" 형태 파싱
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        # 파일 경로 추출
                        file_part = parts[0].replace('File "', '').replace('"', '')
                        location["파일명"] = os.path.basename(file_part)
                        
                        # 라인 번호 추출
                        line_part = parts[1].replace('line ', '')
                        location["라인_번호"] = line_part
                        
                        # 함수명 추출
                        func_part = parts[2].replace('in ', '')
                        location["함수명"] = func_part
                        
                    break
                    
        except Exception:
            # 파싱 실패 시 기본값 유지
            pass
            
        return location
    
    def log_system_error(self, 
                        message: str,
                        module: str = "unknown",
                        additional_info: Optional[Dict[str, Any]] = None):
        """시스템 에러 로그 기록"""
        try:
            timestamp = datetime.now().isoformat()
            
            error_data = {
                "에러_날짜": timestamp,
                "에러_위치": {
                    "파일명": "system",
                    "함수명": module,
                    "라인_번호": "N/A"
                },
                "에러_내용": {
                    "예외_타입": "SystemError",
                    "에러_메시지": message,
                    "컨텍스트": "시스템 에러"
                },
                "스택_트레이스": [message],
                "추가_정보": additional_info or {}
            }
            
            try:
                self._error_queue.put_nowait(error_data)
            except queue.Full:
                self._write_error_to_file(error_data)
                
        except Exception as e:
            print(f"시스템 에러 로그 기록 실패: {e}", file=sys.stderr)
    
    def shutdown(self):
        """에러 로거 종료"""
        self._shutdown_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

# 전역 에러 로거 인스턴스
error_logger = ErrorLogger()

# 편의 함수들
def log_error(error: Exception, 
              context: Optional[str] = None,
              additional_info: Optional[Dict[str, Any]] = None):
    """에러 로그 기록 편의 함수"""
    error_logger.log_error(error, context, additional_info)

def log_system_error(message: str,
                    module: str = "unknown", 
                    additional_info: Optional[Dict[str, Any]] = None):
    """시스템 에러 로그 기록 편의 함수"""
    error_logger.log_system_error(message, module, additional_info)

# 에러 로그 데코레이터
def catch_and_log_errors(context: Optional[str] = None):
    """에러를 자동으로 캐치하고 로그에 기록하는 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or f"{func.__module__}.{func.__name__}"
                log_error(e, error_context, {
                    "함수_인자": str(args)[:200] + "..." if len(str(args)) > 200 else str(args),
                    "키워드_인자": str(kwargs)[:200] + "..." if len(str(kwargs)) > 200 else str(kwargs)
                })
                raise
        return wrapper
    return decorator
