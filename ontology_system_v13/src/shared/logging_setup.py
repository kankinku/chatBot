import logging
import logging.handlers
import sys
from pathlib import Path
from config.settings import get_settings

def setup_logging():
    """로깅 설정을 초기화합니다."""
    settings = get_settings()
    log_settings = settings.logging
    
    # 로그 디렉토리 생성
    log_dir = settings.project_root / log_settings.log_dir
    log_dir.mkdir(exist_ok=True)
    
    log_path = log_dir / log_settings.log_file
    
    # Root Logger 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_settings.log_level)
    
    # 기존 핸들러 제거 (중복 방지)
    root_logger.handlers = []
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 2. File Handler (Rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=log_settings.max_bytes,
        backupCount=log_settings.backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized. Log file: {log_path}")

