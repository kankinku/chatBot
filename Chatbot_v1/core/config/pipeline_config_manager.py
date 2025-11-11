#!/usr/bin/env python3
"""
파이프라인 설정 관리자

JSON 설정 파일에서 파이프라인별 설정을 읽어오는 모듈
환경 변수를 읽어서 범용 RAG 시스템으로 동작
"""

import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PipelineConfigManager:
    """파이프라인 설정 관리자"""
    
    def __init__(self, config_dir: str = "config/pipelines"):
        """파이프라인 설정 관리자 초기화"""
        self.config_dir = Path(config_dir)
        self.pipeline_configs = {}
        
        # 통합 설정에서 회사/프로젝트 정보 읽기(ENV/JSON/기본 순)
        try:
            from core.config.unified_config import get_config
        except Exception:
            def get_config(key, default=None):
                import os
                return os.getenv(key, default)
        self.company_name = str(get_config('COMPANY_NAME', '범용 RAG 시스템'))
        self.project_name = str(get_config('PROJECT_NAME', '범용 RAG 시스템'))
        self.system_type = str(get_config('SYSTEM_TYPE', '범용 RAG 시스템'))
        self.project_domain = str(get_config('PROJECT_DOMAIN', '범용'))
        self.project_industry = str(get_config('PROJECT_INDUSTRY', '범용'))
        
        logger.info(f"파이프라인 설정 관리자 초기화: {self.company_name} - {self.project_name}")
        
        self._load_pipeline_configs()
    
    def _load_pipeline_configs(self):
        """모든 파이프라인 설정 파일 로드"""
        try:
            if not self.config_dir.exists():
                logger.warning(f"설정 디렉토리가 존재하지 않습니다: {self.config_dir}")
                return
            
            # JSON 설정 파일들 로드
            config_files = {
                # "sql_pipeline.json": "SQL_QUERY",  # SQL 파이프라인 제거됨
                "pdf_pipeline.json": "PDF_SEARCH", 
                "greeting_pipeline.json": "GREETING"
            }
            
            for filename, pipeline_name in config_files.items():
                config_path = self.config_dir / filename
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            self.pipeline_configs[pipeline_name] = config
                            logger.info(f"파이프라인 설정 로드 완료: {pipeline_name}")
                    except Exception as e:
                        logger.error(f"파이프라인 설정 로드 실패 {filename}: {e}")
                else:
                    logger.warning(f"파이프라인 설정 파일이 존재하지 않습니다: {config_path}")
            
            logger.info(f"총 {len(self.pipeline_configs)}개 파이프라인 설정 로드 완료")
            
        except Exception as e:
            logger.error(f"파이프라인 설정 로드 중 오류 발생: {e}")
    
    def get_pipeline_config(self, pipeline_name: str) -> Optional[Dict]:
        """특정 파이프라인 설정 반환"""
        return self.pipeline_configs.get(pipeline_name)
    
    def get_pipeline_keywords(self, pipeline_name: str) -> List[str]:
        """특정 파이프라인의 키워드 목록 반환"""
        config = self.get_pipeline_config(pipeline_name)
        if config:
            return config.get("keywords", [])
        return []
    
    def get_pipeline_reference_questions(self, pipeline_name: str) -> List[str]:
        """특정 파이프라인의 참조 질문 목록 반환"""
        config = self.get_pipeline_config(pipeline_name)
        if config:
            return config.get("reference_questions", [])
        return []
    
    def get_pipeline_domain_keywords(self, pipeline_name: str) -> List[str]:
        """특정 파이프라인의 도메인 특화 키워드 목록 반환"""
        config = self.get_pipeline_config(pipeline_name)
        if config:
            return config.get("domain_specific_keywords", [])
        return []
    
    def get_all_pipeline_names(self) -> List[str]:
        """모든 파이프라인 이름 반환"""
        return list(self.pipeline_configs.keys())
    
    def reload_configs(self):
        """설정 파일 재로드"""
        self.pipeline_configs.clear()
        self._load_pipeline_configs()
        logger.info("파이프라인 설정 재로드 완료")
    
    def update_pipeline_config(self, pipeline_name: str, config: Dict):
        """파이프라인 설정 업데이트"""
        self.pipeline_configs[pipeline_name] = config
        logger.info(f"파이프라인 설정 업데이트 완료: {pipeline_name}")
    
    def save_pipeline_config(self, pipeline_name: str, config: Dict):
        """파이프라인 설정을 파일로 저장"""
        try:
            config_dir = Path(self.config_dir)
            config_dir.mkdir(parents=True, exist_ok=True)
            
            filename_map = {
                # "SQL_QUERY": "sql_pipeline.json",  # SQL 파이프라인 제거됨
                "PDF_SEARCH": "pdf_pipeline.json",
                "GREETING": "greeting_pipeline.json"
            }
            
            filename = filename_map.get(pipeline_name)
            if filename:
                config_path = config_dir / filename
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                logger.info(f"파이프라인 설정 저장 완료: {config_path}")
            else:
                logger.error(f"알 수 없는 파이프라인 이름: {pipeline_name}")
                
        except Exception as e:
            logger.error(f"파이프라인 설정 저장 실패: {e}")
    
    def get_pipeline_info(self) -> Dict:
        """모든 파이프라인 정보 반환"""
        info = {}
        for pipeline_name, config in self.pipeline_configs.items():
            info[pipeline_name] = {
                "description": config.get("description", ""),
                "keywords_count": len(config.get("keywords", [])),
                "reference_questions_count": len(config.get("reference_questions", [])),
                "domain_keywords_count": len(config.get("domain_specific_keywords", []))
            }
        return info
