"""
OCR Corrector - OCR 후처리

OCR 결과의 오류를 감지하고 수정합니다 (단일 책임).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import List, Optional

from config.model_config import LLMModelConfig
from modules.core.exceptions import OCRCorrectionError
from modules.core.logger import get_logger

logger = get_logger(__name__)


class OCRCorrector:
    """
    OCR 오류 수정기
    
    단일 책임: OCR 후처리만 수행
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMModelConfig] = None,
        domain_dict_path: Optional[str] = None,
        noise_threshold: float = 0.5,
        max_chars_per_call: int = 10000,
    ):
        """
        Args:
            llm_config: LLM 설정
            domain_dict_path: 도메인 사전 경로
            noise_threshold: 노이즈 임계값
            max_chars_per_call: 한 번에 처리할 최대 문자 수
        """
        self.llm_config = llm_config or LLMModelConfig()
        self.domain_dict_path = domain_dict_path
        self.noise_threshold = noise_threshold
        self.max_chars_per_call = max_chars_per_call
        
        # 도메인 사전 로드
        self.domain_dict = self._load_domain_dict()
        
        logger.info("OCRCorrector initialized", config={
            "noise_threshold": noise_threshold,
            "max_chars": max_chars_per_call,
            "has_domain_dict": self.domain_dict is not None,
        })
    
    def _load_domain_dict(self) -> Optional[str]:
        """도메인 사전 로드"""
        if not self.domain_dict_path:
            return None
        
        try:
            path = Path(self.domain_dict_path)
            if not path.exists():
                logger.warning(f"Domain dictionary not found: {self.domain_dict_path}")
                return None
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 중요한 용어만 추출
            important_terms = []
            for key in ["keywords", "units", "technical_spec", "system_info"]:
                if key in data:
                    important_terms.extend(data[key][:20])  # 각 카테고리에서 상위 20개
            
            return ", ".join(important_terms[:50])  # 최대 50개
        
        except Exception as e:
            logger.warning(f"Failed to load domain dictionary: {e}")
            return None
    
    def calculate_noise_score(self, text: str) -> float:
        """
        텍스트의 노이즈 점수 계산
        
        Args:
            text: 입력 텍스트
            
        Returns:
            노이즈 점수 (0.0 ~ 1.0)
        """
        if not text:
            return 0.0
        
        n = len(text)
        
        # 깨진 문자 수
        bad_chars = text.count("�")
        
        # OCR 오류 패턴
        suspicious_patterns = ["rn", "cl", "0O", "O0", "l1", "1l", "mg|L", "m|n"]
        suspicious_count = sum(text.count(pattern) for pattern in suspicious_patterns)
        
        # 구두점 비율 (너무 많으면 이상함)
        punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        # 점수 계산
        score = (
            0.4 * (bad_chars / n) +
            0.4 * (suspicious_count / max(1, n / 20)) +
            0.2 * (punct_count / n)
        )
        
        return max(0.0, min(1.0, score))
    
    def select_low_quality_texts(
        self,
        texts: List[str],
        max_chars: Optional[int] = None,
    ) -> List[int]:
        """
        낮은 품질의 텍스트 선택
        
        Args:
            texts: 텍스트 리스트
            max_chars: 최대 문자 수 제한
            
        Returns:
            선택된 텍스트의 인덱스 리스트
        """
        max_chars = max_chars or self.max_chars_per_call
        
        # 노이즈 점수 계산
        scored = [
            (i, self.calculate_noise_score(text), len(text))
            for i, text in enumerate(texts)
        ]
        
        # 점수 높은 순으로 정렬
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # 임계값 이상이고 예산 내에 있는 것만 선택
        selected = []
        budget = 0
        
        for idx, score, length in scored:
            if score < self.noise_threshold:
                break
            if budget + length > max_chars:
                break
            
            selected.append(idx)
            budget += length
        
        logger.debug(f"Selected {len(selected)} texts for OCR correction", 
                    total_chars=budget)
        
        return selected
    
    def format_correction_prompt(self, text: str) -> str:
        """
        OCR 교정 프롬프트 생성
        
        Args:
            text: 교정할 텍스트
            
        Returns:
            프롬프트
        """
        parts = [
            "당신은 정수장 전문 OCR 후교정기입니다. 절대 재작성/요약/삭제를 하지 마세요.",
            "",
            "규칙:",
            "1) 줄바꿈/공백/번호/구두점을 유지",
            "2) 숫자/단위/기호 보존(예: mg/L, ppm, ℃, L/s, m³/d, NTU, pH, DO)",
            "3) 확실한 OCR 오류만 교정(rn→m, 0↔O, l↔1, cl→d, mg|L→mg/L 등)",
            "4) 정수장 전문 용어는 사전 기준으로 교정, 불확실하면 원문 그대로",
            "5) 수치 범위 확인: pH(6.5-8.5), 탁도(0-0.5NTU), 잔류염소(0.1-0.4mg/L)",
            "",
            "출력: 교정된 본문만 출력",
        ]
        
        if self.domain_dict:
            parts.extend([
                "",
                "[정수장 전문 사전]",
                self.domain_dict,
            ])
        
        parts.extend([
            "",
            "[원문]",
            text,
        ])
        
        return "\n".join(parts)
    
    def call_llm(self, prompt: str, timeout: int = 20) -> str:
        """
        LLM 호출
        
        Args:
            prompt: 프롬프트
            timeout: 타임아웃 (초)
            
        Returns:
            LLM 응답
        """
        url = f"{self.llm_config.get_ollama_url()}/api/generate"
        
        keep_alive = (
            f"{self.llm_config.keep_alive_minutes}m"
            if self.llm_config.keep_alive_minutes > 0
            else "0"
        )
        
        data = {
            "model": self.llm_config.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="ignore"))
                return body.get("response", "")
        except Exception as e:
            logger.error("LLM call failed for OCR correction", 
                        error=str(e), exc_info=True)
            return ""
    
    def correct_texts(
        self,
        texts: List[str],
        batch_size: int = 4,
    ) -> List[str]:
        """
        여러 텍스트 교정
        
        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기
            
        Returns:
            교정된 텍스트 리스트
        """
        if not texts:
            return []
        
        # 낮은 품질 텍스트 선택
        low_quality_indices = self.select_low_quality_texts(texts)
        
        if not low_quality_indices:
            logger.info("No low quality texts found for OCR correction")
            return texts
        
        logger.info(f"Correcting {len(low_quality_indices)} low quality texts")
        
        # 교정 수행
        result = list(texts)
        
        for i in range(0, len(low_quality_indices), batch_size):
            batch_indices = low_quality_indices[i:i + batch_size]
            
            for idx in batch_indices:
                original = texts[idx]
                prompt = self.format_correction_prompt(original)
                
                try:
                    corrected = self.call_llm(prompt, timeout=20)
                    if corrected.strip():
                        result[idx] = corrected.strip()
                        logger.debug(f"Corrected text at index {idx}")
                    else:
                        logger.warning(f"Empty correction result for index {idx}")
                
                except Exception as e:
                    logger.error(f"Failed to correct text at index {idx}", 
                                error=str(e), exc_info=True)
                    # 원본 유지
        
        return result
    
    def correct_single(self, text: str) -> str:
        """
        단일 텍스트 교정
        
        Args:
            text: 입력 텍스트
            
        Returns:
            교정된 텍스트
        """
        if not text:
            return ""
        
        # 노이즈 점수 확인
        noise_score = self.calculate_noise_score(text)
        
        if noise_score < self.noise_threshold:
            logger.debug(f"Text quality is good (noise={noise_score:.3f}), skipping correction")
            return text
        
        logger.info(f"Correcting text (noise={noise_score:.3f})")
        
        prompt = self.format_correction_prompt(text)
        
        try:
            corrected = self.call_llm(prompt, timeout=20)
            return corrected.strip() if corrected.strip() else text
        
        except Exception as e:
            raise OCRCorrectionError(
                text_sample=text[:100],
                cause=e,
            ) from e

