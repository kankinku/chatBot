"""
Normalizer - 정규화

수치, 단위, 날짜 등을 정규화합니다 (단일 책임).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Tuple, Optional

from config.constants import UNIT_SYNONYMS, UNIT_CONVERSIONS
from modules.core.logger import get_logger

logger = get_logger(__name__)


class Normalizer:
    """
    일반 정규화기
    
    단일 책임: 기본적인 텍스트 정규화만 수행
    """
    
    def normalize_number(self, text: str) -> str:
        """숫자에서 쉼표 제거"""
        return text.replace(",", "")
    
    def normalize_unit(self, unit: str) -> str:
        """단위 정규화"""
        return unit.strip().lower().replace("μ", "µ").replace(" ", "")
    
    def normalize_date(self, date_str: str) -> Optional[str]:
        """
        날짜 문자열 정규화
        
        다양한 날짜 형식을 YYYY-MM-DD 형식으로 변환합니다.
        """
        if not date_str:
            return None
        
        # 이미 정규화된 형식
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return date_str
        
        # 한국어 형식: 2025년 2월 17일
        match = re.match(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", date_str)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # 점 구분자: 2025.2.17
        match = re.match(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", date_str)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # 슬래시 구분자: 2025/2/17
        match = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})", date_str)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        return date_str


class MeasurementNormalizer:
    """
    측정값 정규화기
    
    단일 책임: 수치-단위 쌍 추출 및 정규화
    """
    
    def __init__(self):
        self.normalizer = Normalizer()
        logger.debug("MeasurementNormalizer initialized")
    
    def extract_measurements(self, text: str) -> List[Tuple[str, str]]:
        """
        텍스트에서 측정값 추출
        
        Args:
            text: 입력 텍스트
            
        Returns:
            [(숫자, 단위), ...] 리스트
        """
        pairs: List[Tuple[str, str]] = []
        
        # 기본 수치-단위 패턴
        for m in re.finditer(r"(\d+(?:[\.,]\d+)?)\s*([A-Za-z%°℃µμ/]+)", text):
            num = self.normalizer.normalize_number(m.group(1))
            unit = self.normalizer.normalize_unit(m.group(2))
            pairs.append((num, unit))
        
        # 상관계수 패턴
        for m in re.finditer(
            r"(?:상관계수|correlation|R²|R2|R-squared)\s*[=:]\s*(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE
        ):
            num = self.normalizer.normalize_number(m.group(1))
            pairs.append((num, "correlation"))
        
        # 날짜 패턴
        for m in re.finditer(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", text):
            year, month, day = m.group(1), m.group(2), m.group(3)
            formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            pairs.append((formatted_date, "date"))
        
        # 백분율 패턴
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
            num = self.normalizer.normalize_number(m.group(1))
            pairs.append((num, "percent"))
        
        # 범위 패턴: 6.5-8.5 pH
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*([A-Za-z%°℃µμ/]+)", text):
            min_val = self.normalizer.normalize_number(m.group(1))
            max_val = self.normalizer.normalize_number(m.group(2))
            unit = self.normalizer.normalize_unit(m.group(3))
            pairs.append((f"{min_val}-{max_val}", unit))
        
        return pairs
    
    def units_equivalent(self, unit_a: str, unit_b: str) -> bool:
        """두 단위가 동일한지 확인"""
        a = self.normalizer.normalize_unit(unit_a)
        b = self.normalizer.normalize_unit(unit_b)
        
        if a == b:
            return True
        
        return b in UNIT_SYNONYMS.get(a, set()) or a in UNIT_SYNONYMS.get(b, set())
    
    def convert_value(
        self,
        value: float,
        unit_from: str,
        unit_to: str
    ) -> Optional[float]:
        """
        단위 변환
        
        Args:
            value: 변환할 값
            unit_from: 원본 단위
            unit_to: 목표 단위
            
        Returns:
            변환된 값, 변환 불가능하면 None
        """
        uf = self.normalizer.normalize_unit(unit_from)
        ut = self.normalizer.normalize_unit(unit_to)
        
        # 직접 변환
        if (uf, ut) in UNIT_CONVERSIONS:
            return value * UNIT_CONVERSIONS[(uf, ut)]
        
        # 동일 단위
        if self.units_equivalent(uf, ut):
            return value
        
        return None
    
    def build_measurement_map(self, texts: List[str]) -> dict:
        """
        텍스트 리스트에서 측정값 맵 생성
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            {단위: [숫자, ...], ...}
        """
        measurement_map = {}
        
        for text in texts:
            measurements = self.extract_measurements(text)
            for num, unit in measurements:
                measurement_map.setdefault(unit, []).append(num)
        
        return measurement_map
    
    def verify_numeric_preservation(
        self,
        answer: str,
        context_map: dict,
        tolerance: float = 0.05
    ) -> float:
        """
        답변의 숫자 보존율 검증
        
        Args:
            answer: 답변 텍스트
            context_map: 컨텍스트 측정값 맵
            tolerance: 허용 오차 비율
            
        Returns:
            보존율 (0.0 ~ 1.0)
        """
        answer_measurements = self.extract_measurements(answer)
        
        if not answer_measurements:
            return 1.0  # 숫자가 없으면 완벽하게 보존
        
        matched = 0
        
        for num, unit in answer_measurements:
            # 컨텍스트에서 동일하거나 변환 가능한 단위 찾기
            candidates: List[float] = []
            
            for ctx_unit, ctx_nums in context_map.items():
                for ctx_num in ctx_nums:
                    try:
                        ctx_val = float(ctx_num)
                    except ValueError:
                        continue
                    
                    # 단위 변환 시도
                    converted = self.convert_value(ctx_val, ctx_unit, unit)
                    if converted is not None:
                        candidates.append(converted)
            
            if not candidates:
                continue
            
            # 답변 숫자와 비교
            try:
                answer_val = float(num)
            except ValueError:
                continue
            
            # 허용 오차 내에 있는지 확인
            for candidate in candidates:
                if candidate == 0:
                    continue
                relative_error = abs(answer_val - candidate) / abs(candidate)
                if relative_error <= tolerance:
                    matched += 1
                    break
        
        return matched / len(answer_measurements)

