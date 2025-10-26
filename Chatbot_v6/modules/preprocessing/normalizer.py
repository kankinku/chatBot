"""
Normalizer - 정규화

수치, 단위, 날짜 등을 정규화합니다 (단일 책임).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from config.constants import UNIT_SYNONYMS, UNIT_CONVERSIONS
from modules.core.logger import get_logger

logger = get_logger(__name__)


class Normalizer:
    """
    일반 정규화기
    
    단일 책임: 기본적인 텍스트 정규화만 수행
    """
    
    def normalize_number(self, text: str) -> str:
        """숫자 정규화 강화 - 숫자 정확도 개선"""
        # 쉼표 제거
        text = text.replace(",", "")
        
        # 소수점 정규화 (점과 쉼표 혼용)
        text = re.sub(r'(\d+)[,.](\d+)', r'\1.\2', text)
        
        # 공백 제거 (숫자와 단위 사이)
        text = re.sub(r'(\d+)\s+([a-zA-Z%°℃]+)', r'\1\2', text)
        
        # 특수 문자 정규화
        text = text.replace('°', '도').replace('℃', '도')
        
        return text


class NumericExtractor:
    """
    숫자 추출기 - 숫자 정확도 개선
    
    단일 책임: 텍스트에서 숫자와 단위를 정확히 추출하고 검증
    """
    
    def __init__(self):
        """숫자 추출기 초기화"""
        # 숫자 패턴 (정수, 소수, 과학적 표기법)
        self.number_patterns = [
            r'\d+\.\d+',           # 소수
            r'\d+',                # 정수
            r'\d+\.\d+e[+-]?\d+',  # 과학적 표기법
        ]
        
        # 단위 패턴
        self.unit_patterns = [
            r'%', r'℃', r'°C', r'도', r'°',
            r'mg/L', r'ppm', r'ppb', r'μS/cm',
            r'm³/day', r'm³/h', r'L/min', r'L/s',
            r'bar', r'kPa', r'Pa', r'atm',
            r'kWh', r'MW', r'kW', r'W',
            r'kgCO2e', r'CO2', r'탄소',
            r'pH', r'NTU', r'TU', r'SS',
        ]
        
        logger.debug("NumericExtractor initialized")
    
    def extract_numbers_with_units(self, text: str) -> List[Dict[str, Any]]:
        """
        텍스트에서 숫자와 단위를 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            추출된 숫자-단위 쌍 리스트
        """
        results = []
        
        # 숫자-단위 패턴 매칭
        pattern = r'(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([a-zA-Z%°℃μ]+(?:\/[a-zA-Z]+)?)'
        matches = re.findall(pattern, text)
        
        for number_str, unit in matches:
            try:
                number = float(number_str)
                normalized_unit = self._normalize_unit(unit)
                
                results.append({
                    'number': number,
                    'unit': normalized_unit,
                    'original': f"{number_str} {unit}",
                    'confidence': self._calculate_confidence(number, normalized_unit)
                })
            except ValueError:
                continue
        
        # 단위 없는 숫자도 추출
        standalone_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        for num_str in standalone_numbers:
            try:
                number = float(num_str)
                results.append({
                    'number': number,
                    'unit': '',
                    'original': num_str,
                    'confidence': 0.5  # 단위가 없으면 낮은 신뢰도
                })
            except ValueError:
                continue
        
        logger.debug(f"Extracted {len(results)} numbers from text")
        return results
    
    def _normalize_unit(self, unit: str) -> str:
        """정수장 특화 단위 정규화"""
        unit = unit.strip().lower()
        
        # 정수장 도메인 특화 단위 매핑
        unit_mapping = {
            # 온도
            '도': '℃', 'c': '℃', 'celcius': '℃', '섭씨': '℃',
            
            # 농도
            '퍼센트': '%', 'percent': '%', 'pct': '%',
            '밀리그램': 'mg/L', 'mg': 'mg/L', 'mg/l': 'mg/L',
            'ppm': 'ppm', 'ppb': 'ppb',
            
            # 부피/유량
            '리터': 'L', 'liter': 'L', 'l': 'L',
            '미터': 'm', 'meter': 'm',
            'm³': 'm³', 'm3': 'm³', 'cubic': 'm³',
            'm³/day': 'm³/day', 'm³/h': 'm³/h', 'm³/d': 'm³/day',
            'L/min': 'L/min', 'L/s': 'L/s',
            
            # 압력
            '바': 'bar', 'bar': 'bar',
            '파스칼': 'Pa', 'pa': 'Pa', 'kpa': 'kPa',
            '기압': 'atm', 'atm': 'atm',
            
            # 전력
            '킬로와트': 'kW', 'kw': 'kW', 'kwatt': 'kW',
            '와트': 'W', 'watt': 'W', 'w': 'W',
            '메가와트': 'MW', 'mw': 'MW',
            
            # 탄소
            'kgco2e': 'kgCO2e', 'kgco2': 'kgCO2e',
            'co2': 'CO2', '탄소': 'CO2',
            
            # 수질
            'ntu': 'NTU', 'tu': 'TU', 'ss': 'SS',
            'ph': 'pH', '산성도': 'pH',
            'μs/cm': 'μS/cm', 'us/cm': 'μS/cm',
            
            # 시간
            '시간': 'h', 'hour': 'h', 'hr': 'h',
            '분': 'min', 'minute': 'min',
            '초': 's', 'second': 's', 'sec': 's',
            
            # 기타
            'rpm': 'rpm', '회전': 'rpm',
            'm/s': 'm/s', '속도': 'm/s',
        }
        
        return unit_mapping.get(unit, unit)
    
    def _calculate_confidence(self, number: float, unit: str) -> float:
        """숫자-단위 쌍의 신뢰도 계산"""
        confidence = 0.8  # 기본 신뢰도
        
        # 단위가 있으면 신뢰도 증가
        if unit:
            confidence += 0.2
        
        # 정수장 도메인에 적합한 범위인지 확인
        if self._is_domain_appropriate(number, unit):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _is_domain_appropriate(self, number: float, unit: str) -> bool:
        """정수장 도메인에 적합한 수치인지 확인"""
        # 온도 범위 (0-50℃)
        if unit in ['℃', '도'] and 0 <= number <= 50:
            return True
        
        # pH 범위 (6-9)
        if unit == 'pH' and 6 <= number <= 9:
            return True
        
        # 탁도 범위 (0-100 NTU)
        if unit in ['NTU', 'TU'] and 0 <= number <= 100:
            return True
        
        # 전력 범위 (0-1000 kW)
        if unit in ['kW', 'W'] and 0 <= number <= 1000:
            return True
        
        return False
    
    def validate_numeric_answer(self, answer: str, expected_numbers: List[float]) -> Dict[str, Any]:
        """
        답변의 숫자 정확도 검증
        
        Args:
            answer: 생성된 답변
            expected_numbers: 예상 숫자 리스트
            
        Returns:
            검증 결과
        """
        extracted = self.extract_numbers_with_units(answer)
        answer_numbers = [item['number'] for item in extracted]
        
        # 숫자 매칭 계산
        matched_count = 0
        for expected in expected_numbers:
            for answer_num in answer_numbers:
                if abs(expected - answer_num) < 0.01:  # 0.01 오차 허용
                    matched_count += 1
                    break
        
        accuracy = matched_count / len(expected_numbers) if expected_numbers else 0
        
        return {
            'accuracy': accuracy,
            'matched_count': matched_count,
            'total_expected': len(expected_numbers),
            'extracted_numbers': answer_numbers,
            'confidence': sum(item['confidence'] for item in extracted) / len(extracted) if extracted else 0
        }
    
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

