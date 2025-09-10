"""
정의된 값/범위/상수 감지 및 가중치 시스템

문서에서 정의된 값, 범위, 상수들을 감지하고 답변 생성 시 높은 가중치를 부여하는 모듈
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValueType(Enum):
    """값의 유형"""
    EXACT_VALUE = "exact_value"        # 정확한 값 (예: 10분, 100mg/L)
    RANGE = "range"                    # 범위 (예: 10~60분, 0.5~2.0)
    THRESHOLD = "threshold"            # 임계값 (예: 10분 이상, 100mg/L 이하)
    CONFIGURATION = "configuration"    # 설정값 (예: 기본값, 표준값)
    SPECIFICATION = "specification"    # 사양/규격 (예: 성능 지표, 기준값)

@dataclass
class DefinedValue:
    """정의된 값 정보"""
    value: str                         # 원본 값 문자열
    normalized_value: str              # 정규화된 값
    unit: Optional[str]                # 단위
    value_type: ValueType              # 값의 유형
    context: str                       # 주변 문맥
    confidence: float                  # 신뢰도 (0.0~1.0)
    weight: float                      # 가중치 (1.0~3.0)
    page_number: Optional[int] = None  # 페이지 번호
    source: Optional[str] = None       # 파일명/메타데이터 등 위치 정보

class DefinedValueDetector:
    """정의된 값/범위/상수 감지기"""
    
    def __init__(self):
        """초기화"""
        # 수치 패턴 (개선된 버전)
        self.numeric_patterns = {
            # 정확한 값 패턴
            'exact_value': [
                r'(\d+(?:\.\d+)?)\s*(분|시간|초|mg/L|㎎/L|m³/h|㎥/h|%|퍼센트)',
                r'(\d+(?:\.\d+)?)\s*(이상|이하|초과|미만|이내|범위)',
                r'(http[s]?://[^\s]+)',  # URL 패턴 추가
                r'([A-Z0-9]+/[A-Z0-9]+)',  # 아이디/비밀번호 패턴
            ],
            # 범위 패턴
            'range': [
                r'(\d+(?:\.\d+)?)\s*(~|\-|–|to)\s*(\d+(?:\.\d+)?)\s*(분|시간|초|mg/L|㎎/L|m³/h|㎥/h|%|퍼센트)',
                r'(\d+(?:\.\d+)?)\s*(부터|에서)\s*(\d+(?:\.\d+)?)\s*(까지|사이)',
            ],
            # 임계값 패턴
            'threshold': [
                r'(\d+(?:\.\d+)?)\s*(이상|이하|초과|미만|이내)',
                r'(\d+(?:\.\d+)?)\s*(분|시간|초|mg/L|㎎/L|m³/h|㎥/h|%|퍼센트)\s*(이상|이하|초과|미만|이내)',
            ],
            # 설정값 패턴
            'configuration': [
                r'(기본값|표준값|설정값|권장값|최적값)\s*[:\s]*(\d+(?:\.\d+)?)\s*(분|시간|초|mg/L|㎎/L|m³/h|㎥/h|%|퍼센트)',
                r'(기본|표준|설정|권장|최적)\s*[:\s]*(\d+(?:\.\d+)?)\s*(분|시간|초|mg/L|㎎/L|m³/h|㎥/h|%|퍼센트)',
                r'(아이디|비밀번호|계정)\s*[:\s]*([A-Z0-9]+)',  # 계정 정보 패턴
                r'(아이디|비밀번호)\s*[는은]\s*([A-Z0-9]+)',  # 한국어 계정 정보 패턴
                r'([A-Z0-9]+)\s*[이고]\s*비밀번호\s*[는은]\s*([A-Z0-9]+)',  # 아이디/비밀번호 패턴
            ],
            # 사양/규격 패턴
            'specification': [
                r'(성능|효율|정확도|정밀도|오차|편차)\s*[:\s]*(\d+(?:\.\d+)?)\s*(%|퍼센트|mg/L|㎎/L)',
                r'(상관계수|MAE|MSE|RMSE|R²|R2)\s*[:\s]*(\d+(?:\.\d+)?)',
                r'상관계수\s*[는은]\s*(\d+(?:\.\d+)?)',  # 한국어 상관계수 패턴
                r'(XGB|Extreme Gradient Boosting)',  # 모델명 패턴
                r'(펌프|운전|상태|전력|사용량|진단|에너지|소비량|통계|이력|기록)',  # 기능/서비스 패턴
            ]
        }
        
        # 높은 가중치를 받을 키워드들
        self.high_weight_keywords = {
            '정확한', '정확히', '정확', '명시된', '명시', '기록된', '기록',
            '기본값', '표준값', '설정값', '권장값', '최적값', '기본', '표준', '설정', '권장', '최적',
            '상관계수', '성능', '효율', '정확도', '정밀도', '오차', '편차',
            'MAE', 'MSE', 'RMSE', 'R²', 'R2', '임계값', '기준값',
            '통계', '이력', '기록', '데이터', '정보', '제공', '확인', '상세',
            '펌프', '운전', '상태', '전력', '사용량', '진단', '에너지', '소비량'
        }
        
        # 중간 가중치를 받을 키워드들
        self.medium_weight_keywords = {
            '범위', '사이', '이상', '이하', '초과', '미만', '이내',
            '약', '대략', '평균', '평균적', '일반적', '보통'
        }
        
        # 단위 정규화 매핑
        self.unit_normalization = {
            'min': '분', '분간': '분', 'minute': '분', 'minutes': '분',
            'hr': '시간', 'hour': '시간', 'hours': '시간',
            '퍼센트': '%', 'percent': '%',
            'mg/l': 'mg/L', '㎎/l': 'mg/L', 'mg·l-1': 'mg/L',
            'm3/h': 'm³/h', '㎥/h': 'm³/h'
        }
        
        logger.info("정의된 값 감지기 초기화 완료")
    
    def detect_defined_values(self, text: str, page_number: Optional[int] = None, source: Optional[str] = None) -> List[DefinedValue]:
        """
        텍스트에서 정의된 값들을 감지
        
        Args:
            text: 분석할 텍스트
            page_number: 페이지 번호
            source: 파일명/메타데이터 등 위치 정보
            
        Returns:
            감지된 정의된 값들의 리스트
        """
        defined_values = []
        
        # 각 패턴 유형별로 감지
        for value_type_str, patterns in self.numeric_patterns.items():
            value_type = ValueType(value_type_str)
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    try:
                        defined_value = self._create_defined_value(
                            match, text, value_type, page_number, source
                        )
                        if defined_value:
                            defined_values.append(defined_value)
                    except Exception as e:
                        logger.debug(f"정의된 값 생성 실패: {e}")
                        continue
        
        # 중복 제거 및 정렬
        defined_values = self._deduplicate_and_sort(defined_values)
        
        logger.debug(f"정의된 값 {len(defined_values)}개 감지 완료")
        return defined_values
    
    def _create_defined_value(self, match: re.Match, text: str, value_type: ValueType, 
                            page_number: Optional[int], source: Optional[str]) -> Optional[DefinedValue]:
        """매치된 결과로부터 DefinedValue 객체 생성"""
        try:
            # 매치된 전체 문자열
            full_match = match.group(0)
            
            # 값과 단위 추출
            value, unit = self._extract_value_and_unit(match, value_type)
            
            # 주변 문맥 추출
            context = self._extract_context(text, match.start(), match.end())
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(full_match, context, value_type)
            
            # 가중치 계산
            weight = self._calculate_weight(full_match, context, value_type, confidence)
            
            # 정규화된 값 생성
            normalized_value = self._normalize_value(value, unit)
            
            return DefinedValue(
                value=full_match,
                normalized_value=normalized_value,
                unit=unit,
                value_type=value_type,
                context=context,
                confidence=confidence,
                weight=weight,
                page_number=page_number,
                source=source
            )
            
        except Exception as e:
            logger.debug(f"DefinedValue 생성 실패: {e}")
            return None
    
    def _extract_value_and_unit(self, match: re.Match, value_type: ValueType) -> Tuple[str, Optional[str]]:
        """매치에서 값과 단위 추출"""
        groups = match.groups()
        
        if value_type == ValueType.EXACT_VALUE:
            if len(groups) >= 2:
                # URL이나 계정 정보인 경우
                if groups[0].startswith('http') or '/' in groups[0]:
                    return groups[0], None
                return groups[0], groups[1]
            elif len(groups) >= 1:
                # 단일 값인 경우
                return groups[0], None
        elif value_type == ValueType.RANGE:
            if len(groups) >= 4:
                return f"{groups[0]}~{groups[2]}", groups[3]
        elif value_type == ValueType.THRESHOLD:
            if len(groups) >= 2:
                return groups[0], groups[1] if len(groups) > 1 else None
        elif value_type == ValueType.CONFIGURATION:
            if len(groups) >= 3:
                return groups[1], groups[2]
            elif len(groups) >= 2:
                # 계정 정보인 경우
                if groups[0] in ['아이디', '비밀번호']:
                    return groups[1], None
                elif groups[0] and groups[1]:  # 아이디/비밀번호 패턴
                    return f"{groups[0]}/{groups[1]}", None
                return groups[1], None
        elif value_type == ValueType.SPECIFICATION:
            if len(groups) >= 3:
                return groups[1], groups[2]
            elif len(groups) >= 1:
                # 모델명인 경우
                return groups[0], None
        
        return match.group(0), None
    
    def _extract_context(self, text: str, start: int, end: int, context_window: int = 50) -> str:
        """매치 주변의 문맥 추출"""
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        return text[context_start:context_end].strip()
    
    def _calculate_confidence(self, value_text: str, context: str, value_type: ValueType) -> float:
        """신뢰도 계산 (0.0~1.0)"""
        confidence = 0.5  # 기본 신뢰도
        
        # 값 유형별 기본 신뢰도
        type_confidence = {
            ValueType.EXACT_VALUE: 0.8,
            ValueType.RANGE: 0.7,
            ValueType.THRESHOLD: 0.6,
            ValueType.CONFIGURATION: 0.9,
            ValueType.SPECIFICATION: 0.85
        }
        confidence = type_confidence.get(value_type, 0.5)
        
        # 문맥에서 신뢰도 증감 요소 확인
        context_lower = context.lower()
        
        # 신뢰도 증가 요소
        if any(keyword in context_lower for keyword in self.high_weight_keywords):
            confidence += 0.2
        elif any(keyword in context_lower for keyword in self.medium_weight_keywords):
            confidence += 0.1
        
        # 신뢰도 감소 요소
        if any(word in context_lower for word in ['추정', '예상', '가능', '아마', '대략']):
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_weight(self, value_text: str, context: str, value_type: ValueType, confidence: float) -> float:
        """가중치 계산 (1.0~3.0)"""
        base_weight = 1.0
        
        # 값 유형별 기본 가중치
        type_weights = {
            ValueType.EXACT_VALUE: 2.0,
            ValueType.RANGE: 1.5,
            ValueType.THRESHOLD: 1.8,
            ValueType.CONFIGURATION: 2.5,
            ValueType.SPECIFICATION: 2.2
        }
        base_weight = type_weights.get(value_type, 1.0)
        
        # 신뢰도에 따른 가중치 조정
        weight = base_weight * confidence
        
        # 문맥에서 가중치 증감 요소 확인
        context_lower = context.lower()
        
        # 가중치 증가 요소
        if any(keyword in context_lower for keyword in self.high_weight_keywords):
            weight += 0.5
        elif any(keyword in context_lower for keyword in self.medium_weight_keywords):
            weight += 0.2
        
        return max(1.0, min(3.0, weight))
    
    def _normalize_value(self, value: str, unit: Optional[str]) -> str:
        """값 정규화"""
        if not value:
            return ""
        
        # 단위 정규화
        if unit:
            normalized_unit = self.unit_normalization.get(unit.lower(), unit)
            return f"{value} {normalized_unit}"
        
        return value
    
    def _deduplicate_and_sort(self, defined_values: List[DefinedValue]) -> List[DefinedValue]:
        """중복 제거 및 정렬"""
        # 중복 제거 (정규화된 값 기준)
        seen = set()
        unique_values = []
        
        for value in defined_values:
            key = (value.normalized_value, value.value_type)
            if key not in seen:
                seen.add(key)
                unique_values.append(value)
        
        # 가중치 순으로 정렬 (높은 가중치가 먼저)
        unique_values.sort(key=lambda x: x.weight, reverse=True)
        
        return unique_values
    
    def get_high_priority_values(self, defined_values: List[DefinedValue], 
                               min_weight: float = 2.0) -> List[DefinedValue]:
        """높은 우선순위의 정의된 값들 반환"""
        return [v for v in defined_values if v.weight >= min_weight]
    
    def extract_values_for_question(self, defined_values: List[DefinedValue], 
                                  question: str) -> List[DefinedValue]:
        """질문과 관련된 정의된 값들 추출"""
        question_lower = question.lower()
        relevant_values = []
        
        for value in defined_values:
            # 질문의 키워드와 값의 문맥이 겹치는지 확인
            context_lower = value.context.lower()
            
            # 질문의 주요 키워드들이 문맥에 포함되어 있는지 확인
            question_keywords = set(re.findall(r'\b\w+\b', question_lower))
            context_keywords = set(re.findall(r'\b\w+\b', context_lower))
            
            # 공통 키워드가 있으면 관련성 있다고 판단
            if question_keywords.intersection(context_keywords):
                relevant_values.append(value)
        
        # 가중치 순으로 정렬
        relevant_values.sort(key=lambda x: x.weight, reverse=True)
        
        return relevant_values
