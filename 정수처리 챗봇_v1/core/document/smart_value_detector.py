"""
스마트 값 감지기 - 학습 기반 접근법

하드코딩 대신 학습과 설정을 통한 유연한 값 감지
"""

import re
import json
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

@dataclass
class ValuePattern:
    """값 패턴 정의"""
    name: str
    patterns: List[str]
    weight: float
    context_keywords: List[str]
    value_type: str
    examples: List[str]

@dataclass
class LearnedValue:
    """학습된 값 정보"""
    value: str
    pattern_name: str
    confidence: float
    weight: float
    context: str
    examples_count: int

class SmartValueDetector:
    """스마트 값 감지기 - 학습 기반"""
    
    def __init__(self, config_path: str = "config/smart_value_patterns.json"):
        self.config_path = Path(config_path)
        self.patterns: Dict[str, ValuePattern] = {}
        self.learned_values: Dict[str, LearnedValue] = {}
        self.value_frequency = Counter()
        self.context_patterns = defaultdict(list)
        
        # 기본 패턴 로드
        self._load_default_patterns()
        
        # 설정 파일에서 패턴 로드
        self._load_patterns_from_config()
        
        logger.info(f"스마트 값 감지기 초기화 완료: {len(self.patterns)}개 패턴")
    
    def _load_default_patterns(self):
        """기본 패턴 로드"""
        default_patterns = {
            "account_info": ValuePattern(
                name="account_info",
                patterns=[
                    r'아이디\s*[:\s]*([A-Z0-9]+)',
                    r'비밀번호\s*[:\s]*([A-Z0-9]+)',
                    r'([A-Z0-9]+)\s*[이고]\s*비밀번호\s*[는은]\s*([A-Z0-9]+)'
                ],
                weight=3.0,
                context_keywords=["관리자", "계정", "기본", "설정"],
                value_type="configuration",
                examples=["KWATER", "admin", "root"]
            ),
            "urls": ValuePattern(
                name="urls",
                patterns=[
                    r'(https?://[^\s]+)',
                    r'(http://[^\s]+)'
                ],
                weight=2.0,
                context_keywords=["접속", "주소", "URL", "링크"],
                value_type="exact_value",
                examples=["http://example.com", "https://api.example.com"]
            ),
            "numeric_values": ValuePattern(
                name="numeric_values",
                patterns=[
                    r'(\d+(?:\.\d+)?)\s*(분|시간|초|mg/L|㎎/L|m³/h|㎥/h|%|퍼센트)',
                    r'상관계수\s*[는은]\s*(\d+(?:\.\d+)?)'
                ],
                weight=2.5,
                context_keywords=["수치", "값", "계수", "비율"],
                value_type="specification",
                examples=["0.72", "10분", "100mg/L"]
            ),
            "functionality": ValuePattern(
                name="functionality",
                patterns=[
                    r'(펌프|운전|상태|전력|사용량|진단|에너지|소비량|통계|이력|기록)'
                ],
                weight=2.0,
                context_keywords=["제공", "확인", "모니터링", "관리"],
                value_type="specification",
                examples=["펌프 운전 기록", "에너지 소비량", "진단 이력"]
            )
        }
        
        self.patterns.update(default_patterns)
    
    def _load_patterns_from_config(self):
        """설정 파일에서 패턴 로드"""
        if not self.config_path.exists():
            self._create_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            for pattern_name, pattern_data in config.items():
                pattern = ValuePattern(
                    name=pattern_name,
                    patterns=pattern_data.get("patterns", []),
                    weight=pattern_data.get("weight", 1.0),
                    context_keywords=pattern_data.get("context_keywords", []),
                    value_type=pattern_data.get("value_type", "exact_value"),
                    examples=pattern_data.get("examples", [])
                )
                self.patterns[pattern_name] = pattern
                
        except Exception as e:
            logger.error(f"패턴 설정 로드 실패: {e}")
    
    def _create_default_config(self):
        """기본 설정 파일 생성"""
        config = {}
        for pattern_name, pattern in self.patterns.items():
            config[pattern_name] = {
                "patterns": pattern.patterns,
                "weight": pattern.weight,
                "context_keywords": pattern.context_keywords,
                "value_type": pattern.value_type,
                "examples": pattern.examples
            }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"기본 설정 파일 생성: {self.config_path}")
    
    def learn_from_examples(self, examples: List[Dict[str, Any]]):
        """예시로부터 학습"""
        for example in examples:
            text = example.get("text", "")
            expected_values = example.get("expected_values", [])
            
            # 각 패턴으로 값 추출 시도
            for pattern_name, pattern in self.patterns.items():
                for regex_pattern in pattern.patterns:
                    matches = re.finditer(regex_pattern, text, re.IGNORECASE)
                    for match in matches:
                        value = match.group(1) if match.groups() else match.group(0)
                        
                        # 예상 값과 일치하는지 확인
                        if value in expected_values:
                            self._update_learned_value(value, pattern_name, text, match)
    
    def _update_learned_value(self, value: str, pattern_name: str, context: str, match):
        """학습된 값 업데이트"""
        if value not in self.learned_values:
            self.learned_values[value] = LearnedValue(
                value=value,
                pattern_name=pattern_name,
                confidence=0.5,
                weight=self.patterns[pattern_name].weight,
                context=context,
                examples_count=0
            )
        
        # 빈도수 증가
        self.value_frequency[value] += 1
        self.learned_values[value].examples_count += 1
        
        # 신뢰도 업데이트 (빈도수 기반)
        self.learned_values[value].confidence = min(1.0, 
            self.learned_values[value].examples_count * 0.1)
    
    def detect_values(self, text: str) -> List[LearnedValue]:
        """텍스트에서 값 감지"""
        detected_values = []
        
        # 학습된 값들로부터 검색
        for value, learned_value in self.learned_values.items():
            if value in text:
                # 문맥 키워드 확인
                context_score = self._calculate_context_score(text, learned_value.pattern_name)
                if context_score > 0.3:  # 임계값
                    detected_values.append(learned_value)
        
        # 패턴 기반 검색 (학습되지 않은 값들)
        for pattern_name, pattern in self.patterns.items():
            for regex_pattern in pattern.patterns:
                matches = re.finditer(regex_pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)
                    
                    # 이미 학습된 값이 아닌 경우에만 추가
                    if value not in self.learned_values:
                        context_score = self._calculate_context_score(text, pattern_name)
                        if context_score > 0.3:
                            learned_value = LearnedValue(
                                value=value,
                                pattern_name=pattern_name,
                                confidence=0.5,
                                weight=pattern.weight,
                                context=text,
                                examples_count=1
                            )
                            detected_values.append(learned_value)
        
        # 가중치 순으로 정렬
        detected_values.sort(key=lambda x: x.weight * x.confidence, reverse=True)
        return detected_values
    
    def _calculate_context_score(self, text: str, pattern_name: str) -> float:
        """문맥 점수 계산"""
        if pattern_name not in self.patterns:
            return 0.0
        
        pattern = self.patterns[pattern_name]
        context_keywords = pattern.context_keywords
        
        if not context_keywords:
            return 0.5  # 기본 점수
        
        # 키워드가 문맥에 포함된 비율 계산
        text_lower = text.lower()
        matched_keywords = sum(1 for keyword in context_keywords 
                             if keyword.lower() in text_lower)
        
        return matched_keywords / len(context_keywords)
    
    def add_pattern(self, pattern: ValuePattern):
        """새로운 패턴 추가"""
        self.patterns[pattern.name] = pattern
        self._save_patterns_to_config()
        logger.info(f"새로운 패턴 추가: {pattern.name}")
    
    def _save_patterns_to_config(self):
        """패턴을 설정 파일에 저장"""
        config = {}
        for pattern_name, pattern in self.patterns.items():
            config[pattern_name] = {
                "patterns": pattern.patterns,
                "weight": pattern.weight,
                "context_keywords": pattern.context_keywords,
                "value_type": pattern.value_type,
                "examples": pattern.examples
            }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            "total_patterns": len(self.patterns),
            "learned_values": len(self.learned_values),
            "most_frequent_values": dict(self.value_frequency.most_common(10)),
            "pattern_names": list(self.patterns.keys())
        }
