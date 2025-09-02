"""
경량화된 교통정보 해석 핸들러

교통정보 해석 요청을 간단하게 처리하고 핵심 분석 결과만 제공
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# 상대 경로 import 수정
from core.database.sql_slot_extractor import SQLSlotExtractor, SlotExtractionResult

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """분석 유형 (간소화)"""
    TRAFFIC_VOLUME = "traffic_volume"
    ACCIDENT_PATTERN = "accident_pattern"
    INTERSECTION_ANALYSIS = "intersection_analysis"
    GENERAL_ANALYSIS = "general_analysis"

@dataclass
class AnalysisRequest:
    """경량화된 분석 요청 데이터"""
    question: str
    analysis_type: AnalysisType
    time_range: Optional[Dict] = None
    location: Optional[str] = None

@dataclass
class AnalysisResult:
    """경량화된 분석 결과"""
    summary: str
    key_insights: List[str]

class TrafficAnalysisHandler:
    """경량화된 교통정보 해석 핸들러"""
    
    def __init__(self, sql_generator=None, answer_generator=None):
        """초기화"""
        self.sql_generator = sql_generator
        self.answer_generator = answer_generator
        self.sql_slot_extractor = SQLSlotExtractor()
        
        # 간소화된 분석 유형별 키워드
        self.analysis_keywords = {
            AnalysisType.TRAFFIC_VOLUME: ['교통량', '통행량', '차량'],
            AnalysisType.ACCIDENT_PATTERN: ['사고', '교통사고', '위험'],
            AnalysisType.INTERSECTION_ANALYSIS: ['교차로', '신호등', '정체'],
            AnalysisType.GENERAL_ANALYSIS: ['교통', '정보', '분석']
        }
        
        logger.info("경량화된 교통정보 해석 핸들러 초기화 완료")
    
    def analyze_request(self, question: str, context: Optional[Dict] = None) -> AnalysisRequest:
        """간소화된 분석 요청 파싱"""
        question_lower = question.lower()
        
        # 분석 유형 결정
        analysis_type = self._determine_analysis_type(question_lower)
        
        # 슬롯 추출 (간소화)
        slots = self.sql_slot_extractor.extract_slots(question)
        
        return AnalysisRequest(
            question=question,
            analysis_type=analysis_type,
            time_range=slots.get('time_range'),
            location=slots.get('location')
        )
    
    def _determine_analysis_type(self, question: str) -> AnalysisType:
        """간소화된 분석 유형 결정"""
        for analysis_type, keywords in self.analysis_keywords.items():
            if any(keyword in question for keyword in keywords):
                return analysis_type
        
        return AnalysisType.GENERAL_ANALYSIS
    
    def generate_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """경량화된 분석 결과 생성"""
        try:
            # 기본 분석 로직
            if request.analysis_type == AnalysisType.TRAFFIC_VOLUME:
                summary = "교통량 관련 데이터를 분석할 수 있습니다."
                insights = ["교통량 통계 조회 가능", "시간대별 패턴 분석 가능"]
            elif request.analysis_type == AnalysisType.ACCIDENT_PATTERN:
                summary = "교통사고 패턴을 분석할 수 있습니다."
                insights = ["사고 발생 현황 조회", "위험 지역 식별 가능"]
            elif request.analysis_type == AnalysisType.INTERSECTION_ANALYSIS:
                summary = "교차로별 교통 상황을 분석할 수 있습니다."
                insights = ["신호등 효율성 분석", "정체 구간 파악 가능"]
            else:
                summary = "일반적인 교통 정보를 제공할 수 있습니다."
                insights = ["기본 교통 통계", "지역별 비교 분석"]
            
            return AnalysisResult(
                summary=summary,
                key_insights=insights
            )
            
        except Exception as e:
            logger.error(f"분석 생성 실패: {e}")
            return AnalysisResult(
                summary="분석 중 오류가 발생했습니다.",
                key_insights=["기본 정보만 제공 가능"]
            )
