"""
경량화된 데이터 분석 결과 해석 모듈

SQL 쿼리 결과를 간단하게 분석하여 핵심 인사이트만 제공
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """경량화된 분석 결과"""
    total_records: int
    summary: str
    key_metrics: Dict[str, Any]

class DataAnalysisGenerator:
    """경량화된 데이터 분석 결과 해석기"""
    
    def __init__(self):
        """초기화"""
        self.insight_patterns = {
            'high_traffic': lambda avg: avg > 1000,
            'low_traffic': lambda avg: avg < 500,
            'peak_day': lambda max_val, avg: max_val > avg * 1.5
        }
        
        logger.info("경량화된 데이터 분석 생성기 초기화 완료")
    
    def analyze_sql_results(self, 
                          sql_results: List[Dict], 
                          query_info: Dict,
                          time_range: Dict) -> AnalysisResult:
        """경량화된 SQL 결과 분석"""
        logger.info(f"SQL 결과 분석 시작: {len(sql_results)}개 레코드")
        
        if not sql_results:
            return self._create_empty_result(time_range, query_info)
        
        # 핵심 지표만 계산
        metrics = self._calculate_key_metrics(sql_results)
        
        # 간단한 요약 생성
        summary = self._generate_simple_summary(metrics, time_range, query_info)
        
        return AnalysisResult(
            total_records=len(sql_results),
            summary=summary,
            key_metrics=metrics
        )
    
    def _calculate_key_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """핵심 지표만 계산"""
        if not results:
            return {}
        
        # 첫 번째 결과에서 컬럼 확인
        first_result = results[0]
        
        metrics = {
            'total_records': len(results)
        }
        
        # 교통량 관련 지표 (존재하는 경우만)
        for col in ['traffic_volume', 'vehicle_count', 'total_traffic_volume']:
            if col in first_result:
                values = [r.get(col, 0) for r in results if r.get(col) is not None]
                if values:
                    metrics[f'avg_{col}'] = sum(values) / len(values)
                    metrics[f'max_{col}'] = max(values)
        
        return metrics
    
    def _generate_simple_summary(self, metrics: Dict[str, Any], time_range: Dict, query_info: Dict) -> str:
        """간단한 요약 생성"""
        if not metrics:
            return "분석할 데이터가 없습니다."
        
        summary_parts = []
        
        # 기본 정보
        summary_parts.append(f"총 {metrics.get('total_records', 0)}개 레코드를 분석했습니다.")
        
        # 시간 범위
        if 'description' in time_range:
            summary_parts.append(f"분석 기간: {time_range['description']}")
        
        # 핵심 지표
        for key, value in metrics.items():
            if key.startswith('avg_') and isinstance(value, (int, float)):
                summary_parts.append(f"평균 {key[4:]}: {value:.1f}")
        
        return " ".join(summary_parts)
    
    def _create_empty_result(self, time_range: Dict, query_info: Dict) -> AnalysisResult:
        """빈 결과 생성"""
        return AnalysisResult(
            total_records=0,
            summary="해당 기간에 데이터가 없습니다.",
            key_metrics={}
        )
