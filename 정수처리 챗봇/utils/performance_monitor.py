"""
성능 모니터링 모듈

이 모듈은 챗봇 시스템의 각 단계별 성능을 측정하고 분석하는 기능을 제공합니다.
"""

import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    question_analysis_time: float = 0.0
    vector_search_time: float = 0.0
    answer_generation_time: float = 0.0
    sql_processing_time: float = 0.0
    total_processing_time: float = 0.0
    classification_result: Dict = field(default_factory=dict)
    pipeline_type: str = ""
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = None
        self.current_metrics = None
    
    def start_timer(self):
        """타이머 시작"""
        self.start_time = time.time()
        self.current_metrics = PerformanceMetrics()
    
    def record_metric(self, metric_name: str, value: float):
        """메트릭 기록"""
        if self.current_metrics is None:
            self.current_metrics = PerformanceMetrics()
        
        setattr(self.current_metrics, metric_name, value)
    
    def end_timer(self) -> float:
        """타이머 종료 및 총 시간 반환"""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.record_metric('total_processing_time', total_time)
            return total_time
        return 0.0
    
    def save_metrics(self, metrics: PerformanceMetrics = None):
        """메트릭 저장"""
        if metrics is None:
            metrics = self.current_metrics
        
        if metrics:
            self.metrics_history.append(metrics)
            
            # 성능 로그 출력
            logger.info(f"=== 성능 메트릭 ===")
            logger.info(f"질문 분석 시간: {metrics.question_analysis_time:.3f}초")
            logger.info(f"벡터 검색 시간: {metrics.vector_search_time:.3f}초")
            logger.info(f"답변 생성 시간: {metrics.answer_generation_time:.3f}초")
            if metrics.sql_processing_time > 0:
                logger.info(f"SQL 처리 시간: {metrics.sql_processing_time:.3f}초")
            logger.info(f"총 처리 시간: {metrics.total_processing_time:.3f}초")
            logger.info(f"파이프라인 타입: {metrics.pipeline_type}")
            logger.info(f"분류 결과: {metrics.classification_result}")
            logger.info(f"신뢰도 점수: {metrics.confidence_score:.3f}")
            logger.info(f"==================")
    
    def get_average_metrics(self) -> Dict[str, float]:
        """평균 메트릭 계산"""
        if not self.metrics_history:
            return {}
        
        avg_metrics = {}
        for field_name in PerformanceMetrics.__dataclass_fields__:
            if field_name in ['timestamp', 'classification_result']:
                continue
                
            values = [getattr(metric, field_name) for metric in self.metrics_history 
                     if getattr(metric, field_name, 0) > 0]
            if values:
                avg_metrics[field_name] = sum(values) / len(values)
        
        return avg_metrics
    
    def print_performance_summary(self):
        """성능 요약 출력"""
        avg_metrics = self.get_average_metrics()
        if not avg_metrics:
            return
        
        print("\n" + "="*60)
        print("성능 요약")
        print("="*60)
        print(f"총 질문 수: {len(self.metrics_history)}")
        print(f"평균 질문 분석 시간: {avg_metrics.get('question_analysis_time', 0):.3f}초")
        print(f"평균 벡터 검색 시간: {avg_metrics.get('vector_search_time', 0):.3f}초")
        print(f"평균 답변 생성 시간: {avg_metrics.get('answer_generation_time', 0):.3f}초")
        print(f"평균 SQL 처리 시간: {avg_metrics.get('sql_processing_time', 0):.3f}초")
        print(f"평균 총 처리 시간: {avg_metrics.get('total_processing_time', 0):.3f}초")
        print("="*60)
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """성능 트렌드 분석"""
        if len(self.metrics_history) < 2:
            return {}
        
        trends = {}
        for field_name in ['total_processing_time', 'question_analysis_time', 
                          'vector_search_time', 'answer_generation_time']:
            values = [getattr(metric, field_name) for metric in self.metrics_history]
            trends[field_name] = values
        
        return trends
    
    def export_metrics(self, filename: str = None):
        """메트릭을 파일로 내보내기"""
        if filename is None:
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("성능 메트릭 내보내기\n")
            f.write("="*50 + "\n")
            
            for i, metric in enumerate(self.metrics_history, 1):
                f.write(f"\n질문 {i}:\n")
                f.write(f"  질문 분석 시간: {metric.question_analysis_time:.3f}초\n")
                f.write(f"  벡터 검색 시간: {metric.vector_search_time:.3f}초\n")
                f.write(f"  답변 생성 시간: {metric.answer_generation_time:.3f}초\n")
                f.write(f"  SQL 처리 시간: {metric.sql_processing_time:.3f}초\n")
                f.write(f"  총 처리 시간: {metric.total_processing_time:.3f}초\n")
                f.write(f"  파이프라인 타입: {metric.pipeline_type}\n")
                f.write(f"  신뢰도 점수: {metric.confidence_score:.3f}\n")
                f.write(f"  타임스탬프: {metric.timestamp}\n")
            
            # 평균 메트릭 추가
            avg_metrics = self.get_average_metrics()
            f.write(f"\n평균 메트릭:\n")
            f.write("="*50 + "\n")
            for key, value in avg_metrics.items():
                f.write(f"  {key}: {value:.3f}초\n")
        
        logger.info(f"성능 메트릭이 {filename}에 저장되었습니다.")
