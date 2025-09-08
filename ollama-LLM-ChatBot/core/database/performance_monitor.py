"""
SQL 생성 성능 모니터링 시스템

SQL 생성 과정의 성능을 실시간으로 모니터링하고 분석하는 모듈
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """메트릭 타입"""
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    METHOD_USAGE = "method_usage"
    CONFIDENCE_SCORE = "confidence_score"

@dataclass
class PerformanceMetric:
    """성능 메트릭"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SQLGenerationLog:
    """SQL 생성 로그"""
    timestamp: datetime
    question: str
    generated_sql: str
    method: str
    execution_time: float
    confidence: float
    success: bool
    error_message: Optional[str] = None
    entity_count: int = 0
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceStats:
    """성능 통계"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    avg_confidence: float = 0.0
    cache_hit_count: int = 0
    method_counts: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)
    hourly_stats: Dict[str, int] = field(default_factory=dict)

class PerformanceMonitor:
    """
    성능 모니터링 시스템
    
    기능:
    1. 실시간 성능 메트릭 수집
    2. SQL 생성 로그 기록
    3. 성능 통계 분석
    4. 성능 저하 감지 및 알림
    5. 대시보드용 데이터 제공
    """
    
    def __init__(self, 
                 log_file: str = "logs/sql_performance.jsonl",
                 metrics_file: str = "logs/sql_metrics.jsonl",
                 max_logs: int = 10000,
                 alert_threshold: Dict[str, float] = None):
        """성능 모니터링 시스템 초기화"""
        
        # 파일 경로 설정
        self.log_file = Path(log_file)
        self.metrics_file = Path(metrics_file)
        self.log_file.parent.mkdir(exist_ok=True)
        self.metrics_file.parent.mkdir(exist_ok=True)
        
        # 메모리 저장소 (최근 데이터)
        self.recent_logs = deque(maxlen=max_logs)
        self.recent_metrics = deque(maxlen=max_logs)
        
        # 실시간 통계
        self.current_stats = PerformanceStats()
        self.hourly_data = defaultdict(list)
        
        # 알림 임계값
        self.alert_threshold = alert_threshold or {
            'avg_response_time': 5.0,  # 5초
            'error_rate': 0.1,         # 10%
            'confidence': 0.7          # 70%
        }
        
        # 성능 추적을 위한 윈도우 (최근 100개 요청)
        self.performance_window = deque(maxlen=100)
        
        # 스레드 안전성
        self._lock = threading.Lock()
        
        logger.info("SQL 성능 모니터링 시스템 초기화 완료")
    
    def log_sql_generation(self, 
                          question: str,
                          generated_sql: str,
                          method: str,
                          execution_time: float,
                          confidence: float,
                          success: bool,
                          error_message: Optional[str] = None,
                          entity_count: int = 0,
                          cache_hit: bool = False,
                          metadata: Dict[str, Any] = None) -> str:
        """
        SQL 생성 로그 기록
        
        Returns:
            로그 ID
        """
        with self._lock:
            log_entry = SQLGenerationLog(
                timestamp=datetime.now(),
                question=question,
                generated_sql=generated_sql,
                method=method,
                execution_time=execution_time,
                confidence=confidence,
                success=success,
                error_message=error_message,
                entity_count=entity_count,
                cache_hit=cache_hit,
                metadata=metadata or {}
            )
            
            # 메모리에 저장
            self.recent_logs.append(log_entry)
            self.performance_window.append(log_entry)
            
            # 통계 업데이트
            self._update_stats(log_entry)
            
            # 파일에 저장
            self._save_log_to_file(log_entry)
            
            # 성능 메트릭 기록
            self._record_metrics(log_entry)
            
            # 알림 확인
            self._check_alerts()
            
            log_id = f"{log_entry.timestamp.isoformat()}_{hash(question) % 10000}"
            logger.debug(f"SQL 생성 로그 기록: {log_id}")
            
            return log_id
    
    def record_metric(self, metric_type: MetricType, value: float, metadata: Dict[str, Any] = None):
        """메트릭 기록"""
        with self._lock:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=metric_type,
                value=value,
                metadata=metadata or {}
            )
            
            self.recent_metrics.append(metric)
            self._save_metric_to_file(metric)
    
    def get_current_stats(self) -> PerformanceStats:
        """현재 통계 조회"""
        with self._lock:
            return self.current_stats
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """성능 요약 정보 조회"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # 최근 데이터 필터링
            recent_logs = [log for log in self.recent_logs if log.timestamp >= cutoff_time]
            
            if not recent_logs:
                return {
                    'period_hours': hours,
                    'total_requests': 0,
                    'success_rate': 0.0,
                    'avg_response_time': 0.0,
                    'avg_confidence': 0.0,
                    'method_distribution': {},
                    'error_distribution': {},
                    'performance_trend': []
                }
            
            # 통계 계산
            total_requests = len(recent_logs)
            successful_requests = sum(1 for log in recent_logs if log.success)
            success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
            
            response_times = [log.execution_time for log in recent_logs]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            confidences = [log.confidence for log in recent_logs if log.success]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # 방법별 분포
            method_counts = defaultdict(int)
            for log in recent_logs:
                method_counts[log.method] += 1
            
            method_distribution = {
                method: count / total_requests 
                for method, count in method_counts.items()
            }
            
            # 오류 분포
            error_counts = defaultdict(int)
            for log in recent_logs:
                if not log.success and log.error_message:
                    error_type = log.error_message.split(':')[0] if ':' in log.error_message else log.error_message
                    error_counts[error_type] += 1
            
            error_distribution = dict(error_counts)
            
            # 성능 추세 (시간당)
            performance_trend = self._calculate_performance_trend(recent_logs)
            
            return {
                'period_hours': hours,
                'total_requests': total_requests,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'avg_confidence': avg_confidence,
                'method_distribution': method_distribution,
                'error_distribution': error_distribution,
                'performance_trend': performance_trend,
                'cache_hit_rate': sum(1 for log in recent_logs if log.cache_hit) / total_requests if total_requests > 0 else 0.0
            }
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """실시간 메트릭 조회 (최근 100개 요청 기준)"""
        with self._lock:
            if not self.performance_window:
                return {
                    'current_qps': 0.0,
                    'avg_response_time': 0.0,
                    'success_rate': 0.0,
                    'avg_confidence': 0.0,
                    'active_methods': [],
                    'recent_errors': []
                }
            
            # 최근 1분간 QPS 계산
            one_minute_ago = datetime.now() - timedelta(minutes=1)
            recent_requests = [log for log in self.performance_window if log.timestamp >= one_minute_ago]
            current_qps = len(recent_requests) / 60.0
            
            # 평균 응답 시간
            response_times = [log.execution_time for log in self.performance_window]
            avg_response_time = sum(response_times) / len(response_times)
            
            # 성공률
            successful = sum(1 for log in self.performance_window if log.success)
            success_rate = successful / len(self.performance_window)
            
            # 평균 신뢰도
            confidences = [log.confidence for log in self.performance_window if log.success]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # 활성 방법들
            active_methods = list(set(log.method for log in self.performance_window))
            
            # 최근 오류들
            recent_errors = [
                {
                    'timestamp': log.timestamp.isoformat(),
                    'error': log.error_message,
                    'method': log.method
                }
                for log in list(self.performance_window)[-10:]  # 최근 10개
                if not log.success and log.error_message
            ]
            
            return {
                'current_qps': current_qps,
                'avg_response_time': avg_response_time,
                'success_rate': success_rate,
                'avg_confidence': avg_confidence,
                'active_methods': active_methods,
                'recent_errors': recent_errors
            }
    
    def get_method_comparison(self) -> Dict[str, Dict[str, float]]:
        """방법별 성능 비교"""
        with self._lock:
            method_stats = defaultdict(lambda: {
                'count': 0,
                'success_count': 0,
                'total_time': 0.0,
                'total_confidence': 0.0,
                'cache_hits': 0
            })
            
            for log in self.performance_window:
                stats = method_stats[log.method]
                stats['count'] += 1
                if log.success:
                    stats['success_count'] += 1
                    stats['total_confidence'] += log.confidence
                stats['total_time'] += log.execution_time
                if log.cache_hit:
                    stats['cache_hits'] += 1
            
            comparison = {}
            for method, stats in method_stats.items():
                if stats['count'] > 0:
                    comparison[method] = {
                        'success_rate': stats['success_count'] / stats['count'],
                        'avg_response_time': stats['total_time'] / stats['count'],
                        'avg_confidence': stats['total_confidence'] / stats['success_count'] if stats['success_count'] > 0 else 0.0,
                        'cache_hit_rate': stats['cache_hits'] / stats['count'],
                        'usage_count': stats['count']
                    }
            
            return comparison
    
    def _update_stats(self, log_entry: SQLGenerationLog):
        """통계 업데이트"""
        self.current_stats.total_requests += 1
        
        if log_entry.success:
            self.current_stats.successful_requests += 1
        else:
            self.current_stats.failed_requests += 1
            if log_entry.error_message:
                error_type = log_entry.error_message.split(':')[0]
                self.current_stats.error_types[error_type] = self.current_stats.error_types.get(error_type, 0) + 1
        
        if log_entry.cache_hit:
            self.current_stats.cache_hit_count += 1
        
        # 방법별 카운트
        self.current_stats.method_counts[log_entry.method] = self.current_stats.method_counts.get(log_entry.method, 0) + 1
        
        # 평균 계산 (이동 평균)
        if self.current_stats.total_requests > 0:
            # 응답 시간 이동 평균
            alpha = 0.1  # 평활화 계수
            if self.current_stats.avg_response_time == 0:
                self.current_stats.avg_response_time = log_entry.execution_time
            else:
                self.current_stats.avg_response_time = (1 - alpha) * self.current_stats.avg_response_time + alpha * log_entry.execution_time
            
            # 신뢰도 이동 평균 (성공한 경우만)
            if log_entry.success:
                if self.current_stats.avg_confidence == 0:
                    self.current_stats.avg_confidence = log_entry.confidence
                else:
                    self.current_stats.avg_confidence = (1 - alpha) * self.current_stats.avg_confidence + alpha * log_entry.confidence
        
        # 시간별 통계
        hour_key = log_entry.timestamp.strftime('%Y-%m-%d-%H')
        self.current_stats.hourly_stats[hour_key] = self.current_stats.hourly_stats.get(hour_key, 0) + 1
    
    def _record_metrics(self, log_entry: SQLGenerationLog):
        """메트릭 기록"""
        # 응답 시간 메트릭
        self.record_metric(
            MetricType.RESPONSE_TIME,
            log_entry.execution_time,
            {'method': log_entry.method}
        )
        
        # 신뢰도 메트릭 (성공한 경우만)
        if log_entry.success:
            self.record_metric(
                MetricType.CONFIDENCE_SCORE,
                log_entry.confidence,
                {'method': log_entry.method}
            )
    
    def _save_log_to_file(self, log_entry: SQLGenerationLog):
        """로그를 파일에 저장"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                log_data = asdict(log_entry)
                log_data['timestamp'] = log_entry.timestamp.isoformat()
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"로그 파일 저장 실패: {e}")
    
    def _save_metric_to_file(self, metric: PerformanceMetric):
        """메트릭을 파일에 저장"""
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                metric_data = asdict(metric)
                metric_data['timestamp'] = metric.timestamp.isoformat()
                metric_data['metric_type'] = metric.metric_type.value
                f.write(json.dumps(metric_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"메트릭 파일 저장 실패: {e}")
    
    def _calculate_performance_trend(self, logs: List[SQLGenerationLog]) -> List[Dict[str, Any]]:
        """성능 추세 계산"""
        # 시간별 그룹핑
        hourly_data = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'total_time': 0.0,
            'total_confidence': 0.0
        })
        
        for log in logs:
            hour_key = log.timestamp.strftime('%Y-%m-%d %H:00')
            data = hourly_data[hour_key]
            data['requests'] += 1
            if log.success:
                data['successes'] += 1
                data['total_confidence'] += log.confidence
            data['total_time'] += log.execution_time
        
        # 시간별 통계 계산
        trend = []
        for hour_key in sorted(hourly_data.keys()):
            data = hourly_data[hour_key]
            trend.append({
                'hour': hour_key,
                'requests': data['requests'],
                'success_rate': data['successes'] / data['requests'] if data['requests'] > 0 else 0.0,
                'avg_response_time': data['total_time'] / data['requests'] if data['requests'] > 0 else 0.0,
                'avg_confidence': data['total_confidence'] / data['successes'] if data['successes'] > 0 else 0.0
            })
        
        return trend
    
    def _check_alerts(self):
        """알림 확인"""
        if len(self.performance_window) < 10:  # 최소 10개 데이터 필요
            return
        
        # 최근 성능 계산
        recent_logs = list(self.performance_window)[-10:]  # 최근 10개
        
        # 평균 응답 시간 확인
        avg_response_time = sum(log.execution_time for log in recent_logs) / len(recent_logs)
        if avg_response_time > self.alert_threshold['avg_response_time']:
            logger.warning(f"응답 시간 임계값 초과: {avg_response_time:.2f}초 > {self.alert_threshold['avg_response_time']}초")
        
        # 오류율 확인
        error_count = sum(1 for log in recent_logs if not log.success)
        error_rate = error_count / len(recent_logs)
        if error_rate > self.alert_threshold['error_rate']:
            logger.warning(f"오류율 임계값 초과: {error_rate:.2%} > {self.alert_threshold['error_rate']:.2%}")
        
        # 신뢰도 확인
        successful_logs = [log for log in recent_logs if log.success]
        if successful_logs:
            avg_confidence = sum(log.confidence for log in successful_logs) / len(successful_logs)
            if avg_confidence < self.alert_threshold['confidence']:
                logger.warning(f"신뢰도 임계값 미달: {avg_confidence:.2%} < {self.alert_threshold['confidence']:.2%}")
    
    def export_performance_report(self, output_file: str, hours: int = 24) -> str:
        """성능 보고서 내보내기"""
        summary = self.get_performance_summary(hours)
        real_time = self.get_real_time_metrics()
        method_comparison = self.get_method_comparison()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_hours': hours,
            'summary': summary,
            'real_time_metrics': real_time,
            'method_comparison': method_comparison,
            'current_stats': asdict(self.current_stats)
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"성능 보고서 내보내기 완료: {output_path}")
        return str(output_path)
    
    def reset_stats(self):
        """통계 초기화"""
        with self._lock:
            self.current_stats = PerformanceStats()
            self.recent_logs.clear()
            self.recent_metrics.clear()
            self.performance_window.clear()
            self.hourly_data.clear()
        
        logger.info("성능 통계 초기화 완료")

# 전역 모니터 인스턴스 (싱글톤)
_global_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """전역 성능 모니터 인스턴스 반환"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

if __name__ == "__main__":
    # 테스트 코드
    monitor = PerformanceMonitor()
    
    # 테스트 로그 생성
    test_logs = [
        ("조치원읍 교차로 몇 개?", "SELECT COUNT(*) FROM traffic_intersection WHERE name LIKE '%조치원읍%'", "rule_based", 0.1, 0.9, True),
        ("한솔동 교통량 평균", "SELECT AVG(volume) FROM traffic_trafficvolume WHERE intersection_id IN (...)", "rule_based", 0.2, 0.85, True),
        ("복잡한 질문", "-- SQL 생성 실패", "llm_based", 2.5, 0.0, False, "모든 방법 실패"),
        ("상위 10개 지역", "SELECT * FROM traffic_intersection ORDER BY volume DESC LIMIT 10", "hybrid", 1.2, 0.8, True),
    ]
    
    for question, sql, method, time, confidence, success, *error in test_logs:
        error_msg = error[0] if error else None
        monitor.log_sql_generation(
            question=question,
            generated_sql=sql,
            method=method,
            execution_time=time,
            confidence=confidence,
            success=success,
            error_message=error_msg,
            entity_count=2,
            cache_hit=(method == "rule_based")
        )
    
    # 통계 출력
    stats = monitor.get_current_stats()
    print(f"총 요청: {stats.total_requests}")
    print(f"성공 요청: {stats.successful_requests}")
    print(f"평균 응답 시간: {stats.avg_response_time:.3f}초")
    print(f"평균 신뢰도: {stats.avg_confidence:.3f}")
    print(f"방법별 사용: {stats.method_counts}")
    
    # 실시간 메트릭
    real_time = monitor.get_real_time_metrics()
    print(f"\n실시간 QPS: {real_time['current_qps']:.2f}")
    print(f"실시간 성공률: {real_time['success_rate']:.2%}")
    
    # 방법별 비교
    comparison = monitor.get_method_comparison()
    print(f"\n방법별 성능 비교:")
    for method, metrics in comparison.items():
        print(f"  {method}: 성공률 {metrics['success_rate']:.2%}, 응답시간 {metrics['avg_response_time']:.3f}초")
    
    # 보고서 내보내기
    report_file = monitor.export_performance_report("reports/sql_performance_report.json")
    print(f"\n보고서 생성: {report_file}")
