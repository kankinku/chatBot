"""
법률 재순위화 모듈

크로스엔코더를 사용하여 검색 결과를 재순위화하고,
점수 캘리브레이션을 통해 신뢰도 있는 결과를 제공합니다.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
import torch

from .legal_schema import LegalChunk
from .legal_retriever import SearchResult

import logging
logger = logging.getLogger(__name__)

@dataclass
class RerankConfig:
    """재순위화 설정"""
    model_name: str = "BAAI/bge-reranker-v2-m3"  # 크로스엔코더 모델
    batch_size: int = 16                         # 배치 크기
    max_candidates: int = 80                     # 재순위화할 최대 후보 수
    threshold: float = 0.5                      # 신뢰도 임계값
    calibration_enabled: bool = True            # 점수 캘리브레이션 사용 여부
    temperature: float = 1.0                    # Temperature scaling 파라미터

@dataclass
class RerankResult:
    """재순위화 결과"""
    chunk: LegalChunk
    original_score: float      # 원본 검색 점수
    rerank_score: float       # 재순위화 점수
    calibrated_score: float   # 캘리브레이션된 점수
    confidence: float         # 신뢰도
    rank: int                 # 최종 순위
    passed_threshold: bool    # 임계값 통과 여부

class LegalReranker:
    """법률 재순위화 클래스"""
    
    def __init__(self, config: Optional[RerankConfig] = None):
        """재순위화기 초기화"""
        self.config = config or RerankConfig()
        
        # 크로스엔코더 모델 로드
        self.cross_encoder = None
        self._load_cross_encoder()
        
        # 캘리브레이션 파라미터 (학습된 값 또는 기본값)
        self.calibration_params = {
            'temperature': self.config.temperature,
            'bias': 0.0
        }
        
        # 통계 정보
        self.stats = {
            'total_queries': 0,
            'total_candidates': 0,
            'passed_threshold': 0,
            'avg_score_improvement': 0.0
        }
        
        logger.info(f"법률 재순위화기 초기화 완료 (모델: {self.config.model_name})")
    
    def _load_cross_encoder(self):
        """크로스엔코더 모델 로드"""
        try:
            self.cross_encoder = CrossEncoder(
                self.config.model_name,
                max_length=512,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            logger.info(f"크로스엔코더 로드 완료: {self.config.model_name}")
        except Exception as e:
            logger.error(f"크로스엔코더 로드 실패: {e}")
            # Fallback: 기본 한국어 모델 시도
            try:
                self.cross_encoder = CrossEncoder(
                    "jhgan/ko-sroberta-multitask",
                    max_length=512,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                logger.info("Fallback 모델 로드 완료: jhgan/ko-sroberta-multitask")
            except Exception as e2:
                logger.error(f"Fallback 모델도 로드 실패: {e2}")
                raise
    
    def rerank(self, 
               query: str, 
               search_results: List[SearchResult],
               top_k: Optional[int] = None) -> List[RerankResult]:
        """검색 결과 재순위화"""
        if not search_results:
            return []
        
        if self.cross_encoder is None:
            logger.warning("크로스엔코더가 로드되지 않음. 원본 순서 반환")
            return self._create_fallback_results(search_results, top_k)
        
        # 후보 수 제한
        candidates = search_results[:self.config.max_candidates]
        
        logger.info(f"재순위화 시작: {len(candidates)}개 후보")
        
        # 쿼리-문서 쌍 생성
        query_doc_pairs = []
        for result in candidates:
            # 법률 컨텍스트 추가
            context = self._build_legal_context(result.chunk)
            doc_text = f"{context}\n\n{result.chunk.text}"
            query_doc_pairs.append([query, doc_text])
        
        # 크로스엔코더 점수 계산
        try:
            cross_scores = self.cross_encoder.predict(
                query_doc_pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )
        except Exception as e:
            logger.error(f"크로스엔코더 예측 실패: {e}")
            return self._create_fallback_results(search_results, top_k)
        
        # 캘리브레이션 적용
        if self.config.calibration_enabled:
            calibrated_scores = self._apply_calibration(cross_scores)
        else:
            calibrated_scores = cross_scores
        
        # 결과 생성 및 정렬
        rerank_results = []
        for i, (result, raw_score, cal_score) in enumerate(zip(candidates, cross_scores, calibrated_scores)):
            confidence = self._calculate_confidence(cal_score)
            passed_threshold = cal_score >= self.config.threshold
            
            rerank_result = RerankResult(
                chunk=result.chunk,
                original_score=result.score,
                rerank_score=float(raw_score),
                calibrated_score=float(cal_score),
                confidence=confidence,
                rank=0,  # 정렬 후 설정
                passed_threshold=passed_threshold
            )
            rerank_results.append(rerank_result)
        
        # 캘리브레이션된 점수로 정렬
        rerank_results.sort(key=lambda x: x.calibrated_score, reverse=True)
        
        # 순위 설정
        for i, result in enumerate(rerank_results):
            result.rank = i + 1
        
        # 상위 결과 선택
        if top_k:
            rerank_results = rerank_results[:top_k]
        
        # 통계 업데이트
        self._update_stats(candidates, rerank_results)
        
        logger.info(f"재순위화 완료: {len(rerank_results)}개 결과, "
                   f"{sum(1 for r in rerank_results if r.passed_threshold)}개 임계값 통과")
        
        return rerank_results
    
    def _build_legal_context(self, chunk: LegalChunk) -> str:
        """법률 컨텍스트 구성"""
        context_parts = []
        
        # 법률명
        context_parts.append(f"법률: {chunk.metadata.law_title}")
        
        # 조문 정보
        if chunk.metadata.article_no:
            article_info = f"제{chunk.metadata.article_no}조"
            if chunk.metadata.clause_no:
                article_info += f" 제{chunk.metadata.clause_no}항"
            context_parts.append(f"조문: {article_info}")
        
        # 계층 구조
        if chunk.metadata.section_hierarchy:
            context_parts.append(f"구조: {chunk.metadata.section_hierarchy}")
        
        # 시행일
        if chunk.metadata.effective_date:
            context_parts.append(f"시행일: {chunk.metadata.effective_date}")
        
        return " | ".join(context_parts)
    
    def _apply_calibration(self, scores: np.ndarray) -> np.ndarray:
        """점수 캘리브레이션 적용 (Temperature Scaling)"""
        # Temperature scaling: σ(z/T + b)
        temperature = self.calibration_params['temperature']
        bias = self.calibration_params['bias']
        
        # 시그모이드 함수 적용
        calibrated = 1 / (1 + np.exp(-(scores / temperature + bias)))
        
        return calibrated
    
    def _calculate_confidence(self, calibrated_score: float) -> float:
        """신뢰도 계산 (개선된 버전)"""
        # 캘리브레이션된 점수를 기반으로 비선형 신뢰도 계산
        if calibrated_score >= 0.8:
            confidence = 0.9 + 0.1 * (calibrated_score - 0.8) / 0.2
        elif calibrated_score >= 0.6:
            confidence = 0.7 + 0.2 * (calibrated_score - 0.6) / 0.2
        elif calibrated_score >= 0.4:
            confidence = 0.5 + 0.2 * (calibrated_score - 0.4) / 0.2
        else:
            confidence = 0.5 * calibrated_score / 0.4
        
        return min(confidence, 1.0)
    
    def _create_fallback_results(self, 
                                search_results: List[SearchResult], 
                                top_k: Optional[int]) -> List[RerankResult]:
        """크로스엔코더 실패 시 fallback 결과 생성"""
        fallback_results = []
        
        results_to_process = search_results[:top_k] if top_k else search_results
        
        for i, result in enumerate(results_to_process):
            # 원본 점수를 기반으로 신뢰도 추정
            confidence = min(result.score, 1.0) if result.score > 0 else 0.5
            
            fallback_result = RerankResult(
                chunk=result.chunk,
                original_score=result.score,
                rerank_score=result.score,  # 원본 점수 사용
                calibrated_score=result.score,
                confidence=confidence,
                rank=i + 1,
                passed_threshold=result.score >= self.config.threshold
            )
            fallback_results.append(fallback_result)
        
        return fallback_results
    
    def _update_stats(self, 
                     candidates: List[SearchResult], 
                     rerank_results: List[RerankResult]):
        """통계 정보 업데이트"""
        self.stats['total_queries'] += 1
        self.stats['total_candidates'] += len(candidates)
        self.stats['passed_threshold'] += sum(1 for r in rerank_results if r.passed_threshold)
        
        # 점수 개선도 계산 (상위 결과들 대상)
        if candidates and rerank_results:
            original_top_score = candidates[0].score
            rerank_top_score = rerank_results[0].calibrated_score
            
            if original_top_score > 0:
                improvement = (rerank_top_score - original_top_score) / original_top_score
                # 이동 평균 업데이트
                alpha = 0.1
                self.stats['avg_score_improvement'] = (
                    (1 - alpha) * self.stats['avg_score_improvement'] + 
                    alpha * improvement
                )
    
    def filter_by_threshold(self, 
                           rerank_results: List[RerankResult],
                           threshold: Optional[float] = None) -> List[RerankResult]:
        """임계값으로 결과 필터링"""
        threshold = threshold or self.config.threshold
        
        filtered = [r for r in rerank_results if r.calibrated_score >= threshold]
        
        logger.info(f"임계값 필터링: {len(filtered)}/{len(rerank_results)} 결과 통과 (임계값: {threshold})")
        
        return filtered
    
    def get_conservative_results(self, 
                               rerank_results: List[RerankResult],
                               min_confidence: float = 0.7,
                               max_results: int = 3) -> List[RerankResult]:
        """보수적 결과 반환 (높은 신뢰도만)"""
        high_confidence = [
            r for r in rerank_results 
            if r.confidence >= min_confidence and r.passed_threshold
        ]
        
        return high_confidence[:max_results]
    
    def calibrate_threshold(self, 
                           validation_data: List[Tuple[str, List[SearchResult], List[bool]]],
                           target_precision: float = 0.9) -> float:
        """검증 데이터를 사용한 임계값 캘리브레이션"""
        logger.info("임계값 캘리브레이션 시작...")
        
        all_scores = []
        all_labels = []
        
        for query, search_results, relevance_labels in validation_data:
            rerank_results = self.rerank(query, search_results)
            
            for result, label in zip(rerank_results, relevance_labels):
                all_scores.append(result.calibrated_score)
                all_labels.append(label)
        
        # 다양한 임계값에서 정밀도 계산
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_precision = 0.0
        
        for threshold in thresholds:
            predictions = [score >= threshold for score in all_scores]
            
            # 정밀도 계산
            true_positives = sum(1 for pred, label in zip(predictions, all_labels) 
                               if pred and label)
            predicted_positives = sum(predictions)
            
            if predicted_positives > 0:
                precision = true_positives / predicted_positives
                if precision >= target_precision and precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
        
        logger.info(f"최적 임계값: {best_threshold:.3f} (정밀도: {best_precision:.3f})")
        
        # 설정 업데이트
        self.config.threshold = best_threshold
        
        return best_threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = self.stats.copy()
        
        # 비율 계산
        if stats['total_candidates'] > 0:
            stats['threshold_pass_rate'] = stats['passed_threshold'] / stats['total_candidates']
        else:
            stats['threshold_pass_rate'] = 0.0
        
        stats['config'] = {
            'model_name': self.config.model_name,
            'threshold': self.config.threshold,
            'calibration_enabled': self.config.calibration_enabled,
            'temperature': self.calibration_params['temperature']
        }
        
        return stats
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'total_queries': 0,
            'total_candidates': 0,
            'passed_threshold': 0,
            'avg_score_improvement': 0.0
        }
        logger.info("재순위화 통계 초기화 완료")
