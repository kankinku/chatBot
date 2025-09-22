"""
PDF 전용 재순위화 모듈

크로스엔코더를 사용하여 PDF 검색 결과를 재순위화하고,
점수 캘리브레이션을 통해 신뢰도 있는 결과를 제공합니다.
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
import torch

from .pdf_processor import TextChunk
from .pdf_retriever import PDFSearchResult

import logging
logger = logging.getLogger(__name__)

@dataclass
class PDFRerankConfig:
    """PDF 재순위화 설정"""
    model_name: str = "BAAI/bge-reranker-v2-m3"  # 크로스엔코더 모델
    batch_size: int = 32                         # 배치 크기 (GPU 최적화)
    max_candidates: int = 20                     # 재순위화할 최대 후보 수
    threshold: float = 0.55                      # 신뢰도 임계값
    calibration_enabled: bool = True             # 점수 캘리브레이션 사용 여부
    temperature: float = 1.0                     # Temperature scaling 파라미터
    max_length: int = 384                        # 최대 시퀀스 길이 (지연 최적화)

@dataclass
class PDFRerankResult:
    """PDF 재순위화 결과"""
    chunk: TextChunk
    original_score: float
    rerank_score: float
    calibrated_score: float
    confidence: float
    rank: int
    passed_threshold: bool

class PDFReranker:
    """PDF 재순위화 클래스"""
    
    def __init__(self, config: Optional[PDFRerankConfig] = None):
        """재순위화기 초기화"""
        self.config = config or PDFRerankConfig()
        
        # 크로스엔코더 모델 로드
        self.cross_encoder = None
        self._load_cross_encoder()
        
        # 캘리브레이션 파라미터
        self.calibration_params = {
            'temperature': max(0.8, self.config.temperature * 0.8),
            'bias': 0.0
        }
        
        # 통계 정보
        self.stats = {
            'total_queries': 0,
            'total_candidates': 0,
            'passed_threshold': 0,
            'avg_score_improvement': 0.0
        }
        
        logger.info(f"PDF 재순위화기 초기화 완료 (모델: {self.config.model_name})")
    
    def _load_cross_encoder(self):
        """크로스엔코더 모델 로드"""
        try:
            self.cross_encoder = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            logger.info(f"PDF 크로스엔코더 로드 완료: {self.config.model_name}")
        except Exception as e:
            logger.error(f"크로스엔코더 로드 실패: {e}")
            # Fallback: 기본 한국어 모델 시도
            try:
                self.cross_encoder = CrossEncoder(
                    "jhgan/ko-sroberta-multitask",
                    max_length=self.config.max_length,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                logger.info("Fallback 모델 로드 완료: jhgan/ko-sroberta-multitask")
            except Exception as e2:
                logger.error(f"Fallback 모델도 로드 실패: {e2}")
                raise
    
    def rerank(self, 
               query: str, 
               search_results: List[PDFSearchResult],
               top_k: Optional[int] = None) -> List[PDFRerankResult]:
        """PDF 검색 결과 재순위화"""
        if not search_results:
            return []
        
        if self.cross_encoder is None:
            logger.warning("크로스엔코더가 로드되지 않음. 원본 순서 반환")
            return self._create_fallback_results(search_results, top_k)
        
        # 후보 수 제한
        candidates = search_results[:self.config.max_candidates]
        
        logger.info(f"PDF 재순위화 시작: {len(candidates)}개 후보")
        
        # 쿼리-문서 쌍 생성 (PDF 컨텍스트 강화)
        query_doc_pairs = []
        for result in candidates:
            # PDF 컨텍스트 추가
            context = self._build_pdf_context(result.chunk)
            doc_text = f"{context}\n\n{result.chunk.content}"
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
            confidence = min(float(cal_score), 1.0)
            passed_threshold = confidence >= self.config.threshold
            
            rerank_results.append(PDFRerankResult(
                chunk=result.chunk,
                original_score=result.score,
                rerank_score=float(raw_score),
                calibrated_score=float(cal_score),
                confidence=confidence,
                rank=i + 1,
                passed_threshold=passed_threshold
            ))
        
        # 점수 순으로 정렬
        rerank_results.sort(key=lambda x: x.calibrated_score, reverse=True)
        
        # 순위 재할당
        for i, result in enumerate(rerank_results):
            result.rank = i + 1
        
        # 통계 업데이트
        self._update_stats(search_results, rerank_results)
        
        # 상위 결과 반환
        final_results = rerank_results[:top_k] if top_k else rerank_results
        
        logger.info(f"PDF 재순위화 완료: {len(final_results)}개 결과, 임계값 통과: {sum(1 for r in final_results if r.passed_threshold)}개")
        
        return final_results
    
    def _build_pdf_context(self, chunk: TextChunk) -> str:
        """PDF 청크의 컨텍스트 정보 구성"""
        context_parts = []
        
        # 문서 제목
        if chunk.metadata and 'title' in chunk.metadata:
            context_parts.append(f"문서: {chunk.metadata['title']}")
        
        # 페이지 정보
        if hasattr(chunk, 'page_number'):
            context_parts.append(f"페이지: {chunk.page_number}")
        
        # 섹션 정보
        if chunk.metadata:
            if 'section' in chunk.metadata:
                context_parts.append(f"섹션: {chunk.metadata['section']}")
            if 'subsection' in chunk.metadata:
                context_parts.append(f"하위섹션: {chunk.metadata['subsection']}")
        
        # 키워드 정보
        keywords = []
        if chunk.metadata and 'keywords' in chunk.metadata:
            keywords.extend(chunk.metadata['keywords'])
        if hasattr(chunk, 'keywords') and chunk.keywords:
            keywords.extend(chunk.keywords)
        
        if keywords:
            context_parts.append(f"키워드: {', '.join(keywords[:5])}")  # 상위 5개만
        
        return " | ".join(context_parts) if context_parts else "PDF 문서"
    
    def _apply_calibration(self, scores: np.ndarray) -> np.ndarray:
        """점수 캘리브레이션 적용"""
        # Temperature scaling
        calibrated = scores / self.calibration_params['temperature']
        
        # Bias 조정
        calibrated += self.calibration_params['bias']
        
        # Sigmoid 적용 (0-1 범위로 정규화)
        calibrated = 1 / (1 + np.exp(-calibrated))
        
        return calibrated
    
    def _create_fallback_results(self, 
                                search_results: List[PDFSearchResult],
                                top_k: Optional[int]) -> List[PDFRerankResult]:
        """Fallback 결과 생성"""
        results = []
        for i, result in enumerate(search_results[:top_k or len(search_results)]):
            results.append(PDFRerankResult(
                chunk=result.chunk,
                original_score=result.score,
                rerank_score=result.score,
                calibrated_score=result.score,
                confidence=min(result.score, 1.0),
                rank=i + 1,
                passed_threshold=result.score >= self.config.threshold
            ))
        return results
    
    def _update_stats(self, 
                     original_results: List[PDFSearchResult],
                     rerank_results: List[PDFRerankResult]):
        """통계 정보 업데이트"""
        self.stats['total_queries'] += 1
        self.stats['total_candidates'] += len(original_results)
        self.stats['passed_threshold'] += sum(1 for r in rerank_results if r.passed_threshold)
        
        # 평균 점수 개선 계산
        if original_results and rerank_results:
            original_avg = np.mean([r.score for r in original_results])
            rerank_avg = np.mean([r.calibrated_score for r in rerank_results])
            improvement = rerank_avg - original_avg
            self.stats['avg_score_improvement'] = (
                (self.stats['avg_score_improvement'] * (self.stats['total_queries'] - 1) + improvement) 
                / self.stats['total_queries']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            'config': {
                'model_name': self.config.model_name,
                'batch_size': self.config.batch_size,
                'max_candidates': self.config.max_candidates,
                'threshold': self.config.threshold,
                'calibration_enabled': self.config.calibration_enabled
            },
            'stats': self.stats.copy(),
            'calibration_params': self.calibration_params.copy()
        }
    
    def update_calibration_params(self, temperature: Optional[float] = None, 
                                 bias: Optional[float] = None):
        """캘리브레이션 파라미터 업데이트"""
        if temperature is not None:
            self.calibration_params['temperature'] = temperature
        if bias is not None:
            self.calibration_params['bias'] = bias
        
        logger.info(f"캘리브레이션 파라미터 업데이트: {self.calibration_params}")
    
    def reset_stats(self):
        """통계 정보 초기화"""
        self.stats = {
            'total_queries': 0,
            'total_candidates': 0,
            'passed_threshold': 0,
            'avg_score_improvement': 0.0
        }
        logger.info("통계 정보 초기화 완료")

