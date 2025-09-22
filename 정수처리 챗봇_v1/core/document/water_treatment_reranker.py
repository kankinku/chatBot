"""
정수처리 도메인 특화 재순위화 모듈

기존 PDFReranker를 정수처리 도메인에 특화시켜
공정별 전문 용어와 컨텍스트를 더 정확하게 이해하고 재순위화합니다.
"""

import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
import torch
import time

from .text_chunk import TextChunk
from .pdf_retriever import PDFSearchResult

import logging
logger = logging.getLogger(__name__)

@dataclass
class WaterTreatmentRerankConfig:
    """정수처리 재순위화 설정"""
    model_name: str = "BAAI/bge-reranker-v2-m3"  # 기본 크로스엔코더 모델
    batch_size: int = 16                         # 배치 크기 (정수처리 최적화)
    max_candidates: int = 20                     # 재순위화 후보 수
    threshold: float = 0.4                       # 정수처리 특화 임계값
    calibration_enabled: bool = True             # 점수 캘리브레이션 사용
    temperature: float = 1.2                     # Temperature scaling (정수처리 최적화)
    domain_weight: float = 0.3                   # 도메인 특화 가중치
    process_weight: float = 0.2                  # 공정 매칭 가중치
    technical_weight: float = 0.25               # 기술적 정확성 가중치
    semantic_weight: float = 0.25                # 의미적 유사도 가중치

@dataclass
class WaterTreatmentRerankResult:
    """정수처리 재순위화 결과"""
    chunk: TextChunk
    original_score: float
    rerank_score: float
    calibrated_score: float
    domain_score: float      # 도메인 특화 점수
    process_score: float     # 공정 매칭 점수
    technical_score: float   # 기술적 정확성 점수
    confidence: float
    rank: int
    passed_threshold: bool
    process_matches: List[str]  # 매칭된 공정들

class WaterTreatmentReranker:
    """정수처리 도메인 특화 재순위화 클래스"""
    
    def __init__(self, config: Optional[WaterTreatmentRerankConfig] = None):
        """재순위화기 초기화"""
        self.config = config or WaterTreatmentRerankConfig()
        
        # 크로스엔코더 모델 로드
        self.cross_encoder = None
        self._load_cross_encoder()
        
        # 캘리브레이션 파라미터 (정수처리 도메인 특화)
        self.calibration_params = {
            'temperature': self.config.temperature,
            'bias': 0.1  # 정수처리 도메인에 약간의 bias 추가
        }
        
        # 향상된 정수처리 도메인 특화 패턴 (도메인 분류 기반)
        self.domain_patterns = self._load_enhanced_domain_patterns()
        
        # 통계 정보
        self.stats = {
            'total_queries': 0,
            'total_candidates': 0,
            'passed_threshold': 0,
            'avg_score_improvement': 0.0,
            'process_distribution': {}
        }
        
        logger.info(f"정수처리 재순위화기 초기화 완료 (모델: {self.config.model_name}, 임계값: {self.config.threshold})")
    
    def _load_cross_encoder(self):
        """크로스엔코더 모델 로드"""
        try:
            self.cross_encoder = CrossEncoder(
                self.config.model_name,
                max_length=512,  # 정수처리 문서에 적합한 길이
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            logger.info(f"정수처리 크로스엔코더 로드 완료: {self.config.model_name}")
        except Exception as e:
            logger.error(f"크로스엔코더 로드 실패: {e}")
            # Fallback: 한국어 모델 시도
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
    
    def _load_enhanced_domain_patterns(self) -> Dict[str, Dict[str, Any]]:
        """향상된 도메인 패턴 로드"""
        try:
            import json
            import os
            
            pattern_file = "config/enhanced_patterns/enhanced_reranker_patterns.json"
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 패턴 구조 변환
                enhanced_patterns = {}
                for category, domains in data.get('patterns', {}).items():
                    for domain_name, domain_info in domains.items():
                        enhanced_patterns[domain_name] = {
                            'keywords': domain_info.get('keywords', []),
                            'technical_terms': domain_info.get('technical_terms', []),
                            'weight': domain_info.get('weight', 1.0)
                        }
                
                logger.info(f"향상된 재순위화 패턴 로드 완료: {len(enhanced_patterns)}개 도메인")
                return enhanced_patterns
            else:
                logger.warning("향상된 재순위화 패턴 파일이 없습니다. 기본 패턴을 사용합니다.")
                return self._build_domain_patterns()
                
        except Exception as e:
            logger.error(f"향상된 재순위화 패턴 로드 실패: {e}. 기본 패턴을 사용합니다.")
            return self._build_domain_patterns()
    
    def _build_domain_patterns(self) -> Dict[str, Dict[str, Any]]:
        """정수처리 도메인 특화 패턴 구축"""
        return {
            'processes': {
                'intake': {
                    'keywords': ['착수', '수위', '목표값', '유입량', '정수지', 'k-means', '군집분석'],
                    'technical_terms': ['수위제어', '유입량조절', 'k-means 군집분석', '월별 목표값'],
                    'weight': 1.0
                },
                'coagulation': {
                    'keywords': ['약품', '응집제', '주입률', 'n-beats', '탁도', '알칼리도'],
                    'technical_terms': ['응집제 주입률', 'n-beats 모델', '탁도 측정', '전기전도도'],
                    'weight': 1.2  # 약품 공정은 중요도 높음
                },
                'mixing_flocculation': {
                    'keywords': ['혼화', '응집', '회전속도', 'rpm', '교반', 'g값'],
                    'technical_terms': ['교반기 회전속도', 'g값 계산', '동점성계수', '설비값'],
                    'weight': 1.0
                },
                'sedimentation': {
                    'keywords': ['침전', '슬러지', '발생량', '수집기', '대차'],
                    'technical_terms': ['슬러지 발생량', '수집기 운전', '대차 스케줄', '상관계수'],
                    'weight': 1.0
                },
                'filtration': {
                    'keywords': ['여과', '여과지', '세척', '주기', '운전'],
                    'technical_terms': ['여과지 세척', '역세척 주기', '여과 운전', '수위 관리'],
                    'weight': 1.0
                },
                'disinfection': {
                    'keywords': ['소독', '염소', '잔류염소', '주입률', '체류시간'],
                    'technical_terms': ['염소 주입률', '잔류염소 농도', '체류시간', '전차염'],
                    'weight': 1.0
                }
            },
            'systems': {
                'ems': {
                    'keywords': ['ems', '펌프', '제어', '전력', '피크', '에너지'],
                    'technical_terms': ['ems 시스템', '전력 피크', '에너지 관리', '펌프 제어'],
                    'weight': 1.1
                },
                'pms': {
                    'keywords': ['pms', '모터', '진단', '전류', '진동', '온도'],
                    'technical_terms': ['pms 시스템', '모터 진단', '예방정비', '설비 관리'],
                    'weight': 1.1
                }
            },
            'performance_metrics': {
                'ai_models': {
                    'keywords': ['n-beats', 'xgb', 'lstm', 'mae', 'mse', 'rmse', 'r²'],
                    'technical_terms': ['n-beats 모델', 'xgb 모델', 'mae 0.68', 'r² 0.97'],
                    'weight': 1.3  # 성능 지표는 매우 중요
                },
                'operational': {
                    'keywords': ['정확도', '성능', '효율', '최적화', '예측'],
                    'technical_terms': ['모델 성능', '운전 최적화', '예측 정확도'],
                    'weight': 1.1
                }
            }
        }
    
    def rerank(self, 
               query: str, 
               search_results: List[PDFSearchResult],
               top_k: Optional[int] = None,
               answer_target: Optional[str] = None,
               target_type: Optional[str] = None) -> List[WaterTreatmentRerankResult]:
        """정수처리 도메인 특화 검색 결과 재순위화"""
        if not search_results:
            return []
        
        if self.cross_encoder is None:
            logger.warning("크로스엔코더가 로드되지 않음. 원본 순서 반환")
            return self._create_fallback_results(search_results, top_k)
        
        start_time = time.time()
        
        # 후보 수 제한
        candidates = search_results[:self.config.max_candidates]
        
        logger.info(f"정수처리 재순위화 시작: {len(candidates)}개 후보")
        
        # 1단계: 크로스엔코더 점수 계산
        cross_scores = self._calculate_cross_encoder_scores(query, candidates)
        
        # 2단계: 도메인 특화 점수 계산
        domain_scores = self._calculate_domain_scores(query, candidates, answer_target, target_type)
        
        # 3단계: 종합 점수 계산 및 결과 생성
        rerank_results = self._create_rerank_results(
            query, candidates, cross_scores, domain_scores, answer_target, target_type
        )
        
        # 4단계: 캘리브레이션 및 정렬
        final_results = self._apply_calibration_and_sort(rerank_results)
        
        # 5단계: 상위 결과 선택
        if top_k:
            final_results = final_results[:top_k]
        
        # 통계 업데이트
        self._update_stats(candidates, final_results)
        
        processing_time = time.time() - start_time
        logger.info(f"정수처리 재순위화 완료: {processing_time:.3f}초, {len(final_results)}개 결과, "
                   f"{sum(1 for r in final_results if r.passed_threshold)}개 임계값 통과")
        
        return final_results
    
    def _calculate_cross_encoder_scores(self, query: str, candidates: List[PDFSearchResult]) -> List[float]:
        """크로스엔코더 점수 계산"""
        try:
            # 쿼리-문서 쌍 생성 (정수처리 컨텍스트 강화)
            query_doc_pairs = []
            for result in candidates:
                # 정수처리 컨텍스트 추가
                context = self._build_water_treatment_context(result.chunk)
                doc_text = f"{context}\n\n{result.chunk.content}"
                query_doc_pairs.append([query, doc_text])
            
            # 크로스엔코더 예측
            cross_scores = self.cross_encoder.predict(
                query_doc_pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )
            
            return cross_scores.tolist() if isinstance(cross_scores, np.ndarray) else cross_scores
            
        except Exception as e:
            logger.error(f"크로스엔코더 예측 실패: {e}")
            # 폴백: 원본 점수 사용
            return [result.score for result in candidates]
    
    def _calculate_domain_scores(self, 
                               query: str,
                               candidates: List[PDFSearchResult],
                               answer_target: Optional[str],
                               target_type: Optional[str]) -> List[Dict[str, float]]:
        """도메인 특화 점수 계산"""
        query_lower = query.lower()
        target_lower = answer_target.lower() if answer_target else ""
        
        domain_scores = []
        
        for result in candidates:
            chunk = result.chunk
            content_lower = chunk.content.lower()
            
            # 1. 공정 매칭 점수
            process_score, matched_processes = self._calculate_process_score(
                content_lower, query_lower, target_lower
            )
            
            # 2. 기술적 정확성 점수
            technical_score = self._calculate_technical_score(
                content_lower, query_lower, target_lower, target_type
            )
            
            # 3. 도메인 특화 키워드 점수
            domain_score = self._calculate_domain_keyword_score(
                content_lower, query_lower, target_lower
            )
            
            domain_scores.append({
                'process_score': process_score,
                'technical_score': technical_score,
                'domain_score': domain_score,
                'matched_processes': matched_processes
            })
        
        return domain_scores
    
    def _calculate_process_score(self, 
                               content_lower: str,
                               query_lower: str,
                               target_lower: str) -> Tuple[float, List[str]]:
        """공정 매칭 점수 계산"""
        score = 0.0
        matched_processes = []
        
        # 도메인 패턴 구조에 따라 처리
        if isinstance(self.domain_patterns, dict):
            # 각 공정별 매칭 확인
            for category_or_process, processes_or_info in self.domain_patterns.items():
                if isinstance(processes_or_info, dict):
                    # 중첩 구조인 경우 (카테고리 -> 프로세스)
                    if any(isinstance(v, dict) and 'keywords' in v for v in processes_or_info.values()):
                        # 카테고리 구조: {'processes': {'intake': {...}}}
                        for process_name, process_info in processes_or_info.items():
                            if isinstance(process_info, dict) and 'keywords' in process_info:
                                process_weight = process_info.get('weight', 1.0)
                                
                                # 키워드 매칭
                                keyword_matches = sum(1 for keyword in process_info.get('keywords', []) 
                                                    if keyword in content_lower)
                                
                                # 기술 용어 매칭 (더 높은 가중치)
                                technical_matches = sum(2 for term in process_info.get('technical_terms', []) 
                                                      if term in content_lower)
                                
                                if keyword_matches > 0 or technical_matches > 0:
                                    process_score = (keyword_matches + technical_matches) * process_weight
                                    
                                    # 쿼리나 답변 목표에서도 해당 공정이 언급되면 추가 점수
                                    if any(keyword in query_lower for keyword in process_info.get('keywords', [])):
                                        process_score *= 1.5
                                    
                                    if target_lower and any(keyword in target_lower for keyword in process_info.get('keywords', [])):
                                        process_score *= 1.3
                                    
                                    score += process_score
                                    matched_processes.append(process_name)
                    else:
                        # 평면 구조: {'intake': {...}, 'ems': {...}}
                        process_info = processes_or_info
                        if isinstance(process_info, dict) and 'keywords' in process_info:
                            process_weight = process_info.get('weight', 1.0)
                            
                            # 키워드 매칭
                            keyword_matches = sum(1 for keyword in process_info.get('keywords', []) 
                                                if keyword in content_lower)
                            
                            # 기술 용어 매칭 (더 높은 가중치)
                            technical_matches = sum(2 for term in process_info.get('technical_terms', []) 
                                                  if term in content_lower)
                            
                            if keyword_matches > 0 or technical_matches > 0:
                                process_score = (keyword_matches + technical_matches) * process_weight
                                
                                # 쿼리나 답변 목표에서도 해당 공정이 언급되면 추가 점수
                                if any(keyword in query_lower for keyword in process_info.get('keywords', [])):
                                    process_score *= 1.5
                                
                                if target_lower and any(keyword in target_lower for keyword in process_info.get('keywords', [])):
                                    process_score *= 1.3
                                
                                score += process_score
                                matched_processes.append(category_or_process)
        
        return min(score / 10.0, 1.0), matched_processes  # 정규화
    
    def _calculate_technical_score(self, 
                                 content_lower: str,
                                 query_lower: str,
                                 target_lower: str,
                                 target_type: Optional[str]) -> float:
        """기술적 정확성 점수 계산"""
        score = 0.0
        
        # 성능 지표 매칭 (정량적 값 질문에 중요)
        if target_type == "quantitative_value":
            performance_patterns = [
                r'mae\s*[:=]?\s*\d+\.?\d*',
                r'mse\s*[:=]?\s*\d+\.?\d*',
                r'rmse\s*[:=]?\s*\d+\.?\d*',
                r'r²\s*[:=]?\s*0\.\d+',
                r'r2\s*[:=]?\s*0\.\d+',
                r'\d+\.?\d*\s*(?:mg/l|%|rpm|m³/h)'
            ]
            
            for pattern in performance_patterns:
                if re.search(pattern, content_lower):
                    score += 0.3
        
        # 모델명 정확성 (n-beats, xgb, lstm 등)
        model_keywords = ['n-beats', 'xgb', 'lstm', 'extreme gradient boosting']
        for keyword in model_keywords:
            if keyword in query_lower and keyword in content_lower:
                score += 0.4
        
        # 공정별 전문 용어 매칭
        technical_terms = [
            '응집제 주입률', '교반기 회전속도', '슬러지 발생량', '잔류염소 농도',
            '여과지 세척', '수위 목표값', '전력 피크', '모터 진단'
        ]
        
        for term in technical_terms:
            if term in content_lower:
                # 쿼리나 답변 목표에서 관련 키워드가 있으면 추가 점수
                term_keywords = term.split()
                if any(keyword in query_lower or keyword in target_lower 
                       for keyword in term_keywords):
                    score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_domain_keyword_score(self, 
                                      content_lower: str,
                                      query_lower: str,
                                      target_lower: str) -> float:
        """도메인 특화 키워드 점수 계산"""
        score = 0.0
        
        # 정수처리 핵심 키워드
        core_keywords = [
            '정수처리', '정수장', '수질', '처리공정', '운영관리', '자동제어',
            'ai', '인공지능', '머신러닝', '딥러닝', '최적화', '예측모델'
        ]
        
        for keyword in core_keywords:
            if keyword in content_lower:
                if keyword in query_lower or keyword in target_lower:
                    score += 0.15  # 쿼리/목표와 매칭되면 높은 점수
                else:
                    score += 0.05  # 단순 포함시 낮은 점수
        
        # 시설 관리 키워드
        facility_keywords = [
            '펌프', '밸브', '센서', '모터', '교반기', '여과지', '침전지',
            '에너지', '전력', '설비', '장비', '시설', '관리'
        ]
        
        facility_matches = sum(1 for keyword in facility_keywords if keyword in content_lower)
        if facility_matches > 0:
            score += facility_matches * 0.05

        # 추가 부스팅: 요청된 핵심 도메인 키워드
        boost_pairs = [
            (['여과지 세척', '세척 주기', '역세척'], 0.25),
            (['회전수', '회전속도', 'rpm', 'g값', '교반 강도'], 0.25),
            (['n-beats', 'xgb', 'extreme gradient boosting'], 0.3),
            (['피크 관리', '전력 피크'], 0.25),
        ]
        for keywords, w in boost_pairs:
            if any(k in content_lower for k in keywords):
                # 쿼리나 목표에도 관련 키워드가 있으면 가중 증폭
                if any(k in query_lower or k in target_lower for k in keywords):
                    score += w * 1.5
                else:
                    score += w
        
        return min(score, 1.0)
    
    def _build_water_treatment_context(self, search_result) -> str:
        """정수처리 컨텍스트 정보 구성"""
        context_parts = []
        
        # PDFSearchResult에서 실제 chunk 추출
        chunk = search_result.chunk if hasattr(search_result, 'chunk') else search_result
        
        # 청크 메타데이터에서 공정 정보 추출
        if hasattr(chunk, 'metadata') and chunk.metadata:
            if isinstance(chunk.metadata, dict):
                process_keywords = chunk.metadata.get('process_keywords', [])
                if process_keywords:
                    context_parts.append(f"관련 공정: {', '.join(process_keywords)}")
                
                chunk_type = chunk.metadata.get('chunk_type', '')
                if chunk_type:
                    context_parts.append(f"문서 유형: {chunk_type}")
        
        # 페이지 정보
        if hasattr(chunk, 'page_number') and chunk.page_number:
            context_parts.append(f"페이지: {chunk.page_number}")
        
        return " | ".join(context_parts) if context_parts else "정수처리 관련 문서"
    
    def _create_rerank_results(self, 
                             query: str,
                             candidates: List[PDFSearchResult],
                             cross_scores: List[float],
                             domain_scores: List[Dict[str, float]],
                             answer_target: Optional[str],
                             target_type: Optional[str]) -> List[WaterTreatmentRerankResult]:
        """재순위화 결과 생성"""
        results = []
        
        for i, (candidate, cross_score, domain_info) in enumerate(zip(candidates, cross_scores, domain_scores)):
            # 가중 평균으로 최종 점수 계산
            final_score = (
                cross_score * self.config.semantic_weight +
                domain_info['domain_score'] * self.config.domain_weight +
                domain_info['process_score'] * self.config.process_weight +
                domain_info['technical_score'] * self.config.technical_weight
            )
            
            result = WaterTreatmentRerankResult(
                chunk=candidate.chunk,
                original_score=candidate.score,
                rerank_score=cross_score,
                calibrated_score=final_score,  # 캘리브레이션은 나중에 적용
                domain_score=domain_info['domain_score'],
                process_score=domain_info['process_score'],
                technical_score=domain_info['technical_score'],
                confidence=min(final_score, 1.0),
                rank=i + 1,  # 임시 순위
                passed_threshold=final_score >= self.config.threshold,
                process_matches=domain_info['matched_processes']
            )
            
            results.append(result)
        
        return results
    
    def _apply_calibration_and_sort(self, results: List[WaterTreatmentRerankResult]) -> List[WaterTreatmentRerankResult]:
        """캘리브레이션 적용 및 정렬"""
        if not self.config.calibration_enabled:
            # 캘리브레이션 없이 정렬만
            results.sort(key=lambda x: x.calibrated_score, reverse=True)
        else:
            # 캘리브레이션 적용
            scores = [r.calibrated_score for r in results]
            calibrated_scores = self._apply_calibration(scores)
            
            for result, cal_score in zip(results, calibrated_scores):
                result.calibrated_score = cal_score
                result.confidence = min(cal_score, 1.0)
                result.passed_threshold = cal_score >= self.config.threshold
            
            # 캘리브레이션된 점수로 정렬
            results.sort(key=lambda x: x.calibrated_score, reverse=True)
        
        # 순위 재할당
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _apply_calibration(self, scores: List[float]) -> List[float]:
        """Temperature scaling 캘리브레이션 적용"""
        scores_array = np.array(scores)
        
        # Temperature scaling
        calibrated_scores = scores_array / self.calibration_params['temperature']
        
        # Bias 적용
        calibrated_scores += self.calibration_params['bias']
        
        # 시그모이드 적용하여 0-1 범위로 정규화
        calibrated_scores = 1 / (1 + np.exp(-calibrated_scores))
        
        return calibrated_scores.tolist()
    
    def _create_fallback_results(self, 
                                search_results: List[PDFSearchResult],
                                top_k: Optional[int]) -> List[WaterTreatmentRerankResult]:
        """폴백 결과 생성 (크로스엔코더 실패시)"""
        results = []
        
        for i, result in enumerate(search_results[:top_k] if top_k else search_results):
            fallback_result = WaterTreatmentRerankResult(
                chunk=result.chunk,
                original_score=result.score,
                rerank_score=result.score,
                calibrated_score=result.score,
                domain_score=0.5,  # 기본값
                process_score=0.5,
                technical_score=0.5,
                confidence=result.score,
                rank=i + 1,
                passed_threshold=result.score >= self.config.threshold,
                process_matches=[]
            )
            results.append(fallback_result)
        
        return results
    
    def _update_stats(self, candidates: List[PDFSearchResult], results: List[WaterTreatmentRerankResult]):
        """통계 업데이트"""
        self.stats['total_queries'] += 1
        self.stats['total_candidates'] += len(candidates)
        self.stats['passed_threshold'] += sum(1 for r in results if r.passed_threshold)
        
        # 공정별 분포 업데이트
        for result in results:
            for process in result.process_matches:
                self.stats['process_distribution'][process] = \
                    self.stats['process_distribution'].get(process, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            **self.stats,
            'avg_candidates_per_query': self.stats['total_candidates'] / max(self.stats['total_queries'], 1),
            'threshold_pass_rate': self.stats['passed_threshold'] / max(self.stats['total_candidates'], 1)
        }
