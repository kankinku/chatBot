"""
향상된 테스트 시스템

벡터 검색 강화 및 정확도 검증 기능을 포함한 종합 테스트 시스템
"""

import json
import time
import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.document.enhanced_pdf_pipeline import create_enhanced_pdf_pipeline
from core.document.enhanced_vector_search import create_enhanced_searcher
from core.document.accuracy_validator import AccuracyValidator
from core.document.vector_store import HybridVectorStore

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTestSystem:
    """향상된 테스트 시스템"""
    
    def __init__(self):
        """테스트 시스템 초기화"""
        self.test_results = []
        self.performance_stats = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'avg_search_time': 0.0,
            'avg_accuracy_score': 0.0
        }
        
        # 향상된 파이프라인 초기화
        self.pdf_pipeline = create_enhanced_pdf_pipeline()
        
        # 기존 청크 로드
        self._load_existing_chunks()
        
        logger.info("향상된 테스트 시스템 초기화 완료")
    
    def _load_existing_chunks(self):
        """기존 청크들 로드"""
        try:
            # 기존 벡터 저장소에서 청크 로드
            vector_store = HybridVectorStore()
            vector_store.load()
            
            # 파이프라인에 청크 추가
            if hasattr(vector_store, 'faiss_store') and vector_store.faiss_store.chunks:
                self.pdf_pipeline.process_pdf_chunks(vector_store.faiss_store.chunks)
                logger.info(f"기존 청크 {len(vector_store.faiss_store.chunks)}개 로드 완료")
            else:
                logger.warning("기존 청크를 찾을 수 없습니다")
                
        except Exception as e:
            logger.warning(f"기존 청크 로드 실패: {e}")
    
    def run_comprehensive_test(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """종합 테스트 실행"""
        logger.info(f"종합 테스트 시작: {len(test_data)}개 테스트 케이스")
        
        start_time = time.time()
        
        for i, test_case in enumerate(test_data):
            logger.info(f"테스트 {i+1}/{len(test_data)}: {test_case.get('question', 'Unknown')[:50]}...")
            
            test_result = self._run_single_test(test_case)
            self.test_results.append(test_result)
            
            # 성능 통계 업데이트
            self._update_performance_stats(test_result)
        
        total_time = time.time() - start_time
        
        # 전체 결과 분석
        analysis = self._analyze_results()
        
        return {
            'test_summary': {
                'total_tests': len(test_data),
                'total_time': total_time,
                'performance_stats': self.performance_stats
            },
            'detailed_results': self.test_results,
            'analysis': analysis
        }
    
    def _run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """단일 테스트 실행"""
        question = test_case.get('question', '')
        expected = test_case.get('expected', '')
        
        start_time = time.time()
        
        try:
            # 향상된 파이프라인으로 검색 및 답변 생성
            result = self.pdf_pipeline.search_and_answer(
                query=question,
                expected_answer=expected
            )
            
            test_time = time.time() - start_time
            
            # 정확도 평가
            accuracy_score = self._evaluate_accuracy(result['answer'], expected)
            
            # 검증 결과 분석
            validation = result.get('validation', {})
            enhanced_search = result.get('enhanced_search', {})
            
            return {
                'question': question,
                'expected': expected,
                'generated': result['answer'],
                'accuracy_score': accuracy_score,
                'test_time': test_time,
                'validation': {
                    'is_accurate': validation.get('is_accurate', False),
                    'confidence_score': validation.get('confidence_score', 0.0),
                    'issues': validation.get('accuracy_issues', []),
                    'suggestions': validation.get('improvement_suggestions', [])
                },
                'enhanced_search': {
                    'total_candidates': enhanced_search.get('total_candidates', 0),
                    'avg_confidence': enhanced_search.get('avg_confidence', 0.0),
                    'model_scores': enhanced_search.get('model_scores', {})
                },
                'search_performance': {
                    'search_time': result.get('search_results', {}).get('total_search_time', 0.0),
                    'generation_time': result.get('generation_time', 0.0),
                    'total_time': result.get('total_time', 0.0)
                },
                'success': accuracy_score >= 0.7
            }
            
        except Exception as e:
            logger.error(f"테스트 실행 실패: {e}")
            return {
                'question': question,
                'expected': expected,
                'generated': f"오류 발생: {str(e)}",
                'accuracy_score': 0.0,
                'test_time': time.time() - start_time,
                'validation': {'is_accurate': False, 'confidence_score': 0.0, 'issues': [str(e)], 'suggestions': []},
                'enhanced_search': {'total_candidates': 0, 'avg_confidence': 0.0, 'model_scores': {}},
                'search_performance': {'search_time': 0.0, 'generation_time': 0.0, 'total_time': 0.0},
                'success': False,
                'error': str(e)
            }
    
    def _evaluate_accuracy(self, generated: str, expected: str) -> float:
        """정확도 평가"""
        if not generated or not expected:
            return 0.0
        
        # 간단한 키워드 기반 정확도 평가
        generated_lower = generated.lower()
        expected_lower = expected.lower()
        
        # 중요한 키워드 추출
        important_keywords = [
            'k-means', '군집분석', 'n-beats', 'xgb', 'mae', 'mse', 'r²', 'r2',
            '상관계수', '교반기', '회전속도', '응집제', '주입률', '10분', '60분',
            '관리자', '사용자', '접근', '확인', '제공', '펌프', '모터', '진단'
        ]
        
        # 숫자 패턴 추출
        import re
        expected_numbers = re.findall(r'\d+(?:\.\d+)?', expected_lower)
        generated_numbers = re.findall(r'\d+(?:\.\d+)?', generated_lower)
        
        # 키워드 매칭 점수
        keyword_score = 0.0
        for keyword in important_keywords:
            if keyword in expected_lower:
                if keyword in generated_lower:
                    keyword_score += 1.0
                else:
                    keyword_score += 0.0
        
        if expected_numbers:
            keyword_score /= len(important_keywords)
        
        # 숫자 매칭 점수
        number_score = 0.0
        if expected_numbers:
            matched_numbers = 0
            for num in expected_numbers:
                if num in generated_numbers:
                    matched_numbers += 1
            number_score = matched_numbers / len(expected_numbers)
        
        # 전체 정확도 (키워드 70%, 숫자 30%)
        total_accuracy = keyword_score * 0.7 + number_score * 0.3
        
        return min(total_accuracy, 1.0)
    
    def _update_performance_stats(self, test_result: Dict[str, Any]):
        """성능 통계 업데이트"""
        self.performance_stats['total_tests'] += 1
        
        if test_result['success']:
            self.performance_stats['passed_tests'] += 1
        else:
            self.performance_stats['failed_tests'] += 1
        
        # 평균 검색 시간 업데이트
        search_time = test_result['search_performance']['search_time']
        n = self.performance_stats['total_tests']
        self.performance_stats['avg_search_time'] = (
            (self.performance_stats['avg_search_time'] * (n - 1) + search_time) / n
        )
        
        # 평균 정확도 점수 업데이트
        accuracy_score = test_result['accuracy_score']
        self.performance_stats['avg_accuracy_score'] = (
            (self.performance_stats['avg_accuracy_score'] * (n - 1) + accuracy_score) / n
        )
    
    def _analyze_results(self) -> Dict[str, Any]:
        """결과 분석"""
        if not self.test_results:
            return {}
        
        # 성공률 분석
        success_rate = self.performance_stats['passed_tests'] / self.performance_stats['total_tests']
        
        # 검증 결과 분석
        validation_accurate = sum(1 for r in self.test_results if r['validation']['is_accurate'])
        validation_rate = validation_accurate / len(self.test_results)
        
        # 향상된 검색 성능 분석
        avg_candidates = np.mean([r['enhanced_search']['total_candidates'] for r in self.test_results])
        avg_enhanced_confidence = np.mean([r['enhanced_search']['avg_confidence'] for r in self.test_results])
        
        # 문제점 분석
        common_issues = {}
        for result in self.test_results:
            for issue in result['validation']['issues']:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        # 개선 제안 분석
        common_suggestions = {}
        for result in self.test_results:
            for suggestion in result['validation']['suggestions']:
                common_suggestions[suggestion] = common_suggestions.get(suggestion, 0) + 1
        
        return {
            'success_rate': success_rate,
            'validation_accuracy_rate': validation_rate,
            'enhanced_search_performance': {
                'avg_candidates': avg_candidates,
                'avg_confidence': avg_enhanced_confidence
            },
            'common_issues': dict(sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:5]),
            'common_suggestions': dict(sorted(common_suggestions.items(), key=lambda x: x[1], reverse=True)[:5]),
            'performance_metrics': {
                'avg_search_time': self.performance_stats['avg_search_time'],
                'avg_accuracy_score': self.performance_stats['avg_accuracy_score']
            }
        }
    
    def save_results(self, output_path: str):
        """결과 저장"""
        results = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'performance_stats': self.performance_stats
            },
            'detailed_results': self.test_results,
            'analysis': self._analyze_results()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"테스트 결과 저장 완료: {output_path}")

def main():
    """메인 함수"""
    # 테스트 데이터 로드
    test_data_path = "data/tests/gosan_qa.json"
    
    if not os.path.exists(test_data_path):
        logger.error(f"테스트 데이터 파일을 찾을 수 없습니다: {test_data_path}")
        return
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 테스트 시스템 초기화 및 실행
    test_system = EnhancedTestSystem()
    results = test_system.run_comprehensive_test(test_data)
    
    # 결과 저장
    output_path = "data/tests/enhanced_test_results.json"
    test_system.save_results(output_path)
    
    # 결과 출력
    print("\n=== 향상된 테스트 결과 ===")
    print(f"총 테스트: {results['test_summary']['total_tests']}개")
    print(f"성공률: {results['analysis']['success_rate']:.2%}")
    print(f"검증 정확도: {results['analysis']['validation_accuracy_rate']:.2%}")
    print(f"평균 검색 시간: {results['analysis']['performance_metrics']['avg_search_time']:.3f}초")
    print(f"평균 정확도 점수: {results['analysis']['performance_metrics']['avg_accuracy_score']:.3f}")
    
    print("\n=== 주요 문제점 ===")
    for issue, count in list(results['analysis']['common_issues'].items())[:3]:
        print(f"- {issue}: {count}회")
    
    print("\n=== 주요 개선 제안 ===")
    for suggestion, count in list(results['analysis']['common_suggestions'].items())[:3]:
        print(f"- {suggestion}: {count}회")

if __name__ == "__main__":
    import numpy as np
    main()
