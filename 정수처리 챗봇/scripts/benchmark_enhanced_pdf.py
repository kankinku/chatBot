"""
향상된 PDF 파이프라인 벤치마크 / 오프라인 평가 스켈레톤

법률/정수처리 파이프라인 기법을 적용한 PDF RAG의 성능 측정과
오프라인 품질 평가(nDCG@10, MRR@10, Recall@k, 정답률) 스켈레톤을 제공합니다.
"""

import time
import json
import os
import sys
from typing import List, Dict, Any

# 워크스페이스 루트를 파이썬 경로에 추가
sys.path.insert(0, os.getcwd())

from core.document.enhanced_pdf_pipeline import create_enhanced_pdf_pipeline, create_fast_pdf_pipeline
from core.document.pdf_router import PDFMode
from core.query.enhanced_pdf_handler import create_enhanced_pdf_handler, create_fast_pdf_handler

def run_benchmark(queries: List[str], pipeline_type: str = "enhanced") -> List[Dict[str, Any]]:
    """벤치마크 실행"""
    print(f"벤치마크 시작: {pipeline_type} 파이프라인")
    
    # 파이프라인 초기화
    if pipeline_type == "enhanced":
        pipeline = create_enhanced_pdf_pipeline()
    elif pipeline_type == "fast":
        pipeline = create_fast_pdf_pipeline()
    else:
        raise ValueError(f"지원하지 않는 파이프라인 타입: {pipeline_type}")
    
    results = []
    
    for i, query in enumerate(queries):
        print(f"\n질의 {i+1}/{len(queries)}: {query}")
        
        try:
            # 정확도 모드로 검색
            result = pipeline.search_and_answer(
                query=query,
                mode=PDFMode.ACCURACY,
                max_results=5
            )
            
            # 결과 요약
            summary = {
                'query': query,
                'answer_length': len(result.get('answer', '')),
                'confidence': result.get('confidence', 0.0),
                'sources_count': len(result.get('sources', [])),
                'search_time': result.get('search_results', {}).get('total_search_time', 0.0),
                'generation_time': result.get('generation_time', 0.0),
                'total_time': result.get('total_time', 0.0),
                'threshold_passed': result.get('search_results', {}).get('threshold_passed', 0),
                'pipeline_type': pipeline_type
            }
            
            results.append(summary)
            
            print(f"  답변 길이: {summary['answer_length']}자")
            print(f"  신뢰도: {summary['confidence']:.3f}")
            print(f"  소스 수: {summary['sources_count']}개")
            print(f"  검색 시간: {summary['search_time']:.3f}초")
            print(f"  생성 시간: {summary['generation_time']:.3f}초")
            print(f"  총 시간: {summary['total_time']:.3f}초")
            
        except Exception as e:
            print(f"  오류: {e}")
            results.append({
                'query': query,
                'error': str(e),
                'pipeline_type': pipeline_type
            })
    
    return results

def evaluate_offline(goldenset_path: str) -> Dict[str, Any]:
    """오프라인 평가 스켈레톤: 골든셋 포맷
    [
      {"question": "...", "expected_answer": "...", "relevant_chunk_ids": ["id1", "id2"], "domain": "water"},
      ...
    ]
    """
    try:
        with open(goldenset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # TODO: 파이프라인 호출 및 메트릭 계산 구현
        return {"status": "skipped", "items": len(data)}
    except Exception as e:
        return {"error": str(e)}

def compare_pipelines(queries: List[str]) -> Dict[str, Any]:
    """파이프라인 비교"""
    print("파이프라인 비교 시작...")
    
    # 향상된 파이프라인 테스트
    enhanced_results = run_benchmark(queries, "enhanced")
    
    # 빠른 파이프라인 테스트
    fast_results = run_benchmark(queries, "fast")
    
    # 통계 계산
    enhanced_stats = calculate_stats(enhanced_results)
    fast_stats = calculate_stats(fast_results)
    
    comparison = {
        'enhanced_pipeline': {
            'results': enhanced_results,
            'stats': enhanced_stats
        },
        'fast_pipeline': {
            'results': fast_results,
            'stats': fast_stats
        },
        'comparison': {
            'avg_total_time_enhanced': enhanced_stats['avg_total_time'],
            'avg_total_time_fast': fast_stats['avg_total_time'],
            'avg_confidence_enhanced': enhanced_stats['avg_confidence'],
            'avg_confidence_fast': fast_stats['avg_confidence'],
            'speed_improvement': (enhanced_stats['avg_total_time'] - fast_stats['avg_total_time']) / enhanced_stats['avg_total_time'] * 100,
            'confidence_difference': enhanced_stats['avg_confidence'] - fast_stats['avg_confidence']
        }
    }
    
    return comparison

def calculate_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """통계 계산"""
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return {
            'avg_total_time': 0.0,
            'avg_confidence': 0.0,
            'avg_sources_count': 0.0,
            'success_rate': 0.0,
            'total_queries': len(results)
        }
    
    return {
        'avg_total_time': sum(r.get('total_time', 0.0) for r in valid_results) / len(valid_results),
        'avg_confidence': sum(r.get('confidence', 0.0) for r in valid_results) / len(valid_results),
        'avg_sources_count': sum(r.get('sources_count', 0) for r in valid_results) / len(valid_results),
        'avg_search_time': sum(r.get('search_time', 0.0) for r in valid_results) / len(valid_results),
        'avg_generation_time': sum(r.get('generation_time', 0.0) for r in valid_results) / len(valid_results),
        'success_rate': len(valid_results) / len(results) * 100,
        'total_queries': len(results),
        'successful_queries': len(valid_results)
    }

def print_comparison_summary(comparison: Dict[str, Any]):
    """비교 결과 요약 출력"""
    print("\n" + "="*60)
    print("파이프라인 비교 결과")
    print("="*60)
    
    comp = comparison['comparison']
    
    print(f"향상된 파이프라인:")
    print(f"  평균 총 시간: {comp['avg_total_time_enhanced']:.3f}초")
    print(f"  평균 신뢰도: {comp['avg_confidence_enhanced']:.3f}")
    
    print(f"\n빠른 파이프라인:")
    print(f"  평균 총 시간: {comp['avg_total_time_fast']:.3f}초")
    print(f"  평균 신뢰도: {comp['avg_confidence_fast']:.3f}")
    
    print(f"\n개선 효과:")
    print(f"  속도 개선: {comp['speed_improvement']:.1f}%")
    print(f"  신뢰도 차이: {comp['confidence_difference']:.3f}")
    
    if comp['speed_improvement'] > 0:
        print(f"  → 빠른 파이프라인이 {comp['speed_improvement']:.1f}% 더 빠름")
    else:
        print(f"  → 향상된 파이프라인이 {-comp['speed_improvement']:.1f}% 더 빠름")
    
    if comp['confidence_difference'] > 0:
        print(f"  → 향상된 파이프라인이 {comp['confidence_difference']:.3f} 더 높은 신뢰도")
    else:
        print(f"  → 빠른 파이프라인이 {-comp['confidence_difference']:.3f} 더 높은 신뢰도")

if __name__ == "__main__":
    # 테스트 질의들
    test_queries = [
        "정수처리 공정의 기본 원리는 무엇인가?",
        "수질 기준을 초과했을 때 대응 방법은?",
        "정수장 운영 시 주의사항은 무엇인가?",
        "수질 검사 주기는 어떻게 되나?",
        "정수처리 시 사용되는 화학약품은?"
    ]
    
    print("향상된 PDF 파이프라인 벤치마크/오프라인 평가 스켈레톤 시작")
    print(f"테스트 질의 수: {len(test_queries)}")
    
    try:
        # 파이프라인 비교 실행
        comparison = compare_pipelines(test_queries)
        
        # 결과 출력
        print_comparison_summary(comparison)
        
        # 상세 결과를 JSON으로 저장
        output_file = "benchmark_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        print(f"\n상세 결과가 {output_file}에 저장되었습니다.")
        
        # 오프라인 평가 스켈레톤 실행(옵션)
        try:
            from core.config.unified_config import get_config
        except Exception:
            import os
            def get_config(key, default=None):
                return os.getenv(key, default)
        offline = evaluate_offline(get_config('GOLDENSET_PATH', '')) if get_config('GOLDENSET_PATH') else {"status": "disabled"}
        print(f"오프라인 평가: {offline}")

    except Exception as e:
        print(f"벤치마크 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

