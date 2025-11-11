"""
향상된 PDF 시스템 사용 예제

법률 파이프라인 기법을 적용한 PDF RAG 시스템의 사용법을 보여줍니다:
- 하이브리드 검색 (벡터 + BM25 + RRF)
- 크로스엔코더 재순위화
- 멀티뷰 인덱싱
- 성능 최적화
"""

import sys
import os
import time
from typing import List, Dict, Any

# 워크스페이스 루트를 파이썬 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.query.enhanced_query_router import create_enhanced_query_router
from core.document.pdf_processor import TextChunk
from core.document.pdf_router import PDFMode

def create_sample_chunks() -> List[TextChunk]:
    """샘플 PDF 청크 생성"""
    chunks = [
        TextChunk(
            content="정수처리 공정은 원수에서 불순물을 제거하여 안전한 음용수를 생산하는 과정입니다. 주요 공정으로는 응집, 침전, 여과, 소독이 있습니다.",
            page_number=1,
            chunk_id="chunk_1",
            metadata={
                'title': '정수처리 기본 원리',
                'section': '개요',
                'keywords': ['정수처리', '공정', '원수', '불순물', '음용수']
            }
        ),
        TextChunk(
            content="수질 기준을 초과했을 때는 즉시 원수 공급을 중단하고, 응급처리 시설을 가동해야 합니다. 또한 관련 기관에 신고하고 시민들에게 공지를 해야 합니다.",
            page_number=2,
            chunk_id="chunk_2",
            metadata={
                'title': '수질 기준 초과 시 대응',
                'section': '응급대응',
                'keywords': ['수질기준', '초과', '원수공급', '중단', '응급처리', '신고']
            }
        ),
        TextChunk(
            content="정수장 운영 시 주의사항으로는 정기적인 수질 검사, 시설물 점검, 화학약품 관리, 안전사고 예방 등이 있습니다. 특히 화학약품은 안전한 곳에 보관하고 적절한 농도로 사용해야 합니다.",
            page_number=3,
            chunk_id="chunk_3",
            metadata={
                'title': '정수장 운영 주의사항',
                'section': '운영관리',
                'keywords': ['정수장', '운영', '수질검사', '시설물', '점검', '화학약품', '안전']
            }
        ),
        TextChunk(
            content="수질 검사는 일일, 주간, 월간, 분기별로 실시합니다. 일일 검사는 기본적인 수질 항목을, 주간 검사는 미생물 검사를, 월간 검사는 화학성분을, 분기별 검사는 종합적인 수질 분석을 실시합니다.",
            page_number=4,
            chunk_id="chunk_4",
            metadata={
                'title': '수질 검사 주기',
                'section': '검사관리',
                'keywords': ['수질검사', '주기', '일일', '주간', '월간', '분기별', '미생물', '화학성분']
            }
        ),
        TextChunk(
            content="정수처리 시 사용되는 주요 화학약품으로는 응집제(알루미늄 설페이트), 소독제(염소), pH 조절제(석회), 응집보조제(폴리머) 등이 있습니다. 각 약품은 적절한 농도와 투입량을 유지해야 합니다.",
            page_number=5,
            chunk_id="chunk_5",
            metadata={
                'title': '정수처리 화학약품',
                'section': '화학약품관리',
                'keywords': ['화학약품', '응집제', '알루미늄설페이트', '소독제', '염소', 'pH조절제', '석회', '응집보조제', '폴리머']
            }
        )
    ]
    
    return chunks

def test_enhanced_pdf_system():
    """향상된 PDF 시스템 테스트"""
    print("향상된 PDF 시스템 테스트 시작")
    print("="*50)
    
    # 1. 향상된 쿼리 라우터 생성
    print("1. 향상된 쿼리 라우터 초기화...")
    router = create_enhanced_query_router()
    
    # 2. 샘플 청크로 초기화
    print("2. 샘플 PDF 청크 로드...")
    sample_chunks = create_sample_chunks()
    router.initialize_with_chunks(sample_chunks)
    print(f"   로드된 청크 수: {len(sample_chunks)}개")
    
    # 3. 테스트 질의들
    test_queries = [
        "정수처리 공정의 기본 원리는 무엇인가?",
        "수질 기준을 초과했을 때 대응 방법은?",
        "정수장 운영 시 주의사항은 무엇인가?",
        "수질 검사 주기는 어떻게 되나?",
        "정수처리 시 사용되는 화학약품은?"
    ]
    
    print("\n3. 테스트 질의 실행...")
    print("-" * 50)
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n질의 {i}: {query}")
        print("-" * 30)
        
        start_time = time.time()
        result = router.handle_query(query)
        end_time = time.time()
        
        # 결과 출력
        print(f"답변: {result.get('answer', 'N/A')[:200]}...")
        print(f"신뢰도: {result.get('confidence', 0.0):.3f}")
        print(f"소스 수: {len(result.get('sources', []))}개")
        print(f"처리 시간: {result.get('total_processing_time', 0.0):.3f}초")
        
        # 라우팅 정보
        routing_info = result.get('routing_info', {})
        print(f"라우팅: {routing_info.get('route', 'N/A')}")
        print(f"파이프라인 버전: {routing_info.get('pipeline_version', 'N/A')}")
        
        results.append({
            'query': query,
            'result': result,
            'processing_time': end_time - start_time
        })
    
    # 4. 통계 출력
    print("\n4. 테스트 결과 통계")
    print("=" * 50)
    
    total_time = sum(r['processing_time'] for r in results)
    avg_time = total_time / len(results)
    avg_confidence = sum(r['result'].get('confidence', 0.0) for r in results) / len(results)
    
    print(f"총 질의 수: {len(results)}개")
    print(f"평균 처리 시간: {avg_time:.3f}초")
    print(f"평균 신뢰도: {avg_confidence:.3f}")
    print(f"총 처리 시간: {total_time:.3f}초")
    
    # 5. 라우터 통계
    print("\n5. 라우터 통계")
    print("-" * 30)
    stats = router.get_router_stats()
    print(f"임베딩 모델: {stats.get('embedding_model', 'N/A')}")
    print(f"LLM 모델: {stats.get('llm_model', 'N/A')}")
    print(f"향상된 PDF 활성화: {stats.get('enhanced_pdf_enabled', False)}")
    print(f"파이프라인 버전: {stats.get('pipeline_version', 'N/A')}")
    
    if 'enhanced_pdf_stats' in stats:
        pdf_stats = stats['enhanced_pdf_stats']
        print(f"PDF 처리기 설정:")
        print(f"  - 재순위화: {pdf_stats.get('config', {}).get('enable_reranking', False)}")
        print(f"  - 멀티뷰: {pdf_stats.get('config', {}).get('enable_multiview', False)}")
        print(f"  - 최대 결과 수: {pdf_stats.get('config', {}).get('max_results', 0)}")
    
    print("\n향상된 PDF 시스템 테스트 완료!")

def test_different_modes():
    """다른 모드 테스트"""
    print("\n\n다른 모드 테스트")
    print("=" * 50)
    
    # 향상된 라우터 생성
    router = create_enhanced_query_router()
    sample_chunks = create_sample_chunks()
    router.initialize_with_chunks(sample_chunks)
    
    test_query = "정수처리 공정의 기본 원리는 무엇인가?"
    
    # 정확도 모드 테스트
    print("1. 정확도 모드 테스트")
    print("-" * 30)
    
    if router.enhanced_pdf_handler:
        result_accuracy = router.enhanced_pdf_handler.handle_pdf_query(
            test_query, 
            mode=PDFMode.ACCURACY
        )
        print(f"신뢰도: {result_accuracy.get('confidence', 0.0):.3f}")
        print(f"총 시간: {result_accuracy.get('total_time', 0.0):.3f}초")
        print(f"임계값 통과: {result_accuracy.get('search_results', {}).get('threshold_passed', 0)}개")
    
    # 속도 모드 테스트
    print("\n2. 속도 모드 테스트")
    print("-" * 30)
    
    if router.enhanced_pdf_handler:
        result_speed = router.enhanced_pdf_handler.handle_pdf_query(
            test_query, 
            mode=PDFMode.SPEED
        )
        print(f"신뢰도: {result_speed.get('confidence', 0.0):.3f}")
        print(f"총 시간: {result_speed.get('total_time', 0.0):.3f}초")
        print(f"임계값 통과: {result_speed.get('search_results', {}).get('threshold_passed', 0)}개")

if __name__ == "__main__":
    try:
        # 기본 테스트
        test_enhanced_pdf_system()
        
        # 모드별 테스트
        test_different_modes()
        
    except Exception as e:
        print(f"테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

