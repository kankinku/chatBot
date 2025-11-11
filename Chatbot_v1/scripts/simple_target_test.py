#!/usr/bin/env python3
"""
간단한 목표 기반 시스템 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_question_analyzer():
    """질문 분석기 간단 테스트"""
    print("질문 분석기 테스트 시작...")
    
    try:
        from core.query.question_analyzer import QuestionAnalyzer
        
        analyzer = QuestionAnalyzer()
        
        test_questions = [
            "여과 공정에서 AI 모델의 목표는 무엇인가요?",
            "N-beats 모델의 성능은 어떤가요?",
            "EMS에서 펌프 세부 현황을 확인할 수 있나요?"
        ]
        
        for question in test_questions:
            print(f"\n질문: {question}")
            analyzed = analyzer.analyze_question(question)
            
            print(f"  - 답변 목표: {analyzed.answer_target}")
            print(f"  - 목표 유형: {analyzed.target_type}")
            print(f"  - 신뢰도: {analyzed.confidence_score:.2f}")
        
        print("\n질문 분석기 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"질문 분석기 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store():
    """벡터 저장소 간단 테스트"""
    print("\n벡터 저장소 테스트 시작...")
    
    try:
        from core.document.vector_store import HybridVectorStore
        from core.document.pdf_processor import TextChunk
        import numpy as np
        
        # 벡터 저장소 초기화
        vector_store = HybridVectorStore()
        
        # 간단한 테스트 청크 생성
        test_chunk = TextChunk(
            content="여과 공정에서 AI 모델은 여과지 세척 주기와 운전 스케줄을 최적화하는 것을 목표로 합니다.",
            page_number=1,
            chunk_id="test_chunk_1",
            metadata={"source": "test"}
        )
        test_chunk.embedding = np.random.rand(768).astype(np.float32)
        
        # 청크 추가
        vector_store.add_chunks([test_chunk])
        
        # 검색 테스트
        query_embedding = np.random.rand(768).astype(np.float32)
        results = vector_store.search(query_embedding, top_k=1)
        
        print(f"검색 결과 수: {len(results)}")
        if results:
            chunk, score = results[0]
            print(f"검색 점수: {score:.3f}")
            print(f"청크 내용: {chunk.content[:50]}...")
        
        print("벡터 저장소 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"벡터 저장소 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("간단한 목표 기반 시스템 테스트")
    print("=" * 50)
    
    # 1. 질문 분석기 테스트
    analyzer_ok = test_question_analyzer()
    
    # 2. 벡터 저장소 테스트
    vector_ok = test_vector_store()
    
    print("\n" + "=" * 50)
    if analyzer_ok and vector_ok:
        print("모든 테스트 통과!")
    else:
        print("일부 테스트 실패")
    print("=" * 50)

if __name__ == "__main__":
    main()
