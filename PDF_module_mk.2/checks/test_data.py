#!/usr/bin/env python3
"""데이터 상태 테스트 스크립트"""

from core.pdf_preprocessor import PDFDatabase
from core.fast_vector_store import FastVectorStore
from core.question_analyzer import QuestionAnalyzer

def test_database():
    """데이터베이스 상태 확인"""
    print("="*50)
    print("📊 데이터베이스 상태 확인")
    print("="*50)
    
    db = PDFDatabase()
    stats = db.get_statistics()
    
    print(f"처리된 PDF: {stats['total_files']}개")
    print(f"총 청크: {stats['total_chunks']}개")
    print(f"총 페이지: {stats['total_pages']}개")
    
    if stats['files']:
        print("\n처리된 파일들:")
        for file_info in stats['files']:
            print(f"  - {file_info['filename']} ({file_info['total_chunks']}청크)")
    
    # 실제 청크 로드 테스트
    print("\n📚 청크 로드 테스트...")
    chunks = db.load_all_chunks()
    print(f"로드된 청크 수: {len(chunks)}")
    
    if chunks:
        # 첫 번째 청크 확인
        first_chunk = chunks[0]
        print(f"\n첫 번째 청크:")
        print(f"  ID: {first_chunk.chunk_id}")
        print(f"  페이지: {first_chunk.page_number}")
        print(f"  내용: {first_chunk.content[:150]}...")
        print(f"  임베딩: {'있음' if first_chunk.embedding is not None else '없음'}")
        
        # 여과 관련 청크 찾기
        filtration_chunks = []
        for chunk in chunks[:50]:  # 처음 50개만 확인
            if '여과' in chunk.content:
                filtration_chunks.append(chunk)
        
        print(f"\n'여과' 관련 청크: {len(filtration_chunks)}개 발견")
        if filtration_chunks:
            print("첫 번째 여과 관련 청크:")
            print(f"  페이지: {filtration_chunks[0].page_number}")
            print(f"  내용: {filtration_chunks[0].content[:200]}...")
    
    return db, chunks

def test_vector_store(db):
    """벡터 저장소 테스트"""
    print("\n" + "="*50)
    print("🔍 벡터 저장소 테스트")
    print("="*50)
    
    vector_store = FastVectorStore()
    success = vector_store.load_from_database(db)
    
    print(f"벡터 저장소 로드: {'성공' if success else '실패'}")
    
    if success:
        stats = vector_store.get_statistics()
        print(f"로드된 청크: {stats['total_chunks']}개")
        print(f"인덱스 훈련됨: {stats['index_trained']}")
        
        # 검색 테스트
        print("\n🔍 검색 테스트...")
        analyzer = QuestionAnalyzer()
        
        test_questions = [
            "여과 공정",
            "침전 공정", 
            "정수 처리",
            "시스템"
        ]
        
        for question in test_questions:
            analyzed = analyzer.analyze_question(question)
            results = vector_store.search(analyzed.embedding, top_k=3, score_threshold=0.0)
            
            print(f"\n질문: '{question}'")
            print(f"검색 결과: {len(results)}개")
            
            if results:
                for i, (chunk, score) in enumerate(results[:2]):
                    print(f"  {i+1}. 점수: {score:.3f}")
                    print(f"     페이지: {chunk.page_number}")
                    print(f"     내용: {chunk.content[:100]}...")
            else:
                print("  검색 결과 없음")
    
    return vector_store

def test_full_qa():
    """전체 QA 시스템 테스트"""
    print("\n" + "="*50)
    print("🤖 전체 QA 시스템 테스트")
    print("="*50)
    
    from korean_qa import KoreanQASystem
    
    system = KoreanQASystem()
    
    test_questions = [
        "여과 공정에 대해서 설명해줘",
        "침전 공정이 뭐야?",
        "정수장의 주요 공정은?"
    ]
    
    for question in test_questions:
        print(f"\n질문: {question}")
        result = system.ask_question(question)
        print(f"답변: {result['answer'][:200]}...")
        print(f"신뢰도: {result['confidence']:.2f}")
        print(f"사용된 청크: {result.get('used_chunks', 0)}개")

if __name__ == "__main__":
    # 1. 데이터베이스 테스트
    db, chunks = test_database()
    
    # 2. 벡터 저장소 테스트
    vector_store = test_vector_store(db)
    
    # 3. 전체 QA 테스트
    test_full_qa()


