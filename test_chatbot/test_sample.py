"""
벤치마크 전에 몇 개 샘플만 빠르게 테스트하는 스크립트
"""
import json
from rag_query import RAGSystem
from evaluation_metrics import EvaluationMetrics


def test_sample_questions(num_samples=3):
    """샘플 질문으로 빠른 테스트를 수행합니다."""
    
    # QA 데이터 로드
    with open("qa.json", 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # 처음 num_samples개만 선택
    sample_data = qa_data[:num_samples]
    
    print("=" * 80)
    print(f"샘플 테스트 ({num_samples}개 질문)")
    print("=" * 80)
    
    # RAG 시스템 초기화
    print("\nRAG 시스템 초기화 중...")
    rag = RAGSystem()
    evaluator = EvaluationMetrics()
    
    # 샘플 테스트
    for idx, qa in enumerate(sample_data):
        question = qa['question']
        expected_answer = qa['answer']
        keywords = qa.get('accepted_keywords', [])
        
        print(f"\n{'='*80}")
        print(f"[{idx + 1}/{len(sample_data)}] 질문: {question}")
        print(f"{'='*80}")
        
        try:
            answer, docs, metas = rag.query(question, top_k=3)
            
            # 컨텍스트 추출
            contexts = docs if docs else []
            
            # 평가 수행
            eval_results = evaluator.evaluate_answer(answer, expected_answer, keywords, contexts)
            
            print(f"\n📊 평가 점수:")
            print(f"  기본 Score (v5):      {eval_results['basic_score']*100:6.1f}%")
            print(f"  도메인 특화:          {eval_results['domain_score']*100:6.1f}%")
            print(f"  RAG 종합:             {eval_results['rag_overall']*100:6.1f}%")
            print(f"\n  상세:")
            print(f"    키워드 정확도:      {eval_results['keyword']['accuracy']*100:6.1f}%")
            print(f"    토큰 F1:            {eval_results['token_overlap']['f1']*100:6.1f}%")
            print(f"    숫자 정확도:        {eval_results['numeric']['accuracy']*100:6.1f}%")
            print(f"    단위 정확도:        {eval_results['unit']['accuracy']*100:6.1f}%")
            print(f"    Faithfulness:       {eval_results['faithfulness']*100:6.1f}%")
            print(f"    Answer Correctness: {eval_results['answer_correctness']*100:6.1f}%")
            print(f"    Context Precision:  {eval_results['context_precision']*100:6.1f}%")
            print(f"    ROUGE-L:            {eval_results['text_similarity']['rouge_l']*100:6.1f}%")
            
            print(f"\n생성된 답변:")
            print(f"{answer}")
            
            print(f"\n기대 답변:")
            print(f"{expected_answer}")
            
            print(f"\n기대 키워드: {', '.join(keywords)}")
            
            if eval_results['keyword']['matched_keywords']:
                print(f"매칭된 키워드: {', '.join(eval_results['keyword']['matched_keywords'])}")
            else:
                print("매칭된 키워드: 없음")
            
        except Exception as e:
            print(f"\n오류 발생: {str(e)}")
    
    print(f"\n{'='*80}")
    print("샘플 테스트 완료!")
    print("전체 벤치마크를 실행하려면: python benchmark.py")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    num_samples = 3
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            print("사용법: python test_sample.py [샘플 수]")
            sys.exit(1)
    
    test_sample_questions(num_samples)

