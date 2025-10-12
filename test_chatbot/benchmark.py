"""
qa.json의 질문과 답변을 사용한 RAG 시스템 벤치마크
다양한 평가 지표를 통해 종합적인 성능 평가를 수행합니다.
"""
import json
import time
from datetime import datetime
from rag_query import RAGSystem
from evaluation_metrics import EvaluationMetrics


def load_qa_data(qa_file="qa.json"):
    """QA 데이터를 로드합니다."""
    with open(qa_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_benchmark(rag_system, qa_data, output_file="benchmark_results.json"):
    """벤치마크를 실행합니다."""
    print("=" * 80)
    print("RAG 시스템 벤치마크 시작")
    print("=" * 80)
    print(f"총 {len(qa_data)}개의 질문으로 테스트합니다.\n")
    
    results = []
    total_time = 0
    evaluator = EvaluationMetrics()
    
    for idx, qa in enumerate(qa_data):
        question_id = qa['id']
        question = qa['question']
        expected_answer = qa['answer']
        keywords = qa.get('accepted_keywords', [])
        
        print(f"\n[{idx + 1}/{len(qa_data)}] ID: {question_id}")
        print(f"질문: {question}")
        
        # 시간 측정 시작
        start_time = time.time()
        
        try:
            # RAG 시스템으로 답변 생성
            generated_answer, docs, metas = rag_system.query(question, top_k=3)
            elapsed_time = time.time() - start_time
            
            # 컨텍스트 추출 (검색된 문서들)
            contexts = docs if docs else []
            
            # 종합 평가 수행
            eval_results = evaluator.evaluate_answer(
                generated_answer,
                expected_answer,
                keywords,
                contexts
            )
            
            print(f"답변 생성 완료 ({elapsed_time:.2f}초)")
            print(f"기본 Score: {eval_results['basic_score']*100:.1f}%")
            print(f"도메인 특화: {eval_results['domain_score']*100:.1f}%")
            print(f"RAG 종합: {eval_results['rag_overall']*100:.1f}%")
            print(f"  - Faithfulness: {eval_results['faithfulness']*100:.1f}%")
            print(f"  - Answer Correctness: {eval_results['answer_correctness']*100:.1f}%")
            
            result = {
                "id": question_id,
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "keywords": keywords,
                "elapsed_time": elapsed_time,
                "retrieved_sources": [meta['source'] for meta in metas],
                "success": True,
                
                # 평가 지표
                "evaluation": {
                    # 종합 점수
                    "basic_score": eval_results['basic_score'],
                    "domain_score": eval_results['domain_score'],
                    "rag_overall": eval_results['rag_overall'],
                    
                    # 키워드
                    "keyword_accuracy": eval_results['keyword']['accuracy'],
                    "keyword_matched": eval_results['keyword']['matched_keywords'],
                    
                    # 토큰
                    "token_f1": eval_results['token_overlap']['f1'],
                    "token_precision": eval_results['token_overlap']['precision'],
                    "token_recall": eval_results['token_overlap']['recall'],
                    
                    # 숫자 & 단위
                    "numeric_accuracy": eval_results['numeric']['accuracy'],
                    "numeric_matched": eval_results['numeric']['matched_numbers'],
                    "unit_accuracy": eval_results['unit']['accuracy'],
                    "unit_matched": eval_results['unit']['matched_units'],
                    
                    # RAG 지표
                    "faithfulness": eval_results['faithfulness'],
                    "answer_correctness": eval_results['answer_correctness'],
                    "context_precision": eval_results['context_precision'],
                    
                    # 텍스트 유사도
                    "bleu_2": eval_results['text_similarity']['bleu_2'],
                    "rouge_l": eval_results['text_similarity']['rouge_l'],
                    "exact_match": eval_results['text_similarity']['exact_match'],
                    "contains_match": eval_results['text_similarity']['contains_match']
                }
            }
            
            total_time += elapsed_time
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            elapsed_time = time.time() - start_time
            
            result = {
                "id": question_id,
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": None,
                "keywords": keywords,
                "elapsed_time": elapsed_time,
                "retrieved_sources": [],
                "success": False,
                "error": str(e),
                "evaluation": {
                    "basic_score": 0.0,
                    "domain_score": 0.0,
                    "rag_overall": 0.0,
                    "keyword_accuracy": 0.0,
                    "token_f1": 0.0,
                    "numeric_accuracy": 0.0,
                    "unit_accuracy": 0.0,
                    "faithfulness": 0.0,
                    "answer_correctness": 0.0,
                    "context_precision": 0.0
                }
            }
            
            total_time += elapsed_time
        
        results.append(result)
    
    # 전체 통계 계산
    valid_results = [r for r in results if r['success']]
    success_count = len(valid_results)
    success_rate = (success_count / len(qa_data) * 100) if qa_data else 0
    avg_time = total_time / len(qa_data) if qa_data else 0
    
    # 평가 지표 평균 계산
    if valid_results:
        avg_basic = sum(r['evaluation']['basic_score'] for r in valid_results) / len(valid_results)
        avg_domain = sum(r['evaluation']['domain_score'] for r in valid_results) / len(valid_results)
        avg_rag = sum(r['evaluation']['rag_overall'] for r in valid_results) / len(valid_results)
        
        avg_keyword = sum(r['evaluation']['keyword_accuracy'] for r in valid_results) / len(valid_results)
        avg_token_f1 = sum(r['evaluation']['token_f1'] for r in valid_results) / len(valid_results)
        avg_numeric = sum(r['evaluation']['numeric_accuracy'] for r in valid_results) / len(valid_results)
        avg_unit = sum(r['evaluation']['unit_accuracy'] for r in valid_results) / len(valid_results)
        
        avg_faithfulness = sum(r['evaluation']['faithfulness'] for r in valid_results) / len(valid_results)
        avg_correctness = sum(r['evaluation']['answer_correctness'] for r in valid_results) / len(valid_results)
        avg_context_prec = sum(r['evaluation']['context_precision'] for r in valid_results) / len(valid_results)
        
        avg_bleu = sum(r['evaluation']['bleu_2'] for r in valid_results) / len(valid_results)
        avg_rouge = sum(r['evaluation']['rouge_l'] for r in valid_results) / len(valid_results)
    else:
        avg_basic = avg_domain = avg_rag = 0.0
        avg_keyword = avg_token_f1 = avg_numeric = avg_unit = 0.0
        avg_faithfulness = avg_correctness = avg_context_prec = 0.0
        avg_bleu = avg_rouge = 0.0
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(qa_data),
        "successful_answers": success_count,
        "failed_answers": len(qa_data) - success_count,
        "success_rate": success_rate,
        "total_time": total_time,
        "average_time_per_question": avg_time,
        
        # 평가 지표 평균
        "average_scores": {
            # 종합 점수
            "basic_score": avg_basic,
            "domain_score": avg_domain,
            "rag_overall": avg_rag,
            
            # 상세 지표
            "keyword_accuracy": avg_keyword,
            "token_f1": avg_token_f1,
            "numeric_accuracy": avg_numeric,
            "unit_accuracy": avg_unit,
            
            # RAG 지표
            "faithfulness": avg_faithfulness,
            "answer_correctness": avg_correctness,
            "context_precision": avg_context_prec,
            
            # 텍스트 유사도
            "bleu_2": avg_bleu,
            "rouge_l": avg_rouge
        }
    }
    
    # 결과 저장
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 요약 출력
    print("\n" + "=" * 80)
    print("벤치마크 결과 요약")
    print("=" * 80)
    print(f"\n📊 기본 정보")
    print(f"  총 질문 수: {len(qa_data)}개")
    print(f"  성공: {success_count}개 | 실패: {len(qa_data) - success_count}개")
    print(f"  성공률: {success_rate:.1f}%")
    
    print(f"\n📊 평가 결과 요약:")
    print(f"\n1️⃣  기본 Score (v5):        {avg_basic*100:6.1f}%")
    print(f"2️⃣  도메인 특화 종합:        {avg_domain*100:6.1f}%")
    print(f"    - 숫자 정확도:          {avg_numeric*100:6.1f}%")
    print(f"    - 단위 정확도:          {avg_unit*100:6.1f}%")
    print(f"3️⃣  RAG 핵심 지표:")
    print(f"    - Faithfulness:         {avg_faithfulness*100:6.1f}%")
    print(f"    - Answer Correctness:   {avg_correctness*100:6.1f}%")
    print(f"    - Context Precision:    {avg_context_prec*100:6.1f}%")
    print(f"4️⃣  학술 표준:")
    print(f"    - Token F1:             {avg_token_f1*100:6.1f}%")
    print(f"    - ROUGE-L:              {avg_rouge*100:6.1f}%")
    
    print(f"\n⏱️  성능")
    print(f"  총 소요 시간: {total_time:.2f}초")
    print(f"  평균 응답 시간: {avg_time:.2f}초")
    print(f"\n결과가 '{output_file}'에 저장되었습니다.")
    print("=" * 80)
    
    return output_data


def main():
    # QA 데이터 로드
    print("QA 데이터 로드 중...")
    qa_data = load_qa_data("qa.json")
    print(f"{len(qa_data)}개의 질문-답변 쌍을 로드했습니다.\n")
    
    # RAG 시스템 초기화
    print("RAG 시스템 초기화 중...")
    rag = RAGSystem()
    
    # 벤치마크 실행
    results = run_benchmark(rag, qa_data)
    
    print("\n벤치마크가 완료되었습니다!")


if __name__ == "__main__":
    main()

