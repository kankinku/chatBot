"""
벤치마크 결과를 보기 쉽게 출력하는 스크립트
"""
import json
import sys


def view_results(result_file="benchmark_results.json"):
    """벤치마크 결과를 보기 쉽게 출력합니다."""
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = data['summary']
    results = data['results']
    
    # 요약 출력
    print("=" * 80)
    print("벤치마크 결과 요약")
    print("=" * 80)
    print(f"실행 시간: {summary['timestamp']}")
    print(f"\n📊 기본 정보")
    print(f"  총 질문 수: {summary['total_questions']}개")
    print(f"  성공: {summary['successful_answers']}개 | 실패: {summary.get('failed_answers', 0)}개")
    print(f"  성공률: {summary['success_rate']:.1f}%")
    
    if 'average_scores' in summary:
        scores = summary['average_scores']
        print(f"\n📊 평가 결과 요약:")
        print(f"\n1️⃣  기본 Score (v5):        {scores.get('basic_score', 0)*100:6.1f}%")
        print(f"2️⃣  도메인 특화 종합:        {scores.get('domain_score', 0)*100:6.1f}%")
        print(f"    - 숫자 정확도:          {scores.get('numeric_accuracy', 0)*100:6.1f}%")
        print(f"    - 단위 정확도:          {scores.get('unit_accuracy', 0)*100:6.1f}%")
        print(f"3️⃣  RAG 핵심 지표:")
        print(f"    - Faithfulness:         {scores.get('faithfulness', 0)*100:6.1f}%")
        print(f"    - Answer Correctness:   {scores.get('answer_correctness', 0)*100:6.1f}%")
        print(f"    - Context Precision:    {scores.get('context_precision', 0)*100:6.1f}%")
        print(f"4️⃣  학술 표준:")
        print(f"    - Token F1:             {scores.get('token_f1', 0)*100:6.1f}%")
        print(f"    - ROUGE-L:              {scores.get('rouge_l', 0)*100:6.1f}%")
    else:
        # 구버전 호환성
        print(f"\n🎯 평가 점수")
        print(f"  키워드 정확도: {summary.get('overall_keyword_accuracy', 0):.1f}%")
    
    print(f"\n⏱️  성능")
    print(f"  총 소요 시간: {summary['total_time']:.2f}초")
    print(f"  평균 응답 시간: {summary['average_time_per_question']:.2f}초")
    print("=" * 80)
    
    # 기본 Score별 분류
    if 'evaluation' in results[0]:
        eval_key = 'basic_score' if 'basic_score' in results[0]['evaluation'] else 'composite_score'
        high_score = [r for r in results if r.get('success', False) and r['evaluation'].get(eval_key, 0) >= 0.8]
        medium_score = [r for r in results if r.get('success', False) and 0.5 <= r['evaluation'].get(eval_key, 0) < 0.8]
        low_score = [r for r in results if r.get('success', False) and r['evaluation'].get(eval_key, 0) < 0.5]
        failed = [r for r in results if not r.get('success', False)]
        
        print(f"\n기본 Score 분포:")
        print(f"  높음 (80% 이상): {len(high_score)}개")
        print(f"  중간 (50-80%): {len(medium_score)}개")
        print(f"  낮음 (50% 미만): {len(low_score)}개")
        print(f"  실패: {len(failed)}개")
    else:
        # 구버전 호환성
        high_accuracy = [r for r in results if r.get('success', False) and r.get('keyword_accuracy', 0) >= 80]
        medium_accuracy = [r for r in results if r.get('success', False) and 50 <= r.get('keyword_accuracy', 0) < 80]
        low_accuracy = [r for r in results if r.get('success', False) and r.get('keyword_accuracy', 0) < 50]
        failed = [r for r in results if not r.get('success', False)]
        
        print(f"\n키워드 정확도 분포:")
        print(f"  높음 (80% 이상): {len(high_accuracy)}개")
        print(f"  중간 (50-80%): {len(medium_accuracy)}개")
        print(f"  낮음 (50% 미만): {len(low_accuracy)}개")
        print(f"  실패: {len(failed)}개")
        
        low_score = low_accuracy  # 하위 호환성
    
    # 상세 결과 출력 여부 확인
    print("\n상세 결과를 보시겠습니까?")
    print("1. 전체 결과")
    print("2. 낮은 점수 항목만 (50% 미만)")
    print("3. 실패한 항목만")
    print("4. 종료")
    
    choice = input("\n선택 (1-4): ").strip()
    
    if choice == "1":
        show_results = results
        print("\n" + "=" * 80)
        print("전체 결과")
        print("=" * 80)
    elif choice == "2":
        show_results = low_score
        print("\n" + "=" * 80)
        print("낮은 점수 항목 (50% 미만)")
        print("=" * 80)
    elif choice == "3":
        show_results = failed
        print("\n" + "=" * 80)
        print("실패한 항목")
        print("=" * 80)
    else:
        return
    
    # 상세 결과 출력
    for idx, result in enumerate(show_results):
        print(f"\n[{idx + 1}] ID: {result['id']}")
        print(f"질문: {result['question']}")
        
        if 'evaluation' in result:
            eval_data = result['evaluation']
            
            if 'basic_score' in eval_data:
                # 새 버전
                print(f"기본 Score: {eval_data['basic_score']*100:.1f}% | " 
                      f"도메인: {eval_data.get('domain_score', 0)*100:.1f}% | "
                      f"RAG: {eval_data.get('rag_overall', 0)*100:.1f}%")
                print(f"  - 키워드: {eval_data['keyword_accuracy']*100:.1f}% | "
                      f"토큰 F1: {eval_data['token_f1']*100:.1f}% | "
                      f"숫자: {eval_data['numeric_accuracy']*100:.1f}%")
                print(f"  - Faithfulness: {eval_data.get('faithfulness', 0)*100:.1f}% | "
                      f"Context Precision: {eval_data.get('context_precision', 0)*100:.1f}%")
            else:
                # 구 버전
                print(f"종합 점수: {eval_data.get('composite_score', 0)*100:.1f}%")
                print(f"  - 키워드 정확도: {eval_data['keyword_accuracy']*100:.1f}%")
                print(f"  - 토큰 F1: {eval_data['token_f1']*100:.1f}%")
                print(f"  - 숫자 정확도: {eval_data['numeric_accuracy']*100:.1f}%")
            
            if eval_data.get('keyword_matched'):
                print(f"매칭된 키워드: {', '.join(eval_data['keyword_matched'])}")
        else:
            # 구버전 호환성
            print(f"키워드 정확도: {result.get('keyword_accuracy', 0):.1f}%")
            if result.get('matched_keywords'):
                print(f"매칭된 키워드: {', '.join(result['matched_keywords'])}")
        
        if result.get('success', False):
            print(f"검색된 출처: {', '.join(result.get('retrieved_sources', []))}")
            print(f"응답 시간: {result['elapsed_time']:.2f}초")
            print(f"\n기대 답변:")
            print(f"  {result['expected_answer']}")
            print(f"\n생성된 답변:")
            answer = result.get('generated_answer', '')
            print(f"  {answer[:200]}..." if len(answer) > 200 else f"  {answer}")
        else:
            print(f"오류: {result.get('error', 'Unknown error')}")
        
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        view_results(sys.argv[1])
    else:
        view_results()
