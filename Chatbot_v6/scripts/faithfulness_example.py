#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faithfulness 평가 실제 계산 예시

단계별로 어떻게 계산되는지 실제 데이터로 보여줍니다.
"""

import sys
import re

sys.stdout.reconfigure(encoding='utf-8')


def tokenize(text):
    """토큰화"""
    tokens = re.findall(r'[\w]+', text.lower())
    return [t for t in tokens if len(t) > 1]


def show_calculation_step_by_step():
    """단계별 계산 과정"""
    
    print("=" * 80)
    print("Faithfulness 평가 실제 계산 예시")
    print("=" * 80)
    print()
    
    # 실제 예시
    question = "고산 정수장 시스템 사용자 설명서의 발주기관과 사업명은?"
    
    answer = """발주기관은 한국수자원공사입니다. 
사업명은 금강유역(남부) 스마트 정수장 확대구축 용역입니다."""
    
    contexts = [
        "고산 정수장 시스템 사용자 설명서 발주기관: 한국수자원공사 사업명: 금강 유역 남부 스마트 정수장 확대 구축 용역",
        "작성: (주)에셈블 컨소시엄 발행일: 2025.02.17"
    ]
    
    print(f"질문: {question}")
    print(f"\n답변:\n{answer}")
    print(f"\n참고 자료 {len(contexts)}개:")
    for i, ctx in enumerate(contexts, 1):
        print(f"  [{i}] {ctx[:70]}...")
    
    print("\n" + "=" * 80)
    print("📊 단계별 계산 과정")
    print("=" * 80)
    
    # Step 1: 문장 분리
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', answer) if len(s.strip()) > 10]
    
    print(f"\n[단계 1] 답변을 문장으로 분리")
    print(f"  총 {len(sentences)}개 문장:")
    for i, sent in enumerate(sentences, 1):
        print(f"    문장 {i}: '{sent}'")
    
    # Step 2: 참고 자료 토큰화
    combined_context = ' '.join(contexts).lower()
    context_tokens = set(tokenize(combined_context))
    
    print(f"\n[단계 2] 참고 자료 토큰화")
    print(f"  총 {len(context_tokens)}개 토큰 (중복 제거)")
    print(f"  예시: {list(context_tokens)[:15]}...")
    
    # Step 3: 각 문장 평가
    print(f"\n[단계 3] 각 문장이 자료에 근거하는지 평가")
    print()
    
    supported = 0
    
    for i, sentence in enumerate(sentences, 1):
        sentence_tokens = set(tokenize(sentence))
        
        overlap = sentence_tokens & context_tokens
        overlap_ratio = len(overlap) / len(sentence_tokens) if sentence_tokens else 0
        
        is_supported = overlap_ratio >= 0.7
        
        print(f"  문장 {i}: '{sentence[:50]}...'")
        print(f"    ├─ 문장 토큰 수: {len(sentence_tokens)}개")
        print(f"    ├─ 자료에 있는 토큰: {len(overlap)}개")
        print(f"    ├─ 겹침 비율: {overlap_ratio*100:.1f}%")
        print(f"    └─ 판정: {'✅ 지지됨' if is_supported else '❌ 미지지'} (기준: 70%)")
        print()
        
        if is_supported:
            supported += 1
    
    # Step 4: 최종 점수
    faithfulness_score = supported / len(sentences)
    
    print("=" * 80)
    print("[단계 4] 최종 점수 계산")
    print("=" * 80)
    print()
    print(f"  지지된 문장 수: {supported}개")
    print(f"  전체 문장 수:   {len(sentences)}개")
    print(f"  Faithfulness = {supported}/{len(sentences)} = {faithfulness_score*100:.1f}%")
    print()
    
    if faithfulness_score >= 0.8:
        interpretation = "우수 - 답변이 자료에 매우 충실"
    elif faithfulness_score >= 0.5:
        interpretation = "보통 - 일부 자료 이탈 (자연스러운 표현 추가됨)"
    else:
        interpretation = "낮음 - 자료 근거 부족 (환각 가능성)"
    
    print(f"  해석: {interpretation}")
    print()
    
    print("=" * 80)
    print("💡 실무적 의미")
    print("=" * 80)
    print()
    print("Faithfulness 58.3%의 의미:")
    print()
    print("긍정적 해석:")
    print("  ✅ 답변의 절반 이상이 자료에 근거")
    print("  ✅ 완전한 환각은 아님")
    print("  ✅ 자연스러운 답변 생성 (일부 연결어 추가)")
    print()
    print("주의 사항:")
    print("  ⚠️  약 40%의 내용이 자료에 직접 명시되지 않음")
    print("  ⚠️  LLM이 문맥을 추가하거나 재구성함")
    print()
    print("v5가 87%였는데 왜 v6가 94.3%?")
    print("  → Faithfulness와 도메인 점수는 다른 지표입니다!")
    print("  → 도메인 점수(94.3%): 키워드 일치도 (실무 정확도)")
    print("  → Faithfulness(58.3%): 자료 근거도 (환각 방지)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    show_calculation_step_by_step()



