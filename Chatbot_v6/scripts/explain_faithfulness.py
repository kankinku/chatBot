#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faithfulness 평가 방법 상세 설명

실제 질문 데이터로 어떻게 평가되는지 보여줍니다.
"""

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag_core_metrics import RAGCoreMetrics


def explain_faithfulness():
    """Faithfulness 평가 방법을 실제 예시로 설명"""
    
    print("=" * 80)
    print("Faithfulness (충실성) 평가 방법 상세 설명")
    print("=" * 80)
    print()
    
    # 실제 벤치마크 결과 로드
    result_file = project_root / "out/benchmarks/qa_final_v6.json"
    
    if not result_file.exists():
        print("벤치마크 결과 파일이 없습니다. 먼저 벤치마크를 실행하세요.")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # 예시 1: 높은 Faithfulness
    print("📊 예시 분석 (3개 질문)")
    print("=" * 80)
    
    for idx in [0, 4, 14]:  # 질문 1, 5, 15
        if idx >= len(results):
            continue
            
        r = results[idx]
        question = r['question']
        answer = r['prediction']
        
        # Faithfulness 정보
        fm = r.get('rag_metrics', {}).get('faithfulness', {})
        
        print(f"\n질문 {r['id']}: {question[:60]}...")
        print(f"답변: {answer[:100]}...")
        print()
        print(f"Faithfulness 평가:")
        print(f"  ├─ 답변의 총 문장 수: {fm.get('total_claims', 0)}개")
        print(f"  ├─ 자료에 근거한 문장: {fm.get('supported_claims', 0)}개")
        print(f"  ├─ 근거 없는 문장: {fm.get('total_claims', 0) - fm.get('supported_claims', 0)}개")
        print(f"  └─ 점수: {fm.get('score', 0)*100:.1f}%")
        
        if fm.get('score', 0) >= 0.8:
            print(f"  → 해석: 우수 - 대부분 자료에 근거함")
        elif fm.get('score', 0) >= 0.5:
            print(f"  → 해석: 보통 - 일부 자료 이탈")
        else:
            print(f"  → 해석: 낮음 - 자료 근거 부족, 환각 가능성")
        
        print("-" * 80)
    
    # 전체 통계
    all_faithfulness = [r['rag_metrics']['faithfulness']['score'] for r in results if 'rag_metrics' in r]
    avg_faith = sum(all_faithfulness) / len(all_faithfulness) if all_faithfulness else 0
    
    print()
    print("=" * 80)
    print("📈 전체 통계 (30개 질문)")
    print("=" * 80)
    print()
    print(f"평균 Faithfulness: {avg_faith*100:.1f}%")
    print()
    print("이것은 다음을 의미합니다:")
    print(f"  → 생성된 답변의 문장 중 평균 {avg_faith*100:.1f}%가 참고 자료에 근거함")
    print(f"  → 약 {(1-avg_faith)*100:.1f}%의 문장이 자료 밖 내용 (환각 가능)")
    print()
    
    # 평가 방법 상세 설명
    print("=" * 80)
    print("🔍 Faithfulness 평가 알고리즘")
    print("=" * 80)
    print()
    print("단계 1: 답변을 문장 단위로 분리")
    print("  예) '발주기관은 한국수자원공사입니다. 사업명은 금강 유역입니다.'")
    print("      → 문장 1: '발주기관은 한국수자원공사입니다'")
    print("      → 문장 2: '사업명은 금강 유역입니다'")
    print()
    print("단계 2: 각 문장의 토큰이 참고 자료에 있는지 확인")
    print("  문장 1 토큰: [발주기관, 한국수자원공사]")
    print("  참고 자료 토큰: [발주기관, 한국수자원공사, 사업명, 금강, 유역, ...]")
    print("  → 겹침 비율: 2/2 = 100% ✅")
    print()
    print("  문장 2 토큰: [사업명, 금강, 유역]")
    print("  참고 자료 토큰: [발주기관, 한국수자원공사, ...]")
    print("  → 겹침 비율: 2/3 = 67% ❌ (70% 미만)")
    print()
    print("단계 3: 70% 이상 겹치는 문장만 '지지됨'으로 간주")
    print("  → 지지된 문장: 1개 / 전체 문장: 2개")
    print("  → Faithfulness: 1/2 = 50%")
    print()
    print("=" * 80)
    print("💡 핵심 아이디어")
    print("=" * 80)
    print()
    print("Faithfulness는 '답변이 얼마나 자료에 충실한가'를 측정합니다.")
    print()
    print("높은 점수 (80% 이상):")
    print("  ✅ 답변의 거의 모든 내용이 참고 자료에서 나옴")
    print("  ✅ 환각(거짓 정보 생성) 거의 없음")
    print()
    print("중간 점수 (50~80%):")
    print("  ⚠️  답변의 일부가 자료에 없는 내용 포함")
    print("  ⚠️  LLM이 자체 지식으로 답변한 부분 있음")
    print()
    print("낮은 점수 (<50%):")
    print("  ❌ 답변의 대부분이 자료와 무관")
    print("  ❌ 환각 가능성 높음")
    print()
    print("=" * 80)
    print("🤔 왜 58.3%인가?")
    print("=" * 80)
    print()
    print("v6 챗봇의 경우:")
    print("  - 답변을 '자연스럽게' 생성하도록 설계됨")
    print("  - 자료의 내용을 '요약'하고 '재구성'함")
    print("  - → 자료에 없는 연결어/문맥이 추가됨")
    print()
    print("예시:")
    print("  자료: '발주기관 한국수자원공사, 사업명 금강유역'")
    print("  답변: '발주기관은 한국수자원공사입니다. 또한 사업명은...'")
    print("        ↑ '또한'은 자료에 없음 → Faithfulness 감소")
    print()
    print("개선 방법:")
    print("  1. 자료를 더 직접적으로 인용")
    print("  2. 추가 설명/연결어 최소화")
    print("  3. 추출적(extractive) 답변 강화")
    print()
    print("=" * 80)


if __name__ == "__main__":
    explain_faithfulness()



