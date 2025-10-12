#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
평가 모드 테스트 스크립트

일반 모드 vs 평가 모드의 답변 차이를 비교합니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.types import Chunk, RetrievedSpan
from modules.generation.prompt_builder import PromptBuilder


def test_prompt_comparison():
    """일반 모드 vs 평가 모드 프롬프트 비교"""
    
    # 테스트 데이터
    question = "AI 플랫폼의 기본 관리자 계정은 무엇인가요?"
    
    # 가상의 컨텍스트
    chunk = Chunk(
        doc_id="test",
        filename="test.pdf",
        page=1,
        start_offset=0,
        length=100,
        text="기본 관리자 계정의 아이디는 KWATER이고 비밀번호는 KWATER입니다. "
             "AI 플랫폼 접속 주소는 http://waio-portal-vip:10011 입니다."
    )
    
    contexts = [RetrievedSpan(chunk=chunk, source="test", score=0.95, rank=1)]
    
    # 1. 일반 모드 프롬프트
    print("=" * 80)
    print("1️⃣  일반 모드 프롬프트")
    print("=" * 80)
    
    normal_builder = PromptBuilder(evaluation_mode=False)
    normal_prompt = normal_builder.build_qa_prompt(
        question=question,
        contexts=contexts,
        question_type="system_info"
    )
    
    print(normal_prompt)
    
    # 2. 평가 모드 프롬프트
    print("\n\n" + "=" * 80)
    print("2️⃣  평가 모드 프롬프트")
    print("=" * 80)
    
    eval_builder = PromptBuilder(evaluation_mode=True)
    eval_prompt = eval_builder.build_qa_prompt(
        question=question,
        contexts=contexts,
        question_type="system_info"
    )
    
    print(eval_prompt)
    
    # 3. 차이점 분석
    print("\n\n" + "=" * 80)
    print("📊 주요 차이점")
    print("=" * 80)
    
    print("\n일반 모드:")
    print("  - 간결하고 자연스러운 대화체 강조")
    print("  - 핵심 정보만 포함")
    print("  - 사용자 친화적")
    
    print("\n평가 모드:")
    print("  - 모든 관련 정보 나열")
    print("  - 단위 무조건 포함")
    print("  - 키워드 최대한 포함")
    print("  - 평가 점수 최적화")
    
    print("\n" + "=" * 80)
    print("✅ 프롬프트 비교 완료!")
    print("=" * 80)


def main():
    """메인 실행"""
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("\n" + "=" * 80)
    print("🧪 평가 모드 테스트")
    print("=" * 80)
    print("\n일반 모드와 평가 모드의 프롬프트 차이를 확인합니다.\n")
    
    test_prompt_comparison()
    
    print("\n💡 다음 단계:")
    print("  1. 평가 벤치마크 실행: python scripts/evaluate_qa_unified.py")
    print("  2. 결과 확인: out/benchmarks/qa_unified_result_summary.txt")
    print("  3. 일반 모드와 비교하여 점수 개선 확인")
    print()


if __name__ == "__main__":
    main()

