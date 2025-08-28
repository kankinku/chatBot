#!/usr/bin/env python3
"""
키워드 인식률 향상 기능 테스트 스크립트 (요약형)
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.question_analyzer import QuestionAnalyzer
from utils.keyword_enhancer import KeywordEnhancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_keyword_enhancement():
    print("=" * 60)
    print("키워드 인식률 향상 기능 테스트")
    print("=" * 60)
    
    enhancer = KeywordEnhancer(domain="technical")
    test_questions = [
        "시스템 API 성능을 어떻게 개선할 수 있나요?",
        "데이터베이스 백업 절차는 무엇인가요?",
        "사용자 인증 보안을 강화하는 방법은?",
        "네트워크 연결 문제를 해결하는 단계는?",
        "소프트웨어 배포 과정에서 발생하는 오류는?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n질문 {i}: {question}")
        basic_keywords = [kw for kw in question.lower().split() if len(kw) > 2]
        enhanced_keywords = enhancer.enhance_keywords(basic_keywords)
        domain_keywords = enhancer.extract_domain_specific_keywords(question)
        print(f"  기본 키워드: {basic_keywords}")
        print(f"  향상된 키워드: {enhanced_keywords[:10]}")
        print(f"  도메인 키워드: {domain_keywords}")

def test_question_analyzer_improvement():
    print("\n\n2. 개선된 QuestionAnalyzer 테스트")
    print("-" * 40)
    general_analyzer = QuestionAnalyzer(domain="general")
    technical_analyzer = QuestionAnalyzer(domain="technical")
    test_questions = [
        "API 성능 최적화 방법은?",
        "데이터베이스 백업 정책은?",
        "사용자 인증 보안 강화 방안은?",
        "네트워크 연결 문제 해결책은?",
        "소프트웨어 배포 오류 처리 방법은?",
    ]
    for i, question in enumerate(test_questions, 1):
        print(f"\n질문 {i}: {question}")
        general_result = general_analyzer.analyze_question(question)
        technical_result = technical_analyzer.analyze_question(question)
        print(f"  일반 도메인 키워드: {general_result.keywords[:8]}")
        print(f"  기술 도메인 키워드: {technical_result.keywords[:8]}")
        print(f"  키워드 개수: 일반 {len(general_result.keywords)} vs 기술 {len(technical_result.keywords)}")

def main():
    try:
        test_keyword_enhancement()
        test_question_analyzer_improvement()
        print("\n" + "=" * 60)
        print("키워드 인식률 향상 테스트 완료!")
        print("=" * 60)
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


