#!/usr/bin/env python3
"""
키워드 인식률 향상 기능의 실제 효과 측정 테스트 (실전형)
"""

import sys
import os
import logging
import time
from typing import List, Dict, Tuple

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.question_analyzer import QuestionAnalyzer
from utils.keyword_enhancer import KeywordEnhancer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_keyword_recognition_improvement():
    """키워드 인식률 향상 효과 측정"""
    print("=" * 70)
    print("키워드 인식률 향상 효과 측정 테스트")
    print("=" * 70)
    
    test_questions = [
        "API 성능 최적화 방법은 무엇인가요?",
        "데이터베이스 백업 절차를 어떻게 설정하나요?",
        "사용자 인증 보안을 강화하는 방안은?",
        "네트워크 연결 문제를 해결하는 단계는?",
        "소프트웨어 배포 과정에서 발생하는 오류는?",
        "매출 증대를 위한 마케팅 전략은?",
        "고객 만족도 향상 방안은 무엇인가요?",
        "비용 절감을 위한 운영 효율화 방법은?",
        "시장 분석 결과를 바탕으로 한 전략은?",
        "경쟁사 대비 차별화 방안은?",
        "연구 방법론의 타당성은 어떻게 검증하나요?",
        "데이터 분석 결과의 통계적 유의성은?",
        "실험 설계의 한계점은 무엇인가요?",
        "연구 결과의 일반화 가능성은?",
        "이론적 배경과 실증적 결과의 일치성은?",
    ]
    
    analyzers = {
        "general": QuestionAnalyzer(domain="general"),
        "technical": QuestionAnalyzer(domain="technical"),
        "business": QuestionAnalyzer(domain="business"),
        "academic": QuestionAnalyzer(domain="academic"),
    }
    
    enhancers = {
        "general": KeywordEnhancer(domain="general"),
        "technical": KeywordEnhancer(domain="technical"),
        "business": KeywordEnhancer(domain="business"),
        "academic": KeywordEnhancer(domain="academic"),
    }
    
    results = {"keyword_counts": {}, "domain_keywords": {}, "enhancement_ratios": {}, "processing_times": {}}
    
    print("\n1. 도메인별 키워드 인식률 테스트")
    print("-" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n질문 {i}: {question}")
        for domain, analyzer in analyzers.items():
            start_time = time.time()
            analyzed = analyzer.analyze_question(question)
            enhancer = enhancers[domain]
            enhanced_keywords = enhancer.enhance_keywords(analyzed.keywords[:10])
            processing_time = time.time() - start_time
            domain_keywords = enhancer.extract_domain_specific_keywords(question)
            
            results.setdefault("keyword_counts", {}).setdefault(domain, []).append(len(analyzed.keywords))
            results.setdefault("domain_keywords", {}).setdefault(domain, []).append(len(domain_keywords))
            results.setdefault("enhancement_ratios", {}).setdefault(domain, []).append(len(enhanced_keywords) / max(len(analyzed.keywords), 1))
            results.setdefault("processing_times", {}).setdefault(domain, []).append(processing_time)
            
            print(f"  {domain}: 키워드 {len(analyzed.keywords)}개 → 향상 {len(enhanced_keywords)}개 (도메인 키워드: {len(domain_keywords)}개, 처리시간: {processing_time:.3f}초)")
    
    print("\n\n2. 통계 분석 결과")
    print("-" * 50)
    for domain in analyzers.keys():
        avg_keywords = sum(results["keyword_counts"][domain]) / len(results["keyword_counts"][domain])
        avg_domain_keywords = sum(results["domain_keywords"][domain]) / len(results["domain_keywords"][domain])
        avg_enhancement = sum(results["enhancement_ratios"][domain]) / len(results["enhancement_ratios"][domain])
        avg_processing_time = sum(results["processing_times"][domain]) / len(results["processing_times"][domain])
        print(f"\n{domain.upper()} 도메인:")
        print(f"  평균 키워드 수: {avg_keywords:.1f}개")
        print(f"  평균 도메인 키워드 수: {avg_domain_keywords:.1f}개")
        print(f"  평균 향상 비율: {avg_enhancement:.2f}배")
        print(f"  평균 처리 시간: {avg_processing_time:.3f}초")

def test_specific_improvements():
    """특정 개선사항 테스트"""
    print("\n" + "=" * 70)
    print("특정 개선사항 테스트")
    print("=" * 70)
    
    enhancer = KeywordEnhancer(domain="technical")
    
    print("\n1. 동의어 확장 테스트")
    print("-" * 30)
    for keyword in ["방법", "시스템", "성능", "보안", "데이터"]:
        enhanced = enhancer.enhance_keywords([keyword])
        synonyms = enhancer._get_synonyms(keyword)
        print(f"{keyword} → {enhanced[:5]} (동의어: {synonyms[:3]})")
    
    print("\n2. 약어 확장 테스트")
    print("-" * 30)
    for abbr in ["API", "DB", "UI", "OS", "CPU"]:
        expanded = enhancer._expand_abbreviation(abbr)
        print(f"{abbr} → {expanded}")
    
    print("\n3. 키워드 가중치 테스트")
    print("-" * 30)
    context = "시스템 API 성능 최적화를 위한 캐싱 전략을 구현했습니다."
    for keyword in ["시스템", "API", "성능", "최적화", "캐싱"]:
        weight = enhancer.calculate_keyword_weight(keyword, context)
        print(f"{keyword}: {weight:.2f}")

def main():
    try:
        test_keyword_recognition_improvement()
        test_specific_improvements()
        print("\n" + "=" * 70)
        print("키워드 인식률 향상 테스트 완료!")
        print("=" * 70)
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


