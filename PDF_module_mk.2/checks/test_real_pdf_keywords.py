#!/usr/bin/env python3
"""
실제 PDF 문서를 사용한 키워드 인식률 향상 테스트
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.question_analyzer import QuestionAnalyzer
from utils.keyword_enhancer import KeywordEnhancer
from core.pdf_processor import PDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_with_real_pdf():
    print("=" * 60)
    print("실제 PDF 문서를 사용한 키워드 인식률 테스트")
    print("=" * 60)
    pdf_path = "data/리서치.pdf"
    if not os.path.exists(pdf_path):
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return
    try:
        print("\n1. PDF 문서 처리 중...")
        pdf_processor = PDFProcessor()
        chunks = pdf_processor.process_pdf(pdf_path)
        print(f"   처리된 청크 수: {len(chunks)}")
        print("\n2. 키워드 향상기 초기화...")
        enhancer = KeywordEnhancer(domain="general")
        print("\n3. 질문 분석기 초기화...")
        analyzer = QuestionAnalyzer(domain="general")
        print("\n4. 실제 질문으로 키워드 인식률 테스트")
        print("-" * 50)
        test_questions = [
            "이 문서의 주요 내용은 무엇인가요?",
            "연구 방법론은 어떻게 구성되어 있나요?",
            "결론에서 제시된 핵심 사항은?",
            "데이터 분석 결과는 어떻게 나타났나요?",
            "이 연구의 한계점은 무엇인가요?",
        ]
        for i, question in enumerate(test_questions, 1):
            print(f"\n질문 {i}: {question}")
            analyzed = analyzer.analyze_question(question)
            print(f"  질문 유형: {analyzed.question_type.value}")
            print(f"  추출된 키워드: {analyzed.keywords[:8]}")
            print(f"  키워드 개수: {len(analyzed.keywords)}")
            enhanced_keywords = enhancer.enhance_keywords(analyzed.keywords[:5])
            print(f"  향상된 키워드: {enhanced_keywords[:8]}")
            domain_keywords = enhancer.extract_domain_specific_keywords(question)
            print(f"  도메인 키워드: {domain_keywords}")
        print("\n5. 문서 내용에서 키워드 추출 테스트")
        print("-" * 50)
        if chunks and len(chunks) > 0:
            first_chunk = chunks[0] if isinstance(chunks, list) else chunks
            sample_content = first_chunk.content[:500] if hasattr(first_chunk, 'content') else str(first_chunk)[:500]
            print(f"샘플 내용: {sample_content[:100]}...")
            recommended = enhancer.recommend_keywords(sample_content, max_keywords=10)
            print(f"\n추천된 키워드:")
            for keyword, weight in recommended:
                print(f"  {keyword}: {weight:.2f}")
        print("\n6. 성능 비교")
        print("-" * 50)
        import time
        start_time = time.time()
        basic_keywords = []
        for chunk in chunks[:5]:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            words = content.lower().split()
            basic_keywords.extend([w for w in words if len(w) > 2])
        basic_time = time.time() - start_time
        start_time = time.time()
        enhanced_keywords = []
        for chunk in chunks[:5]:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            recommended = enhancer.recommend_keywords(content, max_keywords=5)
            enhanced_keywords.extend([kw for kw, _ in recommended])
        enhanced_time = time.time() - start_time
        print(f"기본 키워드 추출 시간: {basic_time:.3f}초")
        print(f"향상된 키워드 추출 시간: {enhanced_time:.3f}초")
        print(f"성능 향상: {basic_time/enhanced_time:.1f}배")
        print(f"기본 키워드 수: {len(set(basic_keywords))}")
        print(f"향상된 키워드 수: {len(set(enhanced_keywords))}")
        print("\n7. 키워드 품질 분석")
        print("-" * 50)
        meaningful_basic = len([w for w in set(basic_keywords) if len(w) > 3])
        meaningful_enhanced = len([w for w in set(enhanced_keywords) if len(w) > 3])
        basic_quality = meaningful_basic/len(set(basic_keywords))*100 if len(set(basic_keywords))>0 else 0
        enhanced_quality = meaningful_enhanced/len(set(enhanced_keywords))*100 if len(set(enhanced_keywords))>0 else 0
        print(f"의미있는 기본 키워드: {meaningful_basic}/{len(set(basic_keywords))} ({basic_quality:.1f}%)")
        print(f"의미있는 향상된 키워드: {meaningful_enhanced}/{len(set(enhanced_keywords))} ({enhanced_quality:.1f}%)")
        if basic_quality > 0:
            quality_improvement = ((enhanced_quality - basic_quality) / basic_quality) * 100
            print(f"키워드 품질 개선율: {quality_improvement:.1f}%")
        print("\n" + "=" * 60)
        print("실제 PDF 키워드 인식률 테스트 완료!")
        print("=" * 60)
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def test_keyword_matching_accuracy():
    print("\n" + "=" * 60)
    print("키워드 매칭 정확도 테스트")
    print("=" * 60)
    enhancer = KeywordEnhancer(domain="general")
    test_cases = [
        {"question": "시스템 성능을 어떻게 개선할 수 있나요?", "expected_keywords": ["시스템", "성능", "개선", "방법"], "context": "시스템 성능 최적화를 위한 다양한 방법들이 있습니다."},
        {"question": "데이터 분석 결과는 무엇인가요?", "expected_keywords": ["데이터", "분석", "결과"], "context": "데이터 분석 결과를 통해 중요한 인사이트를 얻을 수 있습니다."},
        {"question": "보안 정책은 어떻게 설정하나요?", "expected_keywords": ["보안", "정책", "설정"], "context": "보안 정책 설정은 시스템 안전성에 매우 중요합니다."},
    ]
    total_accuracy = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}: {test_case['question']}")
        extracted_keywords = enhancer.recommend_keywords(test_case['question'], max_keywords=10)
        extracted = [kw for kw, _ in extracted_keywords]
        expected = set(test_case['expected_keywords'])
        extracted_set = set(extracted)
        accuracy = len(expected & extracted_set) / len(expected) if expected else 0
        total_accuracy += accuracy
        print(f"  예상 키워드: {list(expected)}")
        print(f"  추출된 키워드: {extracted[:5]}")
        print(f"  매칭된 키워드: {list(expected & extracted_set)}")
        print(f"  정확도: {accuracy:.2f} ({accuracy*100:.1f}%)")
    avg_accuracy = total_accuracy / len(test_cases)
    print(f"\n평균 정확도: {avg_accuracy:.2f} ({avg_accuracy*100:.1f}%)")

def main():
    try:
        test_with_real_pdf()
        test_keyword_matching_accuracy()
        print("\n" + "=" * 60)
        print("모든 테스트 완료!")
        print("=" * 60)
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


