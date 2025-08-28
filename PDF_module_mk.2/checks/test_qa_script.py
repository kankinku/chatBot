#!/usr/bin/env python3
"""
테스트용 질문답변 스크립트 (실제 질문 세트 기반)
"""

import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import PDFQASystem

class PDFQATestSuite:
    """PDF QA 시스템 테스트 스위트"""
    
    def __init__(self):
        self.system = PDFQASystem()
        self.test_results = []
        self.test_questions = [
            {"question": "세종시의 2020년 기준 교통수단 분담률에서 승용차가 차지하는 비율은?", "expected_keywords": ["45.4%", "승용차"], "description": "세종시 승용차 교통수단 분담률 확인"},
            {"question": "세종시의 버스 교통수단 분담률은 얼마인가요?", "expected_keywords": ["7.3%", "버스"], "description": "세종시 버스 교통수단 분담률 확인"},
            {"question": "세종시 시민들의 대중교통 불만족도는?", "expected_keywords": ["61%", "불만족"], "description": "세종시 시민 대중교통 불만족도 확인"},
            {"question": "세종시의 교통 문제를 어떻게 설명하고 있나요?", "expected_keywords": ["교통지옥", "병목현상", "자가용 의존도"], "description": "세종시 교통 문제 설명 확인"},
            {"question": "BRT 전용차로가 어떤 문제를 일으키고 있나요?", "expected_keywords": ["교통 체증", "악화", "악순환"], "description": "BRT 전용차로 문제점 확인"},
            {"question": "팀명이 무엇인가요?", "expected_keywords": ["포버스", "포버스팀"], "description": "팀명 확인"},
            {"question": "팀장의 이름은?", "expected_keywords": ["원동영"], "description": "팀장 이름 확인"},
            {"question": "팀원은 몇 명인가요?", "expected_keywords": ["4명", "4"], "description": "팀원 수 확인"},
            {"question": "제안명은 무엇인가요?", "expected_keywords": ["AI 기반 세종시 교통 데이터 해석 플랫폼"], "description": "제안명 확인"},
            {"question": "팀장의 연락처는?", "expected_keywords": ["010-9984-8639"], "description": "팀장 연락처 확인"},
            {"question": "팀장의 이메일은?", "expected_keywords": ["wdyoung11@g.hongik.ac.kr"], "description": "팀장 이메일 확인"},
            {"question": "팀원들의 소속은?", "expected_keywords": ["홍익대학교"], "description": "팀원 소속 확인"},
            {"question": "키워드는 무엇인가요?", "expected_keywords": ["교통 데이터", "데이터 시각화", "AI 분석", "시민 참여", "정책 제안"], "description": "키워드 확인"},
            {"question": "개발 배경은 무엇인가요?", "expected_keywords": ["시민 주도형", "스마트 시티", "데이터 활용"], "description": "개발 배경 확인"},
        ]
    
    def check_answer_accuracy(self, answer: str, expected_keywords: list) -> tuple:
        answer_lower = answer.lower()
        found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        accuracy = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
        return accuracy, found_keywords
    
    def run_single_test(self, test_item: dict) -> dict:
        print(f"\n🔍 테스트: {test_item['description']}")
        print(f"질문: {test_item['question']}")
        start_time = time.time()
        result = self.system.ask_question(test_item['question'])
        response_time = time.time() - start_time
        accuracy, found_keywords = self.check_answer_accuracy(result['answer'], test_item['expected_keywords'])
        return {
            "question": test_item['question'],
            "description": test_item['description'],
            "answer": result['answer'],
            "confidence": result['confidence_score'],
            "expected_keywords": test_item['expected_keywords'],
            "found_keywords": found_keywords,
            "accuracy": accuracy,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
        }
    
    def run_full_test_suite(self, iterations: int = 1) -> list:
        print("="*70)
        print("🧪 PDF QA 시스템 테스트 시작")
        print("="*70)
        print(f"테스트 문서: data 폴더의 PDF 파일들")
        print(f"테스트 질문 수: {len(self.test_questions)}개")
        print(f"반복 횟수: {iterations}회")
        print("="*70)
        print("\n🔧 시스템 초기화 중...")
        if not self.system.initialize_components():
            print("❌ 시스템 초기화 실패")
            return []
        print("✅ 시스템 초기화 완료")
        all_results = []
        for iteration in range(iterations):
            if iterations > 1:
                print(f"\n🔄 {iteration + 1}회차 테스트")
                print("-" * 50)
            for i, test_item in enumerate(self.test_questions, 1):
                print(f"\n[{i}/{len(self.test_questions)}]", end=" ")
                res = self.run_single_test(test_item)
                res['iteration'] = iteration + 1
                all_results.append(res)
                time.sleep(0.5)
        self.test_results = all_results
        return all_results
    
    def analyze_results(self) -> dict:
        if not self.test_results:
            return {}
        total = len(self.test_results)
        avg_accuracy = sum(r['accuracy'] for r in self.test_results) / total
        avg_confidence = sum(r['confidence'] for r in self.test_results) / total
        avg_response_time = sum(r['response_time'] for r in self.test_results) / total
        excellent = sum(1 for r in self.test_results if r['accuracy'] >= 0.8)
        good = sum(1 for r in self.test_results if 0.5 <= r['accuracy'] < 0.8)
        poor = sum(1 for r in self.test_results if r['accuracy'] < 0.5)
        failed = [r for r in self.test_results if r['accuracy'] < 0.5]
        return {
            "total_tests": total,
            "avg_accuracy": avg_accuracy,
            "avg_confidence": avg_confidence,
            "avg_response_time": avg_response_time,
            "excellent_count": excellent,
            "good_count": good,
            "poor_count": poor,
            "failed_tests": failed,
        }
    
    def print_summary(self) -> None:
        analysis = self.analyze_results()
        if not analysis:
            print("❌ 분석할 테스트 결과가 없습니다.")
            return
        print("\n" + "="*70)
        print("📊 테스트 결과 요약")
        print("="*70)
        print(f"총 테스트 수: {analysis['total_tests']}개")
        print(f"평균 정확도: {analysis['avg_accuracy']:.2%}")
        print(f"평균 신뢰도: {analysis['avg_confidence']:.2f}")
        print(f"평균 응답시간: {analysis['avg_response_time']:.2f}초")
        print(f"\n📈 성능 분포:")
        print(f"  ✅ 우수 (80% 이상): {analysis['excellent_count']}개 ({analysis['excellent_count']/analysis['total_tests']:.1%})")
        print(f"  ⚠️ 보통 (50-80%): {analysis['good_count']}개 ({analysis['good_count']/analysis['total_tests']:.1%})")
        print(f"  ❌ 미흡 (50% 미만): {analysis['poor_count']}개 ({analysis['poor_count']/analysis['total_tests']:.1%})")
        if analysis['failed_tests']:
            print(f"\n❌ 실패한 테스트 ({len(analysis['failed_tests'])}개):")
            for test in analysis['failed_tests']:
                print(f"  - {test['description']}: {test['accuracy']:.1%}")
        print(f"\n🎯 전체 시스템 평가:")
        if analysis['avg_accuracy'] >= 0.8:
            print("✅ 우수 - 시스템이 안정적으로 작동합니다")
        elif analysis['avg_accuracy'] >= 0.6:
            print("⚠️ 양호 - 일부 개선이 필요합니다")
        else:
            print("❌ 미흡 - 시스템 개선이 필요합니다")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PDF QA 시스템 테스트")
    parser.add_argument("--iterations", "-i", type=int, default=1, help="테스트 반복 횟수")
    parser.add_argument("--save", "-s", action="store_true", help="결과를 파일로 저장")
    args = parser.parse_args()
    suite = PDFQATestSuite()
    try:
        suite.run_full_test_suite(iterations=args.iterations)
        suite.print_summary()
        if args.save:
            import json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(suite.test_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 테스트 결과 저장: {filename}")
    except KeyboardInterrupt:
        print("\n\n👋 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()


