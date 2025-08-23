#!/usr/bin/env python3
"""
테스트용 질문답변 스크립트
생성한 테스트 PDF에 대한 명확한 질문들을 반복 테스트합니다.
"""

import sys
import os
import time
from datetime import datetime

# 현재 디렉토리를 시스템 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from korean_qa import KoreanQASystem

class PDFQATestSuite:
    """PDF QA 시스템 테스트 스위트"""
    
    def __init__(self):
        """테스트 스위트 초기화"""
        self.system = KoreanQASystem()
        self.test_results = []
        
        # 테스트용 질문-정답 쌍 정의
        self.test_questions = [
            {
                "question": "회사명이 뭐야?",
                "expected_keywords": ["테크노 솔루션즈", "테크노솔루션즈"],
                "description": "회사명 확인"
            },
            {
                "question": "설립연도는 언제야?",
                "expected_keywords": ["2020", "2020년"],
                "description": "설립연도 확인"
            },
            {
                "question": "직원이 몇 명이야?",
                "expected_keywords": ["150", "150명"],
                "description": "직원 수 확인"
            },
            {
                "question": "본사가 어디에 있어?",
                "expected_keywords": ["서울", "강남구", "서울특별시"],
                "description": "본사 위치 확인"
            },
            {
                "question": "클라우드매니저 Pro 가격은?",
                "expected_keywords": ["50만원", "월 50만원"],
                "description": "제품 가격 확인"
            },
            {
                "question": "데이터분석 플랫폼은 언제 출시됐어?",
                "expected_keywords": ["2022년 1월", "2022", "1월"],
                "description": "제품 출시일 확인"
            },
            {
                "question": "개발팀장이 누구야?",
                "expected_keywords": ["김철수"],
                "description": "팀장 정보 확인"
            },
            {
                "question": "2023년 매출액이 얼마야?",
                "expected_keywords": ["180억", "180억원"],
                "description": "매출액 확인"
            },
            {
                "question": "전화번호가 뭐야?",
                "expected_keywords": ["02-1234-5678"],
                "description": "연락처 확인"
            },
            {
                "question": "ISO 인증을 언제 받았어?",
                "expected_keywords": ["2022", "2022년", "ISO 27001"],
                "description": "인증 정보 확인"
            },
            {
                "question": "특허를 몇 건 보유하고 있어?",
                "expected_keywords": ["12건", "12", "특허"],
                "description": "특허 보유 건수 확인"
            },
            {
                "question": "주요 고객사는 어디야?",
                "expected_keywords": ["삼성전자", "LG전자", "네이버"],
                "description": "고객사 정보 확인"
            }
        ]
    
    def check_answer_accuracy(self, answer: str, expected_keywords: list) -> tuple:
        """답변 정확성 검사"""
        answer_lower = answer.lower()
        found_keywords = []
        
        for keyword in expected_keywords:
            if keyword.lower() in answer_lower:
                found_keywords.append(keyword)
        
        accuracy = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
        return accuracy, found_keywords
    
    def run_single_test(self, test_item: dict) -> dict:
        """단일 테스트 실행"""
        print(f"\n🔍 테스트: {test_item['description']}")
        print(f"질문: {test_item['question']}")
        
        start_time = time.time()
        result = self.system.ask_question(test_item['question'])
        response_time = time.time() - start_time
        
        accuracy, found_keywords = self.check_answer_accuracy(
            result['answer'], 
            test_item['expected_keywords']
        )
        
        test_result = {
            "question": test_item['question'],
            "description": test_item['description'],
            "answer": result['answer'],
            "confidence": result['confidence'],
            "expected_keywords": test_item['expected_keywords'],
            "found_keywords": found_keywords,
            "accuracy": accuracy,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # 결과 출력
        print(f"답변: {result['answer']}")
        print(f"신뢰도: {result['confidence']:.2f}")
        print(f"정확도: {accuracy:.2%} ({len(found_keywords)}/{len(test_item['expected_keywords'])})")
        print(f"발견된 키워드: {found_keywords}")
        print(f"응답시간: {response_time:.2f}초")
        
        # 정확도에 따른 평가
        if accuracy >= 0.8:
            print("✅ 우수")
        elif accuracy >= 0.5:
            print("⚠️ 보통")
        else:
            print("❌ 미흡")
        
        return test_result
    
    def run_full_test_suite(self, iterations: int = 1) -> list:
        """전체 테스트 스위트 실행"""
        print("="*70)
        print("🧪 PDF QA 시스템 테스트 시작")
        print("="*70)
        print(f"테스트 문서: 테크노 솔루션즈 회사정보")
        print(f"테스트 질문 수: {len(self.test_questions)}개")
        print(f"반복 횟수: {iterations}회")
        print("="*70)
        
        all_results = []
        
        for iteration in range(iterations):
            if iterations > 1:
                print(f"\n🔄 {iteration + 1}회차 테스트")
                print("-" * 50)
            
            iteration_results = []
            
            for i, test_item in enumerate(self.test_questions, 1):
                print(f"\n[{i}/{len(self.test_questions)}]", end=" ")
                test_result = self.run_single_test(test_item)
                test_result['iteration'] = iteration + 1
                iteration_results.append(test_result)
                
                # 잠시 대기 (시스템 안정성)
                time.sleep(0.5)
            
            all_results.extend(iteration_results)
        
        self.test_results = all_results
        return all_results
    
    def analyze_results(self) -> dict:
        """테스트 결과 분석"""
        if not self.test_results:
            return {}
        
        # 전체 통계
        total_tests = len(self.test_results)
        avg_accuracy = sum(r['accuracy'] for r in self.test_results) / total_tests
        avg_confidence = sum(r['confidence'] for r in self.test_results) / total_tests
        avg_response_time = sum(r['response_time'] for r in self.test_results) / total_tests
        
        # 정확도별 분포
        excellent_count = sum(1 for r in self.test_results if r['accuracy'] >= 0.8)
        good_count = sum(1 for r in self.test_results if 0.5 <= r['accuracy'] < 0.8)
        poor_count = sum(1 for r in self.test_results if r['accuracy'] < 0.5)
        
        # 실패한 테스트
        failed_tests = [r for r in self.test_results if r['accuracy'] < 0.5]
        
        analysis = {
            "total_tests": total_tests,
            "avg_accuracy": avg_accuracy,
            "avg_confidence": avg_confidence,
            "avg_response_time": avg_response_time,
            "excellent_count": excellent_count,
            "good_count": good_count,
            "poor_count": poor_count,
            "failed_tests": failed_tests
        }
        
        return analysis
    
    def print_summary(self):
        """테스트 결과 요약 출력"""
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
        
        # 전체 평가
        print(f"\n🎯 전체 시스템 평가:")
        if analysis['avg_accuracy'] >= 0.8:
            print("✅ 우수 - 시스템이 안정적으로 작동합니다")
        elif analysis['avg_accuracy'] >= 0.6:
            print("⚠️ 양호 - 일부 개선이 필요합니다")
        else:
            print("❌ 미흡 - 시스템 개선이 필요합니다")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF QA 시스템 테스트")
    parser.add_argument("--iterations", "-i", type=int, default=1, 
                       help="테스트 반복 횟수 (기본값: 1)")
    parser.add_argument("--save", "-s", action="store_true",
                       help="결과를 파일로 저장")
    
    args = parser.parse_args()
    
    # 테스트 실행
    test_suite = PDFQATestSuite()
    
    try:
        test_suite.run_full_test_suite(iterations=args.iterations)
        test_suite.print_summary()
        
        # 결과 저장
        if args.save:
            import json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(test_suite.test_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 테스트 결과 저장: {filename}")
        
    except KeyboardInterrupt:
        print("\n\n👋 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
