#!/usr/bin/env python3
"""
격리된 테스트 환경
테스트용 PDF만을 사용하여 QA 시스템을 테스트합니다.
"""

import sys
import os
import time
import shutil
from datetime import datetime

# 현재 디렉토리를 시스템 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pdf_preprocessor import FastPDFPreprocessor, PDFDatabase
from core.fast_vector_store import FastVectorStore
from core.question_analyzer import QuestionAnalyzer
from core.answer_generator import AnswerGenerator, ModelType, GenerationConfig

class IsolatedTestSystem:
    """격리된 테스트 시스템"""
    
    def __init__(self, test_db_path="./data/test_database.db"):
        """초기화"""
        self.test_db_path = test_db_path
        self.database = PDFDatabase(db_path=test_db_path)
        self.vector_store = FastVectorStore()
        self.question_analyzer = None
        self.answer_generator = None
        self.is_ready = False
        
        print("🧪 격리된 테스트 시스템 초기화")
    
    def setup_test_environment(self):
        """테스트 환경 설정"""
        try:
            print("📚 테스트 환경 설정 중...")
            
            # 기존 테스트 DB 삭제
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
                print(f"기존 테스트 DB 삭제: {self.test_db_path}")
            
            # 새 데이터베이스 초기화
            self.database = PDFDatabase(db_path=self.test_db_path)
            
            # 테스트 PDF만 처리
            test_pdf_path = "./data/pdfs/misc/테스트용_회사정보.pdf"
            
            if not os.path.exists(test_pdf_path):
                print(f"❌ 테스트 PDF를 찾을 수 없습니다: {test_pdf_path}")
                return False
            
            preprocessor = FastPDFPreprocessor(db_path=self.test_db_path)
            
            print(f"📄 테스트 PDF 처리 중: {test_pdf_path}")
            result = preprocessor.process_pdf(test_pdf_path)
            
            if result['success']:
                print(f"✅ PDF 처리 완료: {result['total_chunks']}개 청크")
                return True
            else:
                print(f"❌ PDF 처리 실패: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ 테스트 환경 설정 실패: {e}")
            return False
    
    def load_components(self) -> bool:
        """시스템 컴포넌트 로드"""
        try:
            print("🔧 컴포넌트 로드 중...")
            
            # 벡터 저장소 로드
            if not self.vector_store.load_from_database(self.database):
                print("❌ 벡터 저장소 로드 실패")
                return False
            
            stats = self.vector_store.get_statistics()
            print(f"✅ 벡터 저장소 로드 완료: {stats['total_chunks']}개 청크")
            
            # 질문 분석기 초기화
            self.question_analyzer = QuestionAnalyzer()
            
            # 답변 생성기 초기화
            config = GenerationConfig(
                temperature=0.3,
                top_p=0.8,
                top_k=30,
                max_length=256
            )
            
            self.answer_generator = AnswerGenerator(
                model_type=ModelType.OLLAMA,
                model_name="mistral:latest",
                generation_config=config
            )
            
            if not self.answer_generator.load_model():
                print("❌ 답변 생성기 로드 실패")
                return False
            
            self.is_ready = True
            print("✅ 모든 컴포넌트 로드 완료")
            return True
            
        except Exception as e:
            print(f"❌ 컴포넌트 로드 실패: {e}")
            return False
    
    def ask_question(self, question: str) -> dict:
        """질문 처리"""
        if not self.is_ready:
            return {
                "answer": "시스템이 준비되지 않았습니다.",
                "confidence": 0.0
            }
        
        try:
            # 질문 분석
            analyzed = self.question_analyzer.analyze_question(question)
            
            # 관련 문서 검색
            search_results = self.vector_store.search(
                analyzed.embedding, 
                top_k=5,
                score_threshold=0.0
            )
            
            if not search_results:
                return {
                    "answer": "관련 정보를 찾을 수 없습니다.",
                    "confidence": 0.0,
                    "question_type": analyzed.question_type.value
                }
            
            # 답변 생성
            answer_result = self.answer_generator.generate_answer(
                analyzed, search_results, None
            )
            
            return {
                "answer": answer_result.content,
                "confidence": answer_result.confidence_score,
                "question_type": analyzed.question_type.value,
                "used_chunks": len(search_results),
                "search_results": [
                    {
                        "content": chunk.content[:200] + "...",
                        "page": chunk.page_number,
                        "score": score
                    }
                    for chunk, score in search_results[:3]
                ]
            }
            
        except Exception as e:
            print(f"❌ 질문 처리 실패: {e}")
            return {
                "answer": f"질문 처리 중 오류가 발생했습니다: {e}",
                "confidence": 0.0
            }

def run_isolated_test():
    """격리된 테스트 실행"""
    
    # 테스트 질문들
    test_questions = [
        {
            "question": "회사명이 뭐야?",
            "expected_keywords": ["테크노 솔루션즈", "테크노솔루션즈"],
            "description": "회사명 확인"
        },
        {
            "question": "설립연도는?",
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
            "expected_keywords": ["서울", "강남구"],
            "description": "본사 위치 확인"
        },
        {
            "question": "클라우드매니저 Pro 가격은?",
            "expected_keywords": ["50만원", "월 50만원"],
            "description": "제품 가격 확인"
        }
    ]
    
    print("="*70)
    print("🧪 격리된 PDF QA 시스템 테스트")
    print("="*70)
    
    # 시스템 초기화
    system = IsolatedTestSystem()
    
    # 테스트 환경 설정
    if not system.setup_test_environment():
        print("❌ 테스트 환경 설정 실패")
        return
    
    # 컴포넌트 로드
    if not system.load_components():
        print("❌ 컴포넌트 로드 실패")
        return
    
    print(f"\n📝 테스트 시작 ({len(test_questions)}개 질문)")
    print("-" * 70)
    
    results = []
    
    for i, test_item in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] {test_item['description']}")
        print(f"질문: {test_item['question']}")
        
        start_time = time.time()
        result = system.ask_question(test_item['question'])
        response_time = time.time() - start_time
        
        # 정확도 계산
        answer_lower = result['answer'].lower()
        found_keywords = []
        for keyword in test_item['expected_keywords']:
            if keyword.lower() in answer_lower:
                found_keywords.append(keyword)
        
        accuracy = len(found_keywords) / len(test_item['expected_keywords']) if test_item['expected_keywords'] else 0
        
        print(f"답변: {result['answer']}")
        print(f"신뢰도: {result['confidence']:.2f}")
        print(f"정확도: {accuracy:.2%} ({len(found_keywords)}/{len(test_item['expected_keywords'])})")
        print(f"발견된 키워드: {found_keywords}")
        print(f"응답시간: {response_time:.2f}초")
        
        if 'search_results' in result:
            print("검색된 청크:")
            for j, chunk_info in enumerate(result['search_results'], 1):
                print(f"  {j}. 점수: {chunk_info['score']:.3f}, 페이지: {chunk_info['page']}")
                print(f"     내용: {chunk_info['content']}")
        
        # 평가
        if accuracy >= 0.8:
            print("✅ 우수")
        elif accuracy >= 0.5:
            print("⚠️ 보통")
        else:
            print("❌ 미흡")
        
        results.append({
            "question": test_item['question'],
            "answer": result['answer'],
            "accuracy": accuracy,
            "confidence": result['confidence'],
            "response_time": response_time,
            "found_keywords": found_keywords
        })
        
        time.sleep(0.5)  # 잠시 대기
    
    # 결과 요약
    print("\n" + "="*70)
    print("📊 테스트 결과 요약")
    print("="*70)
    
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_response_time = sum(r['response_time'] for r in results) / len(results)
    
    excellent_count = sum(1 for r in results if r['accuracy'] >= 0.8)
    good_count = sum(1 for r in results if 0.5 <= r['accuracy'] < 0.8)
    poor_count = sum(1 for r in results if r['accuracy'] < 0.5)
    
    print(f"총 테스트 수: {len(results)}개")
    print(f"평균 정확도: {avg_accuracy:.2%}")
    print(f"평균 신뢰도: {avg_confidence:.2f}")
    print(f"평균 응답시간: {avg_response_time:.2f}초")
    
    print(f"\n📈 성능 분포:")
    print(f"  ✅ 우수 (80% 이상): {excellent_count}개 ({excellent_count/len(results):.1%})")
    print(f"  ⚠️ 보통 (50-80%): {good_count}개 ({good_count/len(results):.1%})")
    print(f"  ❌ 미흡 (50% 미만): {poor_count}개 ({poor_count/len(results):.1%})")
    
    # 전체 평가
    print(f"\n🎯 전체 시스템 평가:")
    if avg_accuracy >= 0.8:
        print("✅ 우수 - 테스트 PDF에 대해 정확하게 답변합니다")
    elif avg_accuracy >= 0.6:
        print("⚠️ 양호 - 일부 개선이 필요합니다")
    else:
        print("❌ 미흡 - 시스템 개선이 필요합니다")

if __name__ == "__main__":
    run_isolated_test()
