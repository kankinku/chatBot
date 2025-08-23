#!/usr/bin/env python3
"""
한국어 PDF 질문답변 시스템 - 고속 버전

특징:
- 답변 길이 제한으로 2-3배 빠른 속도
- 핵심 정보만 간결하게 제공
- 실시간 질문 응답에 최적화
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 현재 디렉토리를 시스템 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pdf_preprocessor import FastPDFPreprocessor, PDFDatabase
from core.fast_vector_store import FastVectorStore
from core.question_analyzer import QuestionAnalyzer
from core.answer_generator import AnswerGenerator, ModelType, GenerationConfig

# 로깅 레벨을 WARNING으로 설정하여 출력 최소화
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class FastKoreanQASystem:
    """고속 한국어 PDF 질문답변 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        self.database = PDFDatabase()
        self.vector_store = FastVectorStore()
        self.question_analyzer = None
        self.answer_generator = None
        self.is_ready = False
    
    def load_components(self) -> bool:
        """시스템 컴포넌트 고속 로드"""
        try:
            # 1. 벡터 저장소 로드
            if not self.vector_store.load_from_database(self.database):
                return False
            
            # 2. 질문 분석기 초기화
            self.question_analyzer = QuestionAnalyzer()
            
            # 3. 고속 답변 생성기 초기화
            config = GenerationConfig(
                temperature=0.2,  # 더 결정적
                top_p=0.7,       # 더 집중적
                top_k=20,        # 더 제한적
                max_length=128   # 짧은 답변
            )
            
            # 가장 빠른 모델 사용
            self.answer_generator = AnswerGenerator(
                model_type=ModelType.OLLAMA,
                model_name="mistral:latest",
                generation_config=config
            )
            
            if not self.answer_generator.load_model():
                return False
            
            self.is_ready = True
            return True
            
        except Exception as e:
            print(f"시스템 로드 실패: {e}")
            return False
    
    def quick_ask(self, question: str) -> str:
        """빠른 질문 처리"""
        if not self.is_ready:
            if not self.load_components():
                return "시스템 로드 실패"
        
        try:
            # 1. 질문 분석 (간소화)
            analyzed = self.question_analyzer.analyze_question(question)
            
            # 2. 빠른 검색 (상위 3개만)
            search_results = self.vector_store.search(
                analyzed.embedding, 
                top_k=3,
                score_threshold=0.0
            )
            
            if not search_results:
                return "관련 정보를 찾을 수 없습니다."
            
            # 3. 빠른 답변 생성
            answer_result = self.answer_generator.generate_answer(
                analyzed, search_results, None
            )
            
            return answer_result.content
            
        except Exception as e:
            return f"오류: {e}"
    
    def interactive_fast_mode(self):
        """고속 대화형 모드"""
        print("🚀 고속 한국어 PDF QA 시스템")
        print("💡 간결하고 빠른 답변에 최적화")
        print("=" * 40)
        
        # 시스템 로딩
        print("⚡ 시스템 로딩 중...")
        if not self.load_components():
            print("❌ 시스템 로드 실패")
            return
        print("✅ 로딩 완료!")
        
        while True:
            try:
                question = input("\n질문: ").strip()
                
                if not question or question == "/종료":
                    break
                
                print("🤔 생각 중...")
                answer = self.quick_ask(question)
                print(f"💡 {answer}")
                
            except (KeyboardInterrupt, EOFError):
                break
        
        print("\n👋 종료")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="고속 한국어 PDF QA")
    parser.add_argument("question", nargs="?", help="질문")
    parser.add_argument("--interactive", "-i", action="store_true", help="대화형 모드")
    
    args = parser.parse_args()
    
    system = FastKoreanQASystem()
    
    if args.interactive or not args.question:
        system.interactive_fast_mode()
    else:
        answer = system.quick_ask(args.question)
        print(answer)

if __name__ == "__main__":
    main()
