#!/usr/bin/env python3
"""
한국어 PDF 질문답변 시스템 - 간단 실행 스크립트

사용법:
    python korean_qa.py                    # 대화형 모드
    python korean_qa.py --setup            # 초기 설정
    python korean_qa.py --process           # PDF 전처리
    python korean_qa.py "질문내용"         # 직접 질문
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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KoreanQASystem:
    """한국어 PDF 질문답변 시스템 (최적화 버전)"""
    
    def __init__(self):
        """시스템 초기화"""
        self.database = PDFDatabase()
        self.vector_store = FastVectorStore()
        self.question_analyzer = None
        self.answer_generator = None
        self.is_ready = False
        
        logger.info("🚀 한국어 PDF QA 시스템 초기화")
    
    def setup_system(self) -> bool:
        """시스템 초기 설정"""
        try:
            logger.info("📚 시스템 초기 설정 시작...")
            
            # 1. PDF 전처리
            self.preprocess_pdfs()
            
            # 2. 시스템 컴포넌트 로드
            self.load_components()
            
            logger.info("✅ 시스템 설정 완료!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 시스템 설정 실패: {e}")
            return False
    
    def preprocess_pdfs(self):
        """PDF 파일들 전처리"""
        logger.info("📄 PDF 파일 전처리 시작...")
        
        # data/pdfs 디렉토리의 모든 PDF 처리
        pdf_dir = "./data/pdfs"
        if os.path.exists(pdf_dir):
            preprocessor = FastPDFPreprocessor()
            stats = preprocessor.preprocess_directory(pdf_dir)
            
            logger.info(f"📊 전처리 완료: 성공 {stats['success']}, "
                       f"실패 {stats['failed']}, 건너뜀 {stats['skipped']}")
        else:
            logger.warning(f"PDF 디렉토리가 없습니다: {pdf_dir}")
    
    def load_components(self) -> bool:
        """시스템 컴포넌트 로드"""
        try:
            logger.info("🔧 시스템 컴포넌트 로드 중...")
            
            # 1. 벡터 저장소 로드
            if not self.vector_store.load_from_database(self.database):
                logger.error("벡터 저장소 로드 실패")
                return False
            
            # 2. 질문 분석기 초기화
            self.question_analyzer = QuestionAnalyzer()
            
            # 3. 답변 생성기 초기화 (한국어 최적화 + 속도 향상)
            config = GenerationConfig(
                temperature=0.3,
                top_p=0.8,
                top_k=30,
                max_length=256  # 길이 단축으로 속도 향상
            )
            
            # Mistral 모델 우선 사용
            try:
                self.answer_generator = AnswerGenerator(
                    model_type=ModelType.OLLAMA,
                    model_name="mistral:latest",
                    generation_config=config
                )
                if not self.answer_generator.load_model():
                    raise Exception("Mistral 모델 로드 실패")
                    
                logger.info("✅ Mistral 모델 로드 완료")
                
            except Exception as e:
                logger.warning(f"Mistral 모델 로드 실패, llama2 사용: {e}")
                self.answer_generator = AnswerGenerator(
                    model_type=ModelType.OLLAMA,
                    model_name="llama2:7b",
                    generation_config=config
                )
                if not self.answer_generator.load_model():
                    raise Exception("LLM 모델 로드 실패")
            
            self.is_ready = True
            logger.info("✅ 모든 컴포넌트 로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 컴포넌트 로드 실패: {e}")
            return False
    
    def ask_question(self, question: str) -> dict:
        """질문 처리"""
        if not self.is_ready:
            if not self.load_components():
                return {
                    "answer": "시스템이 준비되지 않았습니다. --setup 옵션으로 시스템을 설정해주세요.",
                    "confidence": 0.0
                }
        
        try:
            # 1. 질문 분석
            analyzed = self.question_analyzer.analyze_question(question)
            
            # 2. 관련 문서 검색
            search_results = self.vector_store.search(
                analyzed.embedding, 
                top_k=5,
                score_threshold=0.0  # 임계값을 0으로 설정
            )
            
            if not search_results:
                return {
                    "answer": "관련 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                    "confidence": 0.0,
                    "question_type": analyzed.question_type.value
                }
            
            # 3. 답변 생성
            answer_result = self.answer_generator.generate_answer(
                analyzed, search_results, None
            )
            
            return {
                "answer": answer_result.content,
                "confidence": answer_result.confidence_score,
                "question_type": analyzed.question_type.value,
                "used_chunks": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            return {
                "answer": f"질문 처리 중 오류가 발생했습니다: {e}",
                "confidence": 0.0
            }
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n" + "="*60)
        print("🇰🇷 한국어 PDF 질문답변 시스템")
        print("="*60)
        print("명령어:")
        print("  /상태     - 시스템 상태 확인")
        print("  /통계     - 데이터 통계")
        print("  /전처리   - PDF 다시 전처리")
        print("  /종료     - 프로그램 종료")
        print("="*60)
        
        # 시스템 상태 확인
        if not self.is_ready:
            print("🔧 시스템 로딩 중...")
            if not self.load_components():
                print("❌ 시스템 로드 실패. '/전처리' 명령어로 설정하세요.")
        
        print("\n💬 질문을 입력하세요:")
        
        while True:
            try:
                try:
                    question = input("\n질문: ").strip()
                except EOFError:
                    print("\n👋 프로그램을 종료합니다.")
                    break
                
                if not question:
                    continue
                
                # 명령어 처리
                if question == "/종료":
                    print("👋 프로그램을 종료합니다.")
                    break
                elif question == "/상태":
                    self.show_status()
                    continue
                elif question == "/통계":
                    self.show_statistics()
                    continue
                elif question == "/전처리":
                    print("📄 PDF 전처리 시작...")
                    self.preprocess_pdfs()
                    self.load_components()
                    print("✅ 전처리 완료!")
                    continue
                
                # 질문 처리
                print("🤔 답변 생성 중...")
                result = self.ask_question(question)
                
                print(f"\n💡 {result['answer']}")
                print(f"📊 신뢰도: {result['confidence']:.2f}")
                print(f"📋 질문유형: {result['question_type']}")
                if 'used_chunks' in result:
                    print(f"📚 참조문서: {result['used_chunks']}개")
                
            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")
    
    def show_status(self):
        """시스템 상태 표시"""
        print("\n📊 시스템 상태:")
        print(f"  벡터 저장소: {'✅ 로드됨' if self.vector_store.is_loaded else '❌ 미로드'}")
        print(f"  질문 분석기: {'✅ 준비됨' if self.question_analyzer else '❌ 미준비'}")
        print(f"  답변 생성기: {'✅ 준비됨' if self.answer_generator else '❌ 미준비'}")
        
        if self.vector_store.is_loaded:
            stats = self.vector_store.get_statistics()
            print(f"  총 청크 수: {stats['total_chunks']:,}개")
    
    def show_statistics(self):
        """데이터 통계 표시"""
        print("\n📈 데이터 통계:")
        
        stats = self.database.get_statistics()
        print(f"  처리된 PDF: {stats['total_files']}개")
        print(f"  총 페이지: {stats['total_pages']:,}페이지")
        print(f"  총 청크: {stats['total_chunks']:,}개")
        
        if stats['files']:
            print("\n📚 처리된 파일들:")
            for file_info in stats['files'][:5]:  # 최근 5개만
                print(f"  - {file_info['filename']} "
                      f"({file_info['total_chunks']}청크)")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="한국어 PDF 질문답변 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python korean_qa.py                    # 대화형 모드
  python korean_qa.py --setup            # 초기 설정
  python korean_qa.py --process           # PDF 전처리
  python korean_qa.py "여과공정이 뭐야?"   # 직접 질문
        """
    )
    
    parser.add_argument("question", nargs="?", help="질문 내용")
    parser.add_argument("--setup", action="store_true", help="시스템 초기 설정")
    parser.add_argument("--process", action="store_true", help="PDF 전처리")
    parser.add_argument("--stats", action="store_true", help="통계 정보")
    
    args = parser.parse_args()
    
    system = KoreanQASystem()
    
    # 명령어 처리
    if args.setup:
        system.setup_system()
    elif args.process:
        system.preprocess_pdfs()
    elif args.stats:
        system.show_statistics()
    elif args.question:
        # 직접 질문
        result = system.ask_question(args.question)
        print(f"\n질문: {args.question}")
        print(f"답변: {result['answer']}")
        print(f"신뢰도: {result['confidence']:.2f}")
    else:
        # 대화형 모드
        system.interactive_mode()

if __name__ == "__main__":
    main()
