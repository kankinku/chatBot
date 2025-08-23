#!/usr/bin/env python3
"""
PDF QA 시스템 메인 실행 파일

이 파일은 PDF QA 시스템의 전체 파이프라인을 실행하고 관리하는
메인 엔트리포인트입니다.
"""

import argparse
import sys
import os
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

# 핵심 모듈들 임포트
from core.pdf_processor import PDFProcessor
from core.vector_store import HybridVectorStore
from core.question_analyzer import QuestionAnalyzer
from core.answer_generator import AnswerGenerator, ModelType, GenerationConfig
from core.evaluator import PDFQAEvaluator
from api.endpoints import run_server

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_qa_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class PDFQASystem:
    """
    PDF QA 시스템 메인 클래스
    
    전체 시스템의 초기화, 설정, 실행을 담당합니다.
    """
    
    def __init__(self, 
                 model_type: str = "ollama",
                 model_name: str = "llama2:7b",
                 embedding_model: str = "jhgan/ko-sroberta-multitask"):
        """
        시스템 초기화
        
        Args:
            model_type: 사용할 LLM 타입 (ollama/huggingface/llama_cpp)
            model_name: 모델 이름
            embedding_model: 임베딩 모델 이름
        """
        self.model_type = ModelType(model_type)
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        # 컴포넌트들
        self.pdf_processor: Optional[PDFProcessor] = None
        self.vector_store: Optional[HybridVectorStore] = None
        self.question_analyzer: Optional[QuestionAnalyzer] = None
        self.answer_generator: Optional[AnswerGenerator] = None
        self.evaluator: Optional[PDFQAEvaluator] = None
        
        logger.info(f"PDF QA 시스템 초기화: {model_type}/{model_name}")
    
    def initialize_components(self) -> bool:
        """
        시스템 컴포넌트들 초기화
        
        Returns:
            초기화 성공 여부
        """
        try:
            logger.info("컴포넌트 초기화 시작...")
            
            # 1. PDF 프로세서 초기화
            self.pdf_processor = PDFProcessor(
                embedding_model=self.embedding_model
            )
            logger.info("✓ PDF 프로세서 초기화 완료")
            
            # 2. 벡터 저장소 초기화
            self.vector_store = HybridVectorStore(
                embedding_dimension=768,  # 기본 임베딩 차원
                persist_directory="./data/vector_store"
            )
            logger.info("✓ 벡터 저장소 초기화 완료")
            
            # 3. 질문 분석기 초기화
            self.question_analyzer = QuestionAnalyzer(
                embedding_model=self.embedding_model
            )
            logger.info("✓ 질문 분석기 초기화 완료")
            
            # 4. 답변 생성기 초기화
            config = GenerationConfig(
                max_length=512,
                temperature=0.7,
                top_p=0.9
            )
            
            self.answer_generator = AnswerGenerator(
                model_type=self.model_type,
                model_name=self.model_name,
                generation_config=config
            )
            
            # 모델 로드
            if not self.answer_generator.load_model():
                logger.error("답변 생성 모델 로드 실패")
                return False
            
            logger.info("✓ 답변 생성기 초기화 완료")
            
            # 5. 평가기 초기화
            self.evaluator = PDFQAEvaluator(
                embedding_model=self.embedding_model
            )
            logger.info("✓ 평가기 초기화 완료")
            
            logger.info("모든 컴포넌트 초기화 완료!")
            return True
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            return False
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        PDF 파일 처리
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            처리 결과
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        logger.info(f"PDF 처리 시작: {pdf_path}")
        start_time = time.time()
        
        try:
            # 1. PDF 텍스트 추출 및 임베딩 생성
            chunks, metadata = self.pdf_processor.process_pdf(pdf_path)
            
            # 2. 벡터 저장소에 추가
            self.vector_store.add_chunks(chunks)
            
            # 3. 저장소 저장
            self.vector_store.save()
            
            processing_time = time.time() - start_time
            
            result = {
                "pdf_id": metadata["pdf_id"],
                "filename": os.path.basename(pdf_path),
                "total_chunks": len(chunks),
                "total_pages": metadata.get("pages", 0),
                "processing_time": processing_time,
                "extraction_methods": metadata.get("extraction_method", [])
            }
            
            logger.info(f"PDF 처리 완료: {len(chunks)}개 청크, {processing_time:.2f}초")
            return result
            
        except Exception as e:
            logger.error(f"PDF 처리 실패: {e}")
            raise
    
    def ask_question(self, 
                    question: str, 
                    use_context: bool = True,
                    max_chunks: int = 5) -> Dict:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
            use_context: 이전 대화 컨텍스트 사용 여부
            max_chunks: 검색할 최대 청크 수
            
        Returns:
            답변 결과
        """
        logger.info(f"질문 처리: {question}")
        start_time = time.time()
        
        try:
            # 1. 질문 분석
            analyzed_question = self.question_analyzer.analyze_question(
                question, use_conversation_context=use_context
            )
            
            # 2. 관련 문서 검색
            relevant_chunks = self.vector_store.search(
                analyzed_question.embedding,
                top_k=max_chunks
            )
            
            if not relevant_chunks:
                return {
                    "answer": "관련된 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요.",
                    "confidence_score": 0.0,
                    "question_type": analyzed_question.question_type.value,
                    "used_chunks": [],
                    "processing_time": time.time() - start_time
                }
            
            # 3. 대화 기록 가져오기
            conversation_history = None
            if use_context:
                conversation_history = self.question_analyzer.get_conversation_context(3)
            
            # 4. 답변 생성
            answer = self.answer_generator.generate_answer(
                analyzed_question,
                relevant_chunks,
                conversation_history
            )
            
            # 5. 대화 기록에 추가
            self.question_analyzer.add_conversation_item(
                question,
                answer.content,
                answer.used_chunks,
                answer.confidence_score
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "answer": answer.content,
                "confidence_score": answer.confidence_score,
                "question_type": analyzed_question.question_type.value,
                "intent": analyzed_question.intent,
                "keywords": analyzed_question.keywords,
                "used_chunks": answer.used_chunks,
                "processing_time": processing_time,
                "model_name": answer.model_name
            }
            
            logger.info(f"답변 생성 완료: {processing_time:.2f}초, 신뢰도: {answer.confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            raise
    
    def interactive_mode(self):
        """대화형 모드 실행"""
        print("\n" + "="*60)
        print("PDF QA 시스템 - 대화형 모드")
        print("="*60)
        print("명령어:")
        print("  - 질문 입력: 자유롭게 질문하세요")
        print("  - '/clear': 대화 기록 초기화")
        print("  - '/status': 시스템 상태 조회")
        print("  - '/exit': 프로그램 종료")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n질문: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == '/exit':
                    print("프로그램을 종료합니다.")
                    break
                elif user_input == '/clear':
                    self.question_analyzer.conversation_history.clear()
                    print("대화 기록이 초기화되었습니다.")
                    continue
                elif user_input == '/status':
                    self.show_system_status()
                    continue
                
                # 질문 처리
                result = self.ask_question(user_input)
                
                print(f"\n답변: {result['answer']}")
                print(f"신뢰도: {result['confidence_score']:.2f}")
                print(f"질문 유형: {result['question_type']}")
                print(f"처리 시간: {result['processing_time']:.2f}초")
                
            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류 발생: {e}")
                logger.error(f"대화형 모드 오류: {e}")
    
    def show_system_status(self):
        """시스템 상태 표시"""
        print("\n시스템 상태:")
        print(f"- 답변 생성 모델: {self.answer_generator.llm.model_name}")
        print(f"- 모델 로드 상태: {'정상' if self.answer_generator.llm.is_loaded else '오류'}")
        print(f"- 대화 기록: {len(self.question_analyzer.conversation_history)}개")
        
        # 메모리 사용량
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"- 메모리 사용량: {memory_mb:.1f}MB")
        except:
            pass
    
    def cleanup(self):
        """시스템 정리"""
        logger.info("시스템 정리 중...")
        
        if self.answer_generator:
            self.answer_generator.unload_model()
        
        if self.vector_store:
            self.vector_store.save()
        
        logger.info("시스템 정리 완료")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="PDF QA 시스템")
    parser.add_argument("--mode", choices=["interactive", "server", "process"], 
                       default="interactive", help="실행 모드")
    parser.add_argument("--pdf", type=str, help="처리할 PDF 파일 경로")
    parser.add_argument("--question", type=str, help="질문 (process 모드)")
    parser.add_argument("--model-type", choices=["ollama", "huggingface", "llama_cpp"],
                       default="ollama", help="사용할 모델 타입")
    parser.add_argument("--model-name", type=str, default="llama2:7b", 
                       help="모델 이름")
    parser.add_argument("--embedding-model", type=str, 
                       default="jhgan/ko-sroberta-multitask",
                       help="임베딩 모델")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")
    
    args = parser.parse_args()
    
    # 시스템 초기화
    system = PDFQASystem(
        model_type=args.model_type,
        model_name=args.model_name,
        embedding_model=args.embedding_model
    )
    
    try:
        # 컴포넌트 초기화
        if not system.initialize_components():
            logger.error("시스템 초기화 실패")
            sys.exit(1)
        
        # 모드별 실행
        if args.mode == "server":
            logger.info(f"API 서버 시작: http://{args.host}:{args.port}")
            run_server(host=args.host, port=args.port)
            
        elif args.mode == "process":
            if not args.pdf or not args.question:
                logger.error("process 모드에서는 --pdf와 --question이 필요합니다.")
                sys.exit(1)
            
            # PDF 처리
            pdf_result = system.process_pdf(args.pdf)
            print(f"PDF 처리 완료: {pdf_result}")
            
            # 질문 처리
            qa_result = system.ask_question(args.question)
            print(f"답변: {qa_result['answer']}")
            
        else:  # interactive 모드
            if args.pdf:
                # PDF 먼저 처리
                pdf_result = system.process_pdf(args.pdf)
                print(f"PDF 처리 완료: {pdf_result['filename']} ({pdf_result['total_chunks']}개 청크)")
            
            # 대화형 모드 시작
            system.interactive_mode()
    
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"시스템 실행 중 오류: {e}")
        sys.exit(1)
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()
