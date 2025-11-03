#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대화형 챗봇 테스트 스크립트

evaluate_qa_unified처럼 로컬에서 직접 RAG 파이프라인을 사용하여 질문을 할 수 있는 대화형 스크립트입니다.
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.types import Chunk
from modules.pipeline.rag_pipeline import RAGPipeline
from config.pipeline_config import PipelineConfig
from config.model_config import ModelConfig, EmbeddingModelConfig, LLMModelConfig
from modules.core.logger import setup_logging, get_logger
from scripts.unified_evaluation import UnifiedEvaluator

setup_logging(log_dir="logs", log_level="INFO", log_format="simple")
logger = get_logger(__name__)


def load_chunks_from_corpus(corpus_path: str) -> List[Chunk]:
    """JSONL corpus 파일에서 청크 로드"""
    chunks = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            neighbor_hint = data.get("neighbor_hint")
            if neighbor_hint and isinstance(neighbor_hint, list):
                neighbor_hint = tuple(neighbor_hint)
            
            chunk = Chunk(
                doc_id=data["doc_id"],
                filename=data["filename"],
                page=data.get("page"),
                start_offset=data.get("start_offset", 0),
                length=data.get("length", len(data["text"])),
                text=data["text"],
                neighbor_hint=neighbor_hint,
                extra=data.get("extra", {}),
            )
            chunks.append(chunk)
    
    return chunks


def print_answer(answer, processing_time: float):
    """답변을 보기 좋게 출력"""
    print("\n" + "=" * 80)
    print("🤖 챗봇 답변:")
    print("=" * 80)
    print(f"\n{answer.text}\n")
    
    # 메타데이터 출력
    print("-" * 80)
    print(f"신뢰도: {answer.confidence:.3f}")
    print(f"처리 시간: {processing_time:.2f}초")
    print(f"출처 수: {len(answer.sources)}")
    
    # 상위 3개 출처 출력
    if answer.sources:
        print("\n📚 주요 출처:")
        for i, source in enumerate(answer.sources[:3], 1):
            print(f"\n[{i}] {source.chunk.filename} (페이지: {source.chunk.page})")
            print(f"    점수: {source.score:.4f}")
            print(f"    내용: {source.chunk.text[:150]}...")
    
    print("=" * 80 + "\n")


def evaluate_answer(evaluator: UnifiedEvaluator, question: str, answer_text: str, 
                   gold_answer: str = None, keywords: List[str] = None):
    """답변 평가 (선택적)"""
    if not gold_answer:
        print("\n💡 평가를 위해 정답(gold_answer)이 필요합니다.")
        return
    
    try:
        contexts = [src.chunk.text for src in getattr(answer, 'sources', [])]
        results = evaluator.evaluate_all(
            question=question,
            prediction=answer_text,
            ground_truth=gold_answer,
            contexts=contexts,
            keywords=keywords or []
        )
        
        print("\n" + "=" * 80)
        print("📊 평가 결과:")
        print("=" * 80)
        
        summary = results['summary']
        print(f"\n기본 Score (v5): {summary.get('basic_v5_score', 0)*100:.1f}%")
        
        if 'faithfulness' in summary:
            print(f"Faithfulness: {summary['faithfulness']*100:.1f}%")
            print(f"Answer Correctness: {summary['answer_correctness']*100:.1f}%")
        
        if 'token_f1' in summary:
            print(f"Token F1: {summary['token_f1']*100:.1f}%")
            print(f"ROUGE-L: {summary['rouge_l']*100:.1f}%")
        
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 평가 중 오류 발생: {e}\n")


def interactive_chat(pipeline: RAGPipeline, evaluator: Optional[UnifiedEvaluator] = None):
    """대화형 챗봇"""
    print("\n" + "=" * 80)
    print("🤖 대화형 챗봇 테스트 모드")
    print("=" * 80)
    print("\n명령어:")
    print("  - 'exit' 또는 'quit': 종료")
    print("  - 'clear': 화면 지우기")
    print("  - 'eval <질문> | <정답> | <키워드1,키워드2>': 답변 평가 (선택적)")
    print("  - 그 외: 일반 질문\n")
    print("=" * 80 + "\n")
    
    while True:
        try:
            # 사용자 입력
            question = input("질문: ").strip()
            
            if not question:
                continue
            
            # 명령어 처리
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 테스트를 종료합니다.\n")
                break
            
            if question.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            # 평가 모드 (예: eval 질문 | 정답 | 키워드1,키워드2)
            if question.startswith('eval '):
                parts = question[5:].split('|')
                if len(parts) >= 2:
                    question = parts[0].strip()
                    gold_answer = parts[1].strip()
                    keywords = parts[2].strip().split(',') if len(parts) > 2 else []
                else:
                    print("❌ 잘못된 평가 명령어 형식입니다.")
                    print("   형식: eval 질문 | 정답 | 키워드1,키워드2")
                    continue
            
            # 질문 처리
            start_time = time.time()
            answer = pipeline.ask(question, top_k=50)
            processing_time = time.time() - start_time
            
            # 답변 출력
            print_answer(answer, processing_time)
            
            # 평가 실행 (평가 모드인 경우)
            if question.startswith('eval ') or 'gold_answer' in locals():
                if evaluator and 'gold_answer' in locals():
                    evaluate_answer(evaluator, question, answer.text, gold_answer, keywords)
                    del gold_answer
                elif not evaluator:
                    print("⚠️ 평가 모듈이 초기화되지 않았습니다.\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 테스트를 종료합니다.\n")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}\n")
            logger.error(f"질문 처리 실패: {e}", exc_info=True)


def batch_test(pipeline: RAGPipeline, questions: List[str]):
    """배치 테스트 모드"""
    print("\n" + "=" * 80)
    print("📋 배치 테스트 모드")
    print("=" * 80)
    print(f"총 {len(questions)}개 질문을 처리합니다.\n")
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 질문: {question}")
        
        try:
            start_time = time.time()
            answer = pipeline.ask(question, top_k=50)
            processing_time = time.time() - start_time
            
            print(f"답변: {answer.text[:100]}...")
            print(f"신뢰도: {answer.confidence:.3f}, 시간: {processing_time:.2f}초")
            
            results.append({
                'question': question,
                'answer': answer.text,
                'confidence': answer.confidence,
                'time': processing_time,
                'num_sources': len(answer.sources)
            })
            
        except Exception as e:
            print(f"❌ 오류: {e}")
            results.append({
                'question': question,
                'error': str(e)
            })
    
    print("\n" + "=" * 80)
    print("✅ 배치 테스트 완료!")
    print("=" * 80)
    
    # 통계 출력
    successful = [r for r in results if 'error' not in r]
    if successful:
        avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
        avg_time = sum(r['time'] for r in successful) / len(successful)
        print(f"\n성공: {len(successful)}/{len(questions)}")
        print(f"평균 신뢰도: {avg_confidence:.3f}")
        print(f"평균 처리 시간: {avg_time:.2f}초\n")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="대화형 챗봇 테스트 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 대화형 모드
  python scripts/test_chatbot_interactive.py
  
  # 배치 모드 (qa.json 파일 사용)
  python scripts/test_chatbot_interactive.py --qa data/qa.json --batch
  
  # 다른 corpus 파일 사용
  python scripts/test_chatbot_interactive.py --corpus data/my_corpus.jsonl
  
  # 평가 모듈 사용 안 함
  python scripts/test_chatbot_interactive.py --no-eval
        """
    )
    
    parser.add_argument("--corpus", default="data/corpus.jsonl", help="Corpus 파일 경로")
    parser.add_argument("--config", default="config/default.yaml", help="설정 파일 경로")
    parser.add_argument("--qa", default=None, help="QA 파일 경로 (배치 모드용)")
    parser.add_argument("--batch", action="store_true", help="배치 테스트 모드")
    parser.add_argument("--top-k", type=int, default=50, help="검색 결과 수")
    parser.add_argument("--model", default="qwen2.5:3b-instruct-q4_K_M", help="LLM 모델명")
    parser.add_argument("--no-eval", action="store_true", help="평가 모듈 사용 안 함")
    parser.add_argument("--mode", default="accuracy", choices=["accuracy", "speed"], help="실행 모드")
    
    args = parser.parse_args()
    
    # 경로 변환
    project_root = Path(__file__).parent.parent
    corpus_path = project_root / args.corpus
    config_path = project_root / args.config
    
    logger.info("=" * 80)
    logger.info("대화형 챗봇 테스트 시작")
    logger.info("=" * 80)
    
    # Corpus 로드
    logger.info(f"Corpus 로딩: {corpus_path}")
    if not corpus_path.exists():
        logger.error(f"Corpus 파일이 없습니다: {corpus_path}")
        sys.exit(1)
    
    chunks = load_chunks_from_corpus(str(corpus_path))
    logger.info(f"Corpus 로드 완료: {len(chunks)}개 청크")
    
    # 설정 로드
    pipeline_config = PipelineConfig()
    if config_path.exists():
        logger.info(f"설정 로드: {config_path}")
        pipeline_config = PipelineConfig.from_file(config_path)
    
    pipeline_config.flags.mode = args.mode
    
    # LLM 모델 확인
    logger.info(f"LLM 모델 확인: {args.model}")
    try:
        from modules.generation.ollama_manager import ollama_manager
        if not ollama_manager.ensure_model_available(args.model):
            logger.error(f"LLM 모델을 사용할 수 없습니다: {args.model}")
            logger.info(f"수동 설치: ollama pull {args.model}")
            sys.exit(1)
    except Exception as e:
        logger.warning(f"모델 자동 설치 확인 실패: {e}")
    
    # 파이프라인 초기화
    logger.info("RAG 파이프라인 초기화 중...")
    model_config = ModelConfig(
        embedding=EmbeddingModelConfig(device="cuda"),
        llm=LLMModelConfig(
            host="localhost",
            port=11434,
            model_name=args.model
        )
    )
    
    try:
        pipeline = RAGPipeline(
            chunks=chunks,
            pipeline_config=pipeline_config,
            model_config=model_config,
            evaluation_mode=False,
        )
        logger.info("파이프라인 초기화 완료")
    except Exception as e:
        logger.error(f"파이프라인 초기화 실패: {e}", exc_info=True)
        sys.exit(1)
    
    # 평가 모듈 초기화 (선택적)
    evaluator = None
    if not args.no_eval:
        try:
            evaluator = UnifiedEvaluator()
            logger.info("평가 모듈 초기화 완료")
        except Exception as e:
            logger.warning(f"평가 모듈 초기화 실패: {e}")
            logger.info("평가 기능 없이 계속 진행합니다.")
    
    # 배치 모드 또는 대화형 모드
    if args.batch:
        if args.qa:
            qa_path = project_root / args.qa
            if not qa_path.exists():
                logger.error(f"QA 파일이 없습니다: {qa_path}")
                sys.exit(1)
            
            with open(qa_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            questions = [item['question'] for item in qa_data]
            batch_test(pipeline, questions)
        else:
            logger.error("배치 모드에서는 --qa 옵션이 필요합니다.")
            sys.exit(1)
    else:
        interactive_chat(pipeline, evaluator)


if __name__ == "__main__":
    main()

