#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 평가 시스템을 사용한 QA 벤치마크

qa.json의 질문들에 대해 답변을 생성하고 통합 평가 모듈로 평가합니다.
모든 평가 지표(기본 Score, 도메인 특화, RAG 3대, 학술 표준)를 한 번에 계산합니다.
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.core.types import Chunk
from modules.pipeline.rag_pipeline import RAGPipeline
from config.pipeline_config import PipelineConfig
from config.model_config import ModelConfig, EmbeddingModelConfig, LLMModelConfig
from modules.core.logger import setup_logging, get_logger
from scripts.unified_evaluation import UnifiedEvaluator

setup_logging(log_dir="logs", log_level="INFO", log_format="json")
logger = get_logger(__name__)


class UnifiedQABenchmark:
    """통합 평가 시스템을 사용한 QA 벤치마크"""
    
    def __init__(
        self,
        pipeline: RAGPipeline,
        qa_data: List[Dict[str, Any]],
    ):
        self.pipeline = pipeline
        self.qa_data = qa_data
        self.evaluator = UnifiedEvaluator()
        self.results = []
    
    def run(self) -> Dict[str, Any]:
        """벤치마크 실행"""
        logger.info(f"=== 통합 평가 벤치마크 시작 ===")
        logger.info(f"총 질문 수: {len(self.qa_data)}")
        
        start_time = time.time()
        qa_pairs = []
        
        for i, item in enumerate(self.qa_data, 1):
            q_id = item.get("id", i)
            question = item["question"]
            gold_answer = item.get("answer", "")
            keywords = item.get("accepted_keywords", [])
            
            logger.info(f"\n[{i}/{len(self.qa_data)}] 질문 처리 중...")
            logger.info(f"질문: {question}")
            
            try:
                # 답변 생성
                t0 = time.time()
                answer = self.pipeline.ask(question, top_k=50)
                elapsed_ms = int((time.time() - t0) * 1000)
                
                # Context 텍스트 추출
                context_texts = [src.chunk.text for src in answer.sources]
                
                # 통합 평가 실행
                eval_results = self.evaluator.evaluate_all(
                    question=question,
                    prediction=answer.text,
                    ground_truth=gold_answer,
                    contexts=context_texts,
                    keywords=keywords
                )
                
                logger.info(f"답변: {answer.text[:100]}...")
                logger.info(f"기본 Score: {eval_results['summary']['basic_v5_score']:.3f}")
                logger.info(f"신뢰도: {answer.confidence:.3f}, 시간: {elapsed_ms}ms")
                
                # 결과 저장
                result = {
                    "id": q_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "prediction": answer.text,
                    "keywords": keywords,
                    
                    # 기본 정보
                    "confidence": answer.confidence,
                    "num_sources": len(answer.sources),
                    "time_ms": elapsed_ms,
                    
                    # 통합 평가 결과
                    "evaluation": {
                        "basic_score": eval_results['basic_score'],
                        "domain_specific": eval_results['domain_specific'],
                        "rag_metrics": eval_results['rag_metrics'],
                        "academic_metrics": eval_results['academic_metrics'],
                        "summary": eval_results['summary']
                    },
                    
                    # 파이프라인 메트릭
                    "pipeline_metrics": answer.metrics,
                }
                
                self.results.append(result)
                
                # 배치 평가용 데이터 수집
                qa_pairs.append({
                    'question': question,
                    'prediction': answer.text,
                    'ground_truth': gold_answer,
                    'contexts': context_texts,
                    'keywords': keywords
                })
            
            except Exception as e:
                logger.error(f"질문 처리 실패: {e}", exc_info=True)
                
                # 실패 결과
                self.results.append({
                    "id": q_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "prediction": f"[ERROR] {str(e)}",
                    "keywords": keywords,
                    "confidence": 0.0,
                    "num_sources": 0,
                    "time_ms": 0,
                    "error": str(e),
                })
        
        total_time = time.time() - start_time
        
        # 통계 계산
        stats = self._calculate_stats(total_time)
        
        logger.info("\n=== 벤치마크 완료 ===")
        logger.info(f"총 질문: {stats['total_questions']}")
        logger.info(f"성공: {stats['successful']}, 실패: {stats['failed']}")
        logger.info(f"평균 기본 Score: {stats.get('avg_basic_v5_score', 0):.3f}")
        logger.info(f"평균 시간: {stats['avg_time_ms']:.1f}ms")
        logger.info(f"총 소요 시간: {total_time:.1f}초")
        
        return {
            "stats": stats,
            "results": self.results,
        }
    
    def _calculate_stats(self, total_time: float) -> Dict[str, Any]:
        """통계 계산"""
        valid_results = [r for r in self.results if "error" not in r]
        
        if not valid_results:
            return {
                "total_questions": len(self.qa_data),
                "successful": 0,
                "failed": len(self.qa_data),
                "timestamp": datetime.now().isoformat(),
            }
        
        # 기본 통계
        stats = {
            "total_questions": len(self.qa_data),
            "successful": len(valid_results),
            "failed": len(self.qa_data) - len(valid_results),
            "total_time_seconds": total_time,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 시간 통계
        times = [r["time_ms"] for r in valid_results]
        stats["avg_time_ms"] = sum(times) / len(times) if times else 0.0
        stats["min_time_ms"] = min(times) if times else 0.0
        stats["max_time_ms"] = max(times) if times else 0.0
        
        # 평가 점수 통계 (각 지표별)
        metrics_to_aggregate = [
            'basic_v5_score',
            'domain_total_score',
            'numeric_accuracy',
            'unit_accuracy',
            'keyword_accuracy',
            'faithfulness',
            'answer_correctness',
            'context_precision',
            'rag_overall',
            'token_f1',
            'rouge_l',
            'bleu_2',
            'exact_match'
        ]
        
        for metric in metrics_to_aggregate:
            values = []
            for r in valid_results:
                summary = r.get('evaluation', {}).get('summary', {})
                if metric in summary:
                    values.append(summary[metric])
            
            if values:
                stats[f'avg_{metric}'] = sum(values) / len(values)
                stats[f'min_{metric}'] = min(values)
                stats[f'max_{metric}'] = max(values)
        
        return stats
    
    def save_report(self, output_path: str):
        """결과 저장 (JSON + 요약 텍스트)"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 통계 재계산
        stats = self._calculate_stats(0)
        
        # JSON 저장
        report = {
            "stats": stats,
            "results": self.results,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"결과 저장: {output_path}")
        
        # 텍스트 요약 파일 생성
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        self._save_text_summary(summary_path, stats)
        
        logger.info(f"요약 저장: {summary_path}")
    
    def _save_text_summary(self, summary_path: Path, stats: Dict):
        """텍스트 요약 파일 생성"""
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("통합 평가 시스템 - QA 벤치마크 결과\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"평가 일시: {stats['timestamp']}\n")
            f.write(f"총 질문: {stats['total_questions']}개\n")
            f.write(f"성공: {stats['successful']}개 / 실패: {stats['failed']}개\n\n")
            
            # 1. 기본 Score (v5 방식)
            f.write("=" * 80 + "\n")
            f.write("1️⃣  기본 Score (v5 방식) - 실무 성능\n")
            f.write("=" * 80 + "\n\n")
            if 'avg_basic_v5_score' in stats:
                f.write(f"평균 점수:  {stats['avg_basic_v5_score']*100:6.1f}%\n")
                f.write(f"최고 점수:  {stats['max_basic_v5_score']*100:6.1f}%\n")
                f.write(f"최저 점수:  {stats['min_basic_v5_score']*100:6.1f}%\n\n")
            
            # 2. 도메인 특화
            f.write("=" * 80 + "\n")
            f.write("2️⃣  도메인 특화 평가 - 정수장 특화\n")
            f.write("=" * 80 + "\n\n")
            if 'avg_domain_total_score' in stats:
                f.write(f"종합 점수:  {stats['avg_domain_total_score']*100:6.1f}%\n")
                f.write(f"숫자 정확도: {stats['avg_numeric_accuracy']*100:6.1f}%\n")
                f.write(f"단위 정확도: {stats['avg_unit_accuracy']*100:6.1f}%\n")
                f.write(f"키워드 정확도: {stats['avg_keyword_accuracy']*100:6.1f}%\n\n")
            
            # 3. RAG 핵심 3대 지표
            f.write("=" * 80 + "\n")
            f.write("3️⃣  RAG 핵심 3대 지표 - 학술 연구용\n")
            f.write("=" * 80 + "\n\n")
            if 'avg_faithfulness' in stats:
                f.write(f"Faithfulness (충실성):      {stats['avg_faithfulness']*100:6.1f}%\n")
                f.write(f"Answer Correctness (정확도): {stats['avg_answer_correctness']*100:6.1f}%\n")
                f.write(f"Context Precision (정밀도):  {stats['avg_context_precision']*100:6.1f}%\n")
                f.write(f"RAG 종합 점수:             {stats['avg_rag_overall']*100:6.1f}%\n\n")
            
            # 4. 학술 표준 지표
            f.write("=" * 80 + "\n")
            f.write("4️⃣  학술 표준 지표 - 범용 NLP 평가\n")
            f.write("=" * 80 + "\n\n")
            if 'avg_token_f1' in stats:
                f.write(f"Token F1:    {stats['avg_token_f1']*100:6.1f}%\n")
                f.write(f"ROUGE-L:     {stats['avg_rouge_l']*100:6.1f}%\n")
                f.write(f"BLEU-2:      {stats['avg_bleu_2']*100:6.1f}%\n")
                f.write(f"Exact Match: {stats['avg_exact_match']*100:6.1f}%\n\n")
            
            # 성능 지표
            f.write("=" * 80 + "\n")
            f.write("⏱️  성능 지표\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"평균 응답 시간: {stats['avg_time_ms']/1000:.2f}초\n")
            f.write(f"최소 응답 시간: {stats['min_time_ms']/1000:.2f}초\n")
            f.write(f"최대 응답 시간: {stats['max_time_ms']/1000:.2f}초\n")
            f.write(f"총 소요 시간:  {stats['total_time_seconds']:.1f}초\n\n")
            
            # 질문별 상세 결과
            f.write("=" * 80 + "\n")
            f.write("📋 질문별 상세 결과\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(self.results, 1):
                f.write(f"[{i}] {result['question']}\n")
                
                if 'error' in result:
                    f.write(f"    ❌ 오류: {result['error']}\n")
                else:
                    summary = result['evaluation']['summary']
                    f.write(f"    기본 Score: {summary['basic_v5_score']*100:5.1f}%")
                    f.write(f" | Faithfulness: {summary.get('faithfulness', 0)*100:5.1f}%")
                    f.write(f" | Token F1: {summary['token_f1']*100:5.1f}%\n")
                    f.write(f"    답변: {result['prediction'][:80]}...\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")


def load_qa_data(path: str) -> List[Dict[str, Any]]:
    """QA 데이터 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def auto_build_corpus(pdf_dir: str, output_path: str) -> bool:
    """
    Corpus 자동 생성 (PDF 추출 -> 청킹 -> 저장)
    
    Args:
        pdf_dir: PDF 파일이 있는 디렉토리
        output_path: 생성할 corpus 파일 경로
        
    Returns:
        성공 여부
    """
    try:
        # build_corpus의 process_pdf 함수를 임포트하여 사용
        from scripts.build_corpus import process_pdf, save_corpus
        from pathlib import Path
        
        pdf_dir_path = Path(pdf_dir)
        pdf_files = list(pdf_dir_path.glob("*.pdf"))
        
        if not pdf_files:
            return False
        
        all_chunks = []
        
        for pdf_path in pdf_files:
            logger.info(f"  처리 중: {pdf_path.name}")
            try:
                chunks = process_pdf(
                    pdf_path,
                    use_ocr_correction=False,  # 자동화 시 빠른 처리를 위해 비활성화
                    use_page_based_chunking=True,
                )
                all_chunks.extend(chunks)
                logger.info(f"    ✅ {len(chunks)}개 청크 생성")
            except Exception as e:
                logger.error(f"    ❌ 실패: {e}")
                continue
        
        if not all_chunks:
            return False
        
        # Corpus 저장
        save_corpus(all_chunks, Path(output_path))
        
        logger.info(f"📊 총 {len(all_chunks)}개 청크 생성됨")
        
        # 통계 출력
        measurements_count = sum(1 for chunk in all_chunks if chunk.extra.get('measurements'))
        neighbor_count = sum(1 for chunk in all_chunks if chunk.neighbor_hint)
        
        logger.info(f"   - 측정값 포함: {measurements_count}/{len(all_chunks)}")
        logger.info(f"   - 이웃 정보 포함: {neighbor_count}/{len(all_chunks)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Corpus 자동 생성 중 오류: {e}", exc_info=True)
        return False


def load_chunks_from_corpus(corpus_path: str) -> List[Chunk]:
    """
    JSONL corpus 파일에서 청크 로드 (확장된 메타데이터 포함)
    
    개선된 청킹 시스템의 모든 메타데이터를 로드합니다:
    - neighbor_hint: 이웃 청크 정보
    - extra: 측정값 등 추가 메타데이터
    """
    chunks = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            # neighbor_hint 처리 (tuple로 변환)
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
                neighbor_hint=neighbor_hint,  # 이웃 정보 복원
                extra=data.get("extra", {}),  # 측정값 등 추가 메타데이터 복원
            )
            chunks.append(chunk)
    
    return chunks


def main():
    """메인 실행 함수"""
    # 프로젝트 루트 경로
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 현재 작업 디렉토리 저장
    import os
    original_cwd = Path(os.getcwd())
    
    # 기본 경로
    default_qa = str(project_root / "data" / "qa.json")
    default_corpus = str(project_root / "data" / "corpus.jsonl")
    default_config = str(project_root / "config" / "default.yaml")
    default_output = str(project_root / "out" / "benchmarks" / "qa_unified_result.json")
    
    parser = argparse.ArgumentParser(
        description="통합 평가 시스템을 사용한 QA 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (qa.json 사용)
  python scripts/evaluate_qa_unified.py
  
  # 다른 QA 파일 사용
  python scripts/evaluate_qa_unified.py --qa data/qa_test5.json
  
  # 출력 파일 지정
  python scripts/evaluate_qa_unified.py --output out/my_result.json
        """
    )
    parser.add_argument("--qa", default=default_qa, help="QA 데이터 파일 (기본: data/qa.json)")
    parser.add_argument("--corpus", default=default_corpus, help="Corpus 파일 (기본: data/corpus.jsonl)")
    parser.add_argument("--config", default=default_config, help="설정 파일 (기본: config/default.yaml)")
    parser.add_argument("--output", default=default_output, help="출력 파일 (기본: out/benchmarks/qa_unified_result.json)")
    parser.add_argument("--top-k", type=int, default=50, help="검색 결과 수 (기본: 50)")
    parser.add_argument("--mode", default="accuracy", choices=["accuracy", "speed"], help="실행 모드 (기본: accuracy)")
    parser.add_argument("--model", default="qwen2.5:3b-instruct-q4_K_M", help="LLM 모델명")
    args = parser.parse_args()
    
    # 경로를 절대 경로로 변환
    args.qa = str((original_cwd / args.qa).resolve() if not Path(args.qa).is_absolute() else Path(args.qa))
    args.corpus = str((original_cwd / args.corpus).resolve() if not Path(args.corpus).is_absolute() else Path(args.corpus))
    args.config = str((original_cwd / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    args.output = str((original_cwd / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output))
    
    # 작업 디렉토리를 프로젝트 루트로 변경
    os.chdir(project_root)
    
    logger.info("=" * 80)
    logger.info("통합 평가 시스템 - QA 벤치마크")
    logger.info("=" * 80)
    logger.info(f"프로젝트 루트: {project_root}")
    
    # QA 데이터 로드
    logger.info(f"\n📖 QA 데이터 로딩: {args.qa}")
    if not Path(args.qa).exists():
        logger.error(f"QA 파일이 없습니다: {args.qa}")
        sys.exit(1)
    
    qa_data = load_qa_data(args.qa)
    logger.info(f"✅ QA 데이터 로드 완료: {len(qa_data)}개 질문")
    
    # 설정 로드
    logger.info(f"\n⚙️  파이프라인 설정 로딩: {args.config}")
    config_path = Path(args.config)
    if config_path.exists():
        pipeline_config = PipelineConfig.from_file(config_path)
    else:
        logger.warning(f"설정 파일 없음, 기본 설정 사용")
        pipeline_config = PipelineConfig()
    
    pipeline_config.flags.mode = args.mode
    
    # Corpus 로드 (없으면 자동 생성)
    logger.info(f"\n📚 Corpus 로딩: {args.corpus}")
    if not Path(args.corpus).exists():
        logger.warning(f"⚠️  Corpus 파일이 없습니다. 자동으로 생성합니다...")
        
        # PDF 디렉토리 확인
        pdf_dir = project_root / "data"
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.error(f"❌ PDF 파일이 없습니다: {pdf_dir}")
            logger.error("data/ 디렉토리에 PDF 파일을 넣어주세요.")
            sys.exit(1)
        
        logger.info(f"📄 {len(pdf_files)}개 PDF 파일 발견")
        logger.info("🔧 Corpus 자동 생성 중...")
        
        # build_corpus 자동 실행
        success = auto_build_corpus(
            pdf_dir=str(pdf_dir),
            output_path=args.corpus
        )
        
        if not success:
            logger.error("❌ Corpus 생성 실패")
            sys.exit(1)
        
        logger.info(f"✅ Corpus 자동 생성 완료: {args.corpus}")
    
    chunks = load_chunks_from_corpus(args.corpus)
    logger.info(f"✅ Corpus 로드 완료: {len(chunks)}개 청크")
    
    # 파이프라인 초기화
    logger.info("\n🚀 RAG 파이프라인 초기화 중...")
    model_config = ModelConfig(
        embedding=EmbeddingModelConfig(device="cpu"),
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
            evaluation_mode=True,  # 평가 모드 활성화
        )
        logger.info("✅ 파이프라인 초기화 완료 (평가 모드)")
    except Exception as e:
        logger.error(f"❌ 파이프라인 초기화 실패: {e}", exc_info=True)
        sys.exit(1)
    
    # 벤치마크 실행
    logger.info("\n" + "=" * 80)
    logger.info("📊 벤치마크 실행 시작")
    logger.info("=" * 80)
    
    benchmark = UnifiedQABenchmark(pipeline, qa_data)
    
    try:
        result = benchmark.run()
        
        # 결과 저장
        benchmark.save_report(args.output)
        
        # 최종 요약 출력
        stats = result['stats']
        
        print("\n" + "=" * 80)
        print("✅ 벤치마크 완료!")
        print("=" * 80)
        
        print("\n📊 평가 결과 요약:")
        print(f"\n1️⃣  기본 Score (v5):        {stats.get('avg_basic_v5_score', 0)*100:6.1f}%")
        print(f"2️⃣  도메인 특화 종합:        {stats.get('avg_domain_total_score', 0)*100:6.1f}%")
        print(f"    - 숫자 정확도:          {stats.get('avg_numeric_accuracy', 0)*100:6.1f}%")
        print(f"    - 단위 정확도:          {stats.get('avg_unit_accuracy', 0)*100:6.1f}%")
        
        if 'avg_faithfulness' in stats:
            print(f"3️⃣  RAG 핵심 지표:")
            print(f"    - Faithfulness:         {stats['avg_faithfulness']*100:6.1f}%")
            print(f"    - Answer Correctness:   {stats['avg_answer_correctness']*100:6.1f}%")
            print(f"    - Context Precision:    {stats['avg_context_precision']*100:6.1f}%")
        
        if 'avg_token_f1' in stats:
            print(f"4️⃣  학술 표준:")
            print(f"    - Token F1:             {stats['avg_token_f1']*100:6.1f}%")
            print(f"    - ROUGE-L:              {stats['avg_rouge_l']*100:6.1f}%")
        
        print(f"\n⏱️  평균 응답 시간:          {stats['avg_time_ms']/1000:.2f}초")
        
        print("\n📁 결과 파일:")
        print(f"  - JSON: {args.output}")
        print(f"  - TXT:  {Path(args.output).parent / f'{Path(args.output).stem}_summary.txt'}")
        
        print("\n" + "=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\n사용자에 의해 중단되었습니다")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 벤치마크 실행 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

