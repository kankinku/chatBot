"""
Performance Demo

개선된 기능들의 사용 예시.
"""

import asyncio
import time
from typing import List

# 개선된 모듈들
from modules.generation.async_llm_client import AsyncOllamaClient
from modules.generation.batch_answer_generator import BatchAnswerGenerator
from modules.embedding.batch_embedder import BatchEmbedder
from modules.embedding.sbert_embedder import SBERTEmbedder
from modules.monitoring.metrics import get_metrics
from modules.quality.answer_quality_checker import AnswerQualityChecker
from modules.filtering.adaptive_context_selector import AdaptiveContextSelector

from modules.core.logger import get_logger

logger = get_logger(__name__)


async def demo_async_llm():
    """비동기 LLM 데모"""
    print("\n" + "=" * 60)
    print("1. 비동기 LLM 클라이언트 데모")
    print("=" * 60)
    
    async with AsyncOllamaClient() as client:
        # 단일 요청
        print("\n[단일 요청]")
        t0 = time.time()
        answer = await client.generate_async("pH 기준은 무엇인가요?")
        elapsed = time.time() - t0
        print(f"답변: {answer[:100]}...")
        print(f"소요 시간: {elapsed:.2f}초")
        
        # 배치 요청
        print("\n[배치 요청 (5개)]")
        questions = [
            "pH 기준은?",
            "탁도 기준은?",
            "대장균 기준은?",
            "잔류염소 기준은?",
            "수온 기준은?",
        ]
        
        t0 = time.time()
        answers = await client.generate_batch(questions, max_concurrent=3)
        elapsed = time.time() - t0
        
        print(f"답변 {len(answers)}개 생성 완료")
        print(f"총 소요 시간: {elapsed:.2f}초")
        print(f"평균: {elapsed/len(answers):.2f}초/질문")
        
        for i, (q, a) in enumerate(zip(questions, answers)):
            print(f"\nQ{i+1}: {q}")
            print(f"A{i+1}: {a[:80]}...")


async def demo_batch_generation():
    """배치 답변 생성 데모"""
    print("\n" + "=" * 60)
    print("2. 배치 답변 생성 데모")
    print("=" * 60)
    
    # 더미 컨텍스트 생성 (실제로는 검색 결과)
    from modules.core.types import Chunk, RetrievedSpan
    
    dummy_chunk = Chunk(
        doc_id="doc1",
        filename="test.pdf",
        page=1,
        start_offset=0,
        length=100,
        text="pH는 5.8~8.6 범위여야 합니다.",
    )
    
    dummy_span = RetrievedSpan(
        chunk=dummy_chunk,
        source="test",
        score=0.9,
        rank=1,
    )
    
    questions = [f"질문 {i+1}" for i in range(5)]
    contexts_list = [[dummy_span] for _ in range(5)]
    
    generator = BatchAnswerGenerator(max_concurrent=3)
    
    t0 = time.time()
    answers = await generator.generate_batch(questions, contexts_list)
    elapsed = time.time() - t0
    
    print(f"\n{len(answers)}개 답변 생성 완료")
    print(f"총 소요 시간: {elapsed:.2f}초")
    print(f"평균: {elapsed/len(answers):.2f}초/질문")


def demo_batch_embedding():
    """배치 임베딩 데모"""
    print("\n" + "=" * 60)
    print("3. 배치 임베딩 데모")
    print("=" * 60)
    
    # 대량 텍스트 생성
    texts = [f"샘플 텍스트 {i}" for i in range(1000)]
    
    embedder = SBERTEmbedder()
    batch_embedder = BatchEmbedder(
        embedder,
        batch_size=64,
        show_progress=True
    )
    
    # 예상 시간
    estimated = batch_embedder.estimate_time(len(texts))
    print(f"\n{len(texts)}개 텍스트 임베딩 예상 시간: {estimated:.1f}초")
    
    # 실제 실행
    t0 = time.time()
    embeddings = batch_embedder.embed_texts_batched(texts)
    elapsed = time.time() - t0
    
    print(f"\n임베딩 완료")
    print(f"실제 소요 시간: {elapsed:.2f}초")
    print(f"임베딩 shape: {embeddings.shape}")


def demo_quality_checker():
    """품질 검증 데모"""
    print("\n" + "=" * 60)
    print("4. 답변 품질 검증 데모")
    print("=" * 60)
    
    from modules.core.types import Chunk, RetrievedSpan
    
    # 컨텍스트
    chunk = Chunk(
        doc_id="doc1",
        filename="test.pdf",
        page=1,
        start_offset=0,
        length=100,
        text="pH는 5.8~8.6 범위여야 합니다. 이는 수질 기준에 명시되어 있습니다.",
    )
    
    span = RetrievedSpan(chunk=chunk, source="test", score=0.9, rank=1)
    
    checker = AnswerQualityChecker()
    
    # 테스트 케이스들
    test_cases = [
        {
            "question": "pH 기준은 무엇인가요?",
            "answer": "pH는 5.8~8.6 범위여야 합니다.",
            "expected": "좋은 답변"
        },
        {
            "question": "pH 기준은 무엇인가요?",
            "answer": "pH는 7.0~9.0 범위여야 합니다.",  # 환각!
            "expected": "환각 감지"
        },
        {
            "question": "pH 기준은 무엇인가요?",
            "answer": "네.",  # 너무 짧음
            "expected": "불완전한 답변"
        },
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n[테스트 케이스 {i+1}: {case['expected']}]")
        print(f"질문: {case['question']}")
        print(f"답변: {case['answer']}")
        
        scores = checker.check_quality(
            case['question'],
            case['answer'],
            [span]
        )
        
        print(f"\n품질 점수:")
        print(f"  전체: {scores['overall_quality']:.3f}")
        print(f"  일치도: {scores['context_alignment']:.3f}")
        print(f"  환각: {scores['hallucination_score']:.3f}")
        print(f"  완전성: {scores['completeness']:.3f}")
        print(f"  유창성: {scores['fluency']:.3f}")
        
        should_fallback = checker.should_use_fallback(scores)
        print(f"  폴백 사용: {'예' if should_fallback else '아니오'}")


def demo_adaptive_selector():
    """적응형 컨텍스트 선택 데모"""
    print("\n" + "=" * 60)
    print("5. 적응형 컨텍스트 선택 데모")
    print("=" * 60)
    
    from modules.core.types import Chunk, RetrievedSpan
    
    # 더미 컨텍스트 생성 (다양한 점수와 문서)
    candidates = []
    for i in range(20):
        chunk = Chunk(
            doc_id=f"doc{i}",
            filename=f"file{i % 5}.pdf",  # 5개 파일
            page=i // 2,
            start_offset=0,
            length=200,
            text=f"컨텍스트 {i+1}: " + "샘플 텍스트 " * 20,
        )
        
        span = RetrievedSpan(
            chunk=chunk,
            source="hybrid",
            score=0.9 - i * 0.04,  # 점수 감소
            rank=i,
        )
        
        candidates.append(span)
    
    selector = AdaptiveContextSelector(
        llm_max_ctx=8192,
        ctx_usage_ratio=0.6,
        min_contexts=2,
        max_contexts=8,
    )
    
    print(f"\n총 후보: {len(candidates)}개")
    print(f"최대 토큰: {selector.max_ctx_tokens}")
    
    selected = selector.select_contexts(
        question="pH 기준은?",
        candidates=candidates,
        question_type="numeric"
    )
    
    print(f"\n선택된 컨텍스트: {len(selected)}개")
    for i, span in enumerate(selected):
        print(f"{i+1}. {span.chunk.filename} (page {span.chunk.page}) - 점수: {span.score:.3f}")


def demo_metrics():
    """메트릭 수집 데모"""
    print("\n" + "=" * 60)
    print("6. 메트릭 수집 데모")
    print("=" * 60)
    
    metrics = get_metrics(enabled=True)
    
    # 시스템 정보 설정
    metrics.set_system_info({
        "version": "1.0.0",
        "model": "llama3.2:3b",
        "device": "cuda"
    })
    
    # 요청 시뮬레이션
    print("\n요청 시뮬레이션 중...")
    
    for i in range(10):
        question_type = ["numeric", "definition", "general"][i % 3]
        
        with metrics.track_request(question_type=question_type):
            # 검색 단계
            with metrics.track_stage("retrieval"):
                time.sleep(0.1)  # 100ms
                metrics.record_search_results(30, "hybrid")
            
            # 필터링 단계
            with metrics.track_stage("filtering"):
                time.sleep(0.05)  # 50ms
                metrics.record_contexts_used(5)
            
            # 생성 단계
            with metrics.track_stage("generation"):
                time.sleep(0.2)  # 200ms
            
            # 캐시 히트/미스
            if i % 3 == 0:
                metrics.record_cache_hit("embedding")
            else:
                metrics.record_cache_miss("embedding")
            
            # 신뢰도
            metrics.record_confidence(0.7 + i * 0.02, question_type)
    
    print("✓ 메트릭 수집 완료")
    print("\n메트릭 확인:")
    print("  - Prometheus: http://localhost:9090")
    print("  - Grafana: http://localhost:3000")
    print("  - Endpoint: http://localhost:8000/metrics")


async def main():
    """전체 데모 실행"""
    print("\n" + "=" * 60)
    print("RAG 시스템 성능 개선 데모")
    print("=" * 60)
    
    # 1. 비동기 LLM
    await demo_async_llm()
    
    # 2. 배치 생성
    await demo_batch_generation()
    
    # 3. 배치 임베딩
    demo_batch_embedding()
    
    # 4. 품질 검증
    demo_quality_checker()
    
    # 5. 적응형 선택
    demo_adaptive_selector()
    
    # 6. 메트릭
    demo_metrics()
    
    print("\n" + "=" * 60)
    print("데모 완료!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

