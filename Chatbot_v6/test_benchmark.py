#!/usr/bin/env python3
"""
간단한 벤치마크 테스트

더미 데이터로 시스템이 정상 작동하는지 확인합니다.
"""

import sys
import json
import time
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.core.types import Chunk, Answer
from modules.pipeline.rag_pipeline import RAGPipeline
from config.pipeline_config import PipelineConfig
from config.model_config import ModelConfig, EmbeddingModelConfig, LLMModelConfig

print("=" * 60)
print("챗봇 v6 벤치마크 테스트")
print("=" * 60)

# 1. 더미 청크 생성
print("\n[1] 더미 청크 생성...")
dummy_chunks = [
    Chunk(
        doc_id="demo",
        filename="고산정수장_매뉴얼.pdf",
        page=1,
        start_offset=0,
        length=200,
        text="고산 정수장 AI플랫폼 URL은 http://waio-portal-vip:10011 입니다. 로그인 계정은 관리자(KWATER)이며, 비밀번호도 KWATER입니다.",
    ),
    Chunk(
        doc_id="demo",
        filename="고산정수장_매뉴얼.pdf",
        page=2,
        start_offset=200,
        length=180,
        text="발주기관은 한국수자원공사이며, 사업명은 금강 유역 남부 스마트 정수장 확대 구축 용역입니다. 발행일은 2025년 2월 17일입니다.",
    ),
    Chunk(
        doc_id="demo",
        filename="고산정수장_매뉴얼.pdf",
        page=3,
        start_offset=380,
        length=150,
        text="AI 플랫폼 대시보드는 상단 영역, 좌측 사이드바 영역, 중앙 콘텐츠 영역, 우측 메뉴 영역의 4가지 영역으로 구성됩니다.",
    ),
    Chunk(
        doc_id="demo",
        filename="수질기준.pdf",
        page=1,
        start_offset=0,
        length=100,
        text="pH는 5.8~8.6 범위여야 합니다. 탁도는 0.5 NTU 이하로 유지해야 합니다.",
    ),
    Chunk(
        doc_id="demo",
        filename="수질기준.pdf",
        page=2,
        start_offset=100,
        length=120,
        text="잔류염소는 0.1~4.0 mg/L 범위를 유지해야 하며, 대장균은 검출되지 않아야 합니다.",
    ),
]

print(f"[OK] 총 {len(dummy_chunks)}개 청크 생성 완료")

# 2. 설정 로드
print("\n[2] 설정 로드...")
try:
    config_path = project_root / "config" / "default.yaml"
    pipeline_config = PipelineConfig.from_file(config_path)
    print(f"[OK] 설정 파일 로드: {config_path}")
except Exception as e:
    print(f"[WARN] 설정 파일 로드 실패, 기본 설정 사용: {e}")
    pipeline_config = PipelineConfig()

# 모델 설정
embedding_config = EmbeddingModelConfig()
llm_config = LLMModelConfig()
model_config = ModelConfig(
    embedding=embedding_config,
    llm=llm_config
)

# 3. 파이프라인 초기화
print("\n[3] RAG 파이프라인 초기화...")
try:
    pipeline = RAGPipeline(
        chunks=dummy_chunks,
        pipeline_config=pipeline_config,
        model_config=model_config,
    )
    print("[OK] 파이프라인 초기화 완료")
except Exception as e:
    print(f"[ERROR] 파이프라인 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 실제 QA 데이터 로드
print("\n[4] QA 데이터 로드...")
try:
    with open("data/qa.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    print(f"[OK] QA 데이터 로드 완료: {len(qa_data)}개 질문")
except Exception as e:
    print(f"[ERROR] QA 데이터 로드 실패: {e}")
    sys.exit(1)

print("\n[5] 테스트 질문 실행...")
print("-" * 60)

# QA 데이터에서 질문만 추출
test_questions = [qa["question"] for qa in qa_data]

results = []

for i, question in enumerate(test_questions, 1):
    print(f"\n[질문 {i}/{len(test_questions)}]")
    print(f"Q: {question}")
    
    try:
        start_time = time.time()
        answer = pipeline.ask(question, top_k=10)
        elapsed = time.time() - start_time
        
        print(f"A: {answer.text}")
        print(f"신뢰도: {answer.confidence:.3f}")
        print(f"소스: {len(answer.sources)}개")
        print(f"처리시간: {elapsed:.2f}초")
        
        results.append({
            "question": question,
            "answer": answer.text,
            "confidence": answer.confidence,
            "time": elapsed,
            "sources": len(answer.sources),
        })
        
        print("[OK] 성공")
        
    except Exception as e:
        print(f"[ERROR] 오류: {e}")
        import traceback
        traceback.print_exc()
        
        results.append({
            "question": question,
            "error": str(e),
        })

# 5. 결과 요약
print("\n" + "=" * 60)
print("[결과] 벤치마크 결과 요약")
print("=" * 60)

success_count = sum(1 for r in results if "error" not in r)
total_count = len(results)

print(f"\n총 질문 수: {total_count}")
print(f"성공: {success_count}")
print(f"실패: {total_count - success_count}")

if success_count > 0:
    avg_time = sum(r["time"] for r in results if "error" not in r) / success_count
    avg_confidence = sum(r["confidence"] for r in results if "error" not in r) / success_count
    
    print(f"\n평균 응답 시간: {avg_time:.2f}초")
    print(f"평균 신뢰도: {avg_confidence:.3f}")

# 6. 결과 저장
output_file = project_root / "benchmark_results_simple.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total_count,
        "success_count": success_count,
        "results": results,
    }, f, ensure_ascii=False, indent=2)

print(f"\n[OK] 결과 저장: {output_file}")
print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)

