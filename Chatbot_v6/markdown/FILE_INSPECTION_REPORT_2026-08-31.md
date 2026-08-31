# Chatbot_v6 파일 점검 리포트

- 점검일: 2026-08-31
- 대상: `Chatbot_v6`
- 기준 경로: `C:\Users\hanji\Documents\ChatGPT\ChatBot_Re\Chatbot_v6`
- 원격 기준: `https://github.com/kankinku/chatBot.git`, `main`
- 평가 기준: `SYSTEM VALIDATION / LOAD TEST PORTFOLIO EVALUATOR v1.0`

## 1. Executive Summary

Chatbot_v6는 RAG 파이프라인, 검색 모듈, 파이프라인 단계별 처리시간 메트릭, 단위·통합 테스트 구조를 포함하고 있다. 다만 현재 파일과 실행 증거만으로는 production-like 부하테스트와 재현 가능한 성능 개선을 입증할 수 없다.

종합 평가는 **20/100, D등급**으로 기록한다. 평가 서브 에이전트 3명의 점수는 14점, 18점, 22점이었으며, 공통 결론은 “구현은 있으나 성능·SRE Evidence가 부족하다”는 것이다.

## 2. 현재 파일 상태

- `Chatbot_v6` 경로는 원격 `main`에서 sparse checkout으로 가져와져 있다.
- 현재 작업 브랜치는 `main`이며 `origin/main`과 일치한다.
- 작업공간의 기존 미추적 `.omx/`는 변경하지 않았다.
- 주요 구성은 다음과 같다.
  - `api/app.py`: FastAPI API
  - `modules/pipeline/rag_pipeline.py`: RAG 파이프라인
  - `modules/retrieval/`: BM25·벡터·Hybrid 검색
  - `modules/monitoring/metrics.py`: 메트릭 구조
  - `scripts/`: QA 및 RAG 평가 스크립트
  - `tests/unit/`, `tests/integration/`: 테스트 스위트
  - `markdown/`: 성능·최적화·프로젝트 보고서
  - `Server/backend/`, `Server/frontend/`: Django proxy/backend 및 React frontend

## 3. 실행 점검

### 테스트

`python -m pytest Chatbot_v6\\tests -q` 실행에서 pytest가 41개 테스트를 수집했다. 통합 테스트 구간에서 실행 시간이 길어졌으며, 서브 에이전트의 완료 보고 기준 실제 결과는 다음과 같다.

- 39 passed
- 2 failed
- 문서의 기존 성공 수치와 현재 실행 결과가 일치하지 않음

실패가 보고된 대표 테스트:

- `tests/unit/test_bm25_retriever.py:65-72`
- `tests/unit/test_question_analyzer.py:91-101`

통합 테스트는 실제 SBERT 의존성에 영향을 받아 환경에 따라 setup error가 발생할 수 있다. 따라서 현재 테스트 상태를 “전체 통과”로 표시하면 안 된다.

## 4. 성능 및 포트폴리오 Evidence 점검

### 확인된 강점

1. `modules/pipeline/rag_pipeline.py:212-246`에서 retrieval, generation, rerank, total time 등 단계별 메트릭을 기록한다.
2. `modules/monitoring/metrics.py:47-100`에 요청·단계·cache·timeout 메트릭 구조가 있다.
3. `tests/unit/`, `tests/integration/`으로 테스트 책임이 구분되어 있다.
4. `markdown/PERFORMANCE_REPORT.md:140-153`에 100개 청크·5개 질의를 사용하는 소규모 벤치마크가 기록되어 있다.
5. `markdown/OPTIMIZATION_ANALYSIS_v2.md:15,36`에 병렬화의 역효과와 캐시·환경 차이에 대한 판단이 기록되어 있다.

### 확인되지 않은 핵심 Evidence

- RPS/TPS, concurrency, endpoint mix, read/write 비율
- peak/spike/soak workload
- production-like workload 선정 근거와 SLO
- baseline/candidate의 반복 실행 및 run-level raw 결과
- p50/p95/p99, error rate, timeout
- generator CPU·메모리·NIC·file descriptor·achieved RPS
- DB/cache/queue/infrastructure의 동일 시간축 메트릭
- 장애 주입, MTTR, recovery time
- 성능 회귀 시 CI job 실패 또는 배포 차단
- RPS/vCPU, CPU-seconds/transaction, 비용 등 효율 지표

## 5. Hard Gate 판정

| 항목 | 판정 | 근거 |
|---|---|---|
| Workload Validity | FAIL | RPS·concurrency·endpoint mix·peak 조건 및 선정 근거 없음 |
| Baseline Validity | FAIL | 전후 평균 수치는 있으나 반복 raw 결과·p95/p99·자원 baseline 없음 |
| Variable Control | FAIL | cache 상태·실행 환경·반복 순서·동일 조건 미입증 |
| Generator Validity | FAIL | 부하 생성기 및 generator 자원 검증 없음 |

Hard Gate 2개 이상 FAIL이므로 **PORTFOLIO EVIDENCE = WEAK**이다.

## 6. 약하거나 검증되지 않은 주장

### 평균 검색 29.88ms, 33.5 QPS, 실시간 서비스 가능

100개 청크·5개 질의의 소규모 합성 측정이다. 동시성, percentile latency, 오류율, 자원 사용량, generator capacity가 없어 전체 API 성능으로 일반화할 수 없다.

### Hybrid 검색 53%, 캐시 99% 향상

반복 run, raw 결과, cold/warm cache 분리, 통계적 검증이 없다. 일부 문서 표에서는 평균 검색시간이 29.88ms에서 33.61ms로 악화된 기록도 있다.

### 프로덕션 준비 완료

실제 API 부하, 장시간 안정성, 장애 복구, 운영 모니터링, release gate가 확인되지 않아 현재 파일 Evidence로는 지지되지 않는다.

## 7. 우선순위별 문제

- **P0:** 단일 전후 수치 중심이라 성능 결론을 재현할 수 없다.
- **P0:** 문서의 테스트 성공 수치와 현재 실행 결과가 다르다.
- **P1:** `api/app.py:449-509`의 async endpoint가 동기식 `pipeline.ask()`를 직접 호출한다.
- **P1:** 파이프라인 내부 메트릭 정의와 실제 운영 `/metrics` 노출·요청 경로 연결이 일관되게 입증되지 않는다.
- **P1:** `requirements.txt`의 유연한 버전 범위와 Docker 이미지 `latest` 사용으로 재현성이 낮다.
- **P1:** Compose 설정에 기본 secret/password와 DEBUG 설정이 존재한다.

## 8. 다음 점검 권장 순서

1. `/ask`에 concurrency 1/4/8/16 부하를 걸고 p50/p95/p99, timeout, error, CPU, event-loop delay를 수집한다.
2. 동일 corpus·모델·설정으로 baseline/candidate를 각각 5회 이상 실행한다.
3. cold-cache/warm-cache를 분리하고 run별 raw JSON을 보존한다.
4. FastAPI 직접 경로와 Django proxy 경로를 비교해 backend·proxy·DB 시간을 분리한다.
5. Ollama 중단·복구 실험으로 retry, error, p99, recovery time을 측정한다.
6. 성능 threshold를 CI에 연결하고 threshold 실패 시 job이 실제 실패하는지 확인한다.

## 9. 포트폴리오용 현재 표현

> Chatbot_v6는 Hybrid RAG 검색과 파이프라인 단계별 처리시간 계측을 구현하고, 소규모 합성 질의 벤치마크와 단위 테스트 구조를 보유한다. 다만 production-like API 부하, 반복 A/B 실험, 자원 영향, 장애 복구, CI 성능 회귀 게이트는 아직 정량적으로 검증되지 않았다.

## 10. 최종 Hiring Signal

| 관점 | 판정 |
|---|---|
| Backend Engineering | Moderate |
| Performance Engineering | Weak |
| SRE / Reliability | Weak |
| Experimental Thinking | Weak |
| Technical Decision Making | Moderate |

면접관이라면 추가 질문은 **YES**다. 메트릭·검색·캐시 구현은 기술 질문의 소재가 되지만, 성능 수치를 신뢰하려면 workload 선정, 반복성, 자원 영향, 장애 복구에 대한 추가 증거가 필요하다.

## 11. 주요 근거 파일

- `markdown/PERFORMANCE_REPORT.md`
- `markdown/FINAL_OPTIMIZATION_REPORT.md`
- `markdown/OPTIMIZATION_ANALYSIS_v2.md`
- `modules/pipeline/rag_pipeline.py`
- `modules/monitoring/metrics.py`
- `api/app.py`
- `tests/unit/test_bm25_retriever.py`
- `tests/unit/test_question_analyzer.py`
- `pytest.ini`
- `docker-compose.yml`

