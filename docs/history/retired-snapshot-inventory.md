# Retired Snapshot Inventory

이 문서는 active tree에서 제거한 historical snapshot의 역할과 다시 가져올 가치가 있는 메커니즘을 기록한다. 원본 소스는 다음 두 ref에 그대로 보존되어 있다.

- branch: `origin/archive/pre-unification-2026-09-18`
- annotated tag: `archive/pre-unification-e05aca3`

아래 snapshot은 두 archive ref 모두에서 경로 단위 복구 가능함을 삭제 전에 확인했다.

| Snapshot | 코드에서 확인한 핵심 역할 | Canonical 상태 |
| --- | --- | --- |
| Chatbot_v1 | PDF RAG, legal retrieval, dynamic query expansion, hallucination prevention, cache/memory optimization, quality loop, domain-specific routing/reranking, unified logging 실험 | wholesale 이식하지 않음. query expansion·quality/recovery 아이디어만 필요 시 archive에서 재구현 |
| Chatbot_v2 | 작은 실행 가능 RAG, OCR fallback, PDF→corpus→vector index, CPU fallback, QA benchmark | ingestion/evaluation 의도는 현재 canonical scripts/core에 계승 |
| Chatbot_v3 | v1 계열 + dynamic SQL/schema, structured extraction, query routing, legal search | 사용 요구가 생길 때만 선택 복원. 현재 정수처리 RAG core에는 포함하지 않음 |
| Chatbot_v4 | React + Django proxy + FastAPI/Ollama + Docker/GPU 서비스화 | 현재 apps/services/deploy 구조의 계보로 흡수 완료 |
| Chatbot_v5.final | hybrid retrieval, reranking, adaptive filtering, dynamic K, guardrail, retry/recovery, conversation persistence/metrics | v6를 거쳐 현재 src/chatbot + services/gateway에 계승 |
| onTology_system_v9 | 초기 graph/ontology, entity/relation resolution, scenario/market-domain 실험 | domain-specific historical source만 보존 |
| ontology_system_v11 | Extraction → Validation → Domain/Personal Update → Reasoning lifecycle, graph repository/path reasoning | 핵심 구조는 v13이 더 작은 형태로 계승하므로 archive-only |
| ontology_system_v12 | evidence binding, delta/idempotent ingestion, feature dependency, incremental orchestration, replay/backtest, scenario/regime, snapshot/learning | 중요한 mechanism source. 아래 backlog로 명시하고 소스는 archive에서 복구 |
| test_chatbot | 단순 Chroma/Ollama RAG baseline과 benchmark/evaluation prototype | 평가 메커니즘은 현재 scripts에 이미 상위 구현 존재. 별도 runtime은 제거 |

## v12에서 보존해야 할 구현 아이디어

v12 전체를 다시 import하지 않는다. 다음 메커니즘을 canonical architecture에 맞게 독립적으로 재구현한다.

1. **Evidence binding**
   - archive source: `ontology_system_v12/src/evidence/`
   - 관계/추론 결과가 어떤 source evidence에서 나왔는지 연결하는 provenance layer의 참고 구현.

2. **Incremental update + idempotency**
   - archive source: `src/ingestion/delta_fetcher.py`, `src/ingestion/idempotency_guard.py`, `src/orchestration/incremental_orchestrator.py`
   - 전체 재색인 대신 변경된 source/entity/relation만 갱신하는 구조의 참고 구현.

3. **Dependency invalidation / blast radius**
   - archive source: `src/features/feature_dependency_index.py`, `src/orchestration/dependency_graph_manager.py`
   - 변경된 근거가 어떤 파생 feature/relation/evaluation을 무효화하는지 추적하는 구조.

4. **Replay / evaluation**
   - archive source: `src/replay/`
   - snapshot 기준 재현, 결과 비교, regression/evaluation loop의 참고 구현.

5. **Scenario / regime**
   - archive source: `src/scenario/`, `src/scenario_inference/`, `src/regime/`
   - 현재 제품 기본 경로에는 넣지 않고 실제 요구가 생길 때 선택적으로 재구현.

## test_chatbot 처리

`test_chatbot`의 평가 아이디어는 이미 canonical scripts에 존재한다.

- Token F1 / ROUGE-L / BLEU: `scripts/academic_metrics.py`
- 숫자·단위 정확도: `scripts/enhanced_scoring.py`
- Faithfulness / Answer Correctness / Context Precision: `scripts/rag_core_metrics.py`
- 통합 평가: `scripts/unified_evaluation.py`, `scripts/evaluate_qa_unified.py`

따라서 prototype 코드를 별도로 유지할 이유가 없다. 과거 `qa.json`은 특정 환경의 endpoint/account 예시가 섞인 historical benchmark이므로 현재 fixture로 자동 승격하지 않고 archive에만 보존한다.

## 복구 방법

과거 구현을 확인할 때 active tree에 snapshot 폴더를 다시 복사하지 않는다.

```bash
git show archive/pre-unification-e05aca3:ontology_system_v12/src/evidence/evidence_binder.py
git show origin/archive/pre-unification-2026-09-18:Chatbot_v1/README.md
```

필요한 메커니즘은 새 branch에서 canonical 위치에 재구현하고 테스트/ADR/provenance를 함께 추가한다.
