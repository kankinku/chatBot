# Version Lineage

과거 snapshot 소스 전체는 Git history와 다음 archive ref에 보존한다.

- `origin/archive/pre-unification-2026-09-18`
- `archive/pre-unification-e05aca3`

active tree에는 historical snapshot을 복제해 두지 않는다. 상세 메커니즘 inventory는 `docs/history/retired-snapshot-inventory.md`를 참고한다.

| Snapshot | 주요 역할 | 현재 상태 |
| --- | --- | --- |
| Chatbot_v1 | 초기 대형 RAG 모놀리스, legal/query/domain 실험 | active tree 제거, archive-only |
| Chatbot_v2 | minimal RAG, OCR/corpus/vector/benchmark 실험 | active tree 제거, 핵심 의도는 canonical ingestion/evaluation에 계승 |
| Chatbot_v3 | v1 계열 + SQL/schema/query routing 실험 | active tree 제거, archive-only |
| Chatbot_v4 | React/Django/FastAPI/Ollama 서비스 분리 | active tree 제거, apps/services/deploy 계보로 흡수 |
| Chatbot_v5.final | product RAG와 persistence/metrics | active tree 제거, v6/canonical core에 계승 |
| Chatbot_v6 | modular RAG core + production web stack | canonical tree로 이동 완료 |
| onTology_system_v9 | 초기 ontology/graph 실험 | active tree 제거, archive-only |
| ontology_system_v11 | extraction/validation/domain/reasoning lifecycle | active tree 제거, v13 계보로 보존 |
| ontology_system_v12 | evidence/incremental/dependency/replay/scenario 최대 실험 | active tree 제거, mechanism source는 archive + inventory로 보존 |
| ontology_system_v13 | 축소된 extraction/validation/reasoning knowledge core | **임시 migration source**. 다음 Knowledge Core PR에서 canonical `src/chatbot/knowledge`로 이동 예정 |
| test_chatbot | 단순 RAG/benchmark 비교용 prototype | active tree 제거, 평가 개념은 canonical scripts에 이미 계승 |

## Canonical mapping

- `Chatbot_v6/modules` → `src/chatbot`
- `Chatbot_v6/api` → `services/inference/api`
- `Chatbot_v6/Server/backend` → `services/gateway`
- `Chatbot_v6/Server/frontend` → `apps/web`
- `Chatbot_v6/config` → `config`
- `Chatbot_v6/data` → `data`
- `Chatbot_v6/tests` → `tests`
- `Chatbot_v6/scripts` → `scripts`
- Docker/Compose → `deploy/docker`, `deploy/compose`

앞으로 버전은 directory snapshot이 아니라 Git commit/branch/tag/release로 표현한다. `ontology_system_v13`은 버전 관리 방식의 예외가 아니라 아직 완료되지 않은 migration source이며, 통합 후 제거한다.
