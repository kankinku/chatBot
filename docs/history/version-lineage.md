# Version Lineage

이 문서는 과거 snapshot 폴더가 어떤 역할을 했고 canonical repository에 무엇이 계승되는지 기록한다. 소스 전체 보존은 Git history와 `archive/pre-unification-2026-09-18`, `archive/pre-unification-e05aca3`에서 담당한다.

| Snapshot | 주요 역할 | Canonical 처리 |
| --- | --- | --- |
| Chatbot_v1 | 초기 대형 RAG 모놀리스, legal/query/domain 실험 | 역사 보존. 필요한 mechanism만 선택적으로 재구현 |
| Chatbot_v2 | 별도 src/server 구조 실험 | 역사 보존 |
| Chatbot_v3 | v1 계열 개선 및 질의/검색 실험 | 역사 보존 |
| Chatbot_v4 | Django/React/Ollama 서비스 분리 실험 | 후대 production 구조의 참고 계보 |
| Chatbot_v5.final | web/API 통합 직전 snapshot | 역사 보존 |
| Chatbot_v6 | modular RAG core, FastAPI, Django gateway, React UI, 평가/보안 기준선 | 현재 canonical tree로 이동 |
| onTology_system_v9 | 초기 ontology pipeline snapshot | 역사 보존 |
| ontology_system_v11 | ontology 구조 개선 snapshot | 역사 보존 |
| ontology_system_v12 | evidence/replay/incremental 등 최대 실험 버전 | mechanism source. 필요한 기능만 선택적 이식 |
| ontology_system_v13 | 최소화된 extraction/validation/reasoning core | 다음 Knowledge Core 통합의 기준선 |

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

앞으로 버전은 directory snapshot이 아니라 Git commit/branch/tag/release로 표현한다.
