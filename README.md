# Water Treatment Chatbot

정수처리 문서를 대상으로 하는 RAG 챗봇의 canonical repository입니다.

기존 `Chatbot_v1~v6`와 ontology snapshot 폴더에서 발전하던 구조를 Git history/archive 기반의 단일 제품 트리로 전환하고 있습니다. 현재 실행 기준선은 기존 Chatbot v6의 RAG core와 production web stack입니다.

## Canonical structure

```text
apps/web/                 React UI
services/gateway/         Django public gateway
services/inference/       FastAPI internal inference API
src/chatbot/              framework-independent RAG core
src/chatbot/knowledge/    graph knowledge extraction/validation/reasoning core
src/chatbot/knowledge/evidence/ extraction-to-relation evidence provenance chain
src/chatbot/knowledge/workspace/ derived relationship/provenance graph tooling
config/                   runtime/model/pipeline configuration
config/ontology/          knowledge schema and backend configuration
data/                     small tracked fixtures and domain data
data/ontology/            knowledge domain seeds and samples
tests/                    unit/integration/contracts
scripts/                  corpus/evaluation/interactive tools
deploy/docker/            container images
deploy/compose/           multi-service runtime
docs/                     architecture, evidence, history, modernization
```

활성 코드는 더 이상 `Chatbot_vN/` 또는 `ontology_system_vN/` snapshot 폴더를 사용하지 않습니다. retired source는 Git archive branch/tag와 `docs/history/`에서 추적하며, v13 Knowledge Core도 `src/chatbot/knowledge/`로 통합했습니다. 관계/provenance 탐색용 `knowledge-workspace/`는 canonical source에서 재생성되는 로컬 cache이며 Git source-of-truth가 아닙니다.

## Validation

경량 오프라인 검증:

```bash
python3 -m compileall -q services/gateway services/inference src/chatbot tests/unit config scripts
python3 -m pytest tests/unit -q -o addopts= -p no:cacheprovider
```

전체 RAG 런타임 의존성은 `requirements.txt`, CI 검증 의존성은 `requirements-ci.lock`을 사용합니다.

## CLI

```bash
python3 scripts/build_corpus.py --help
python3 scripts/evaluate_qa_unified.py --help
python3 scripts/test_chatbot_interactive.py --help
python3 scripts/knowledge_core_demo.py --help
python3 scripts/knowledge_workspace.py --help
```

## Local services

필수 secret을 환경변수로 제공한 뒤 실행합니다.

```bash
export MYSQL_ROOT_PASSWORD='<strong-root-password>'
export MYSQL_PASSWORD='<strong-app-password>'
export SECRET_KEY='<strong-django-secret>'
docker compose -f deploy/compose/docker-compose.yml up --build
```

기본 노출 경계:

- React: `localhost:3000`
- Django gateway: `127.0.0.1:8001`
- FastAPI, MySQL, Ollama: internal Docker networks only

환경변수 예시는 `services/gateway/env.example`을 참고하세요.

## Development policy

- 새 버전을 새 폴더로 복제하지 않습니다.
- 기능 개발은 branch/PR로 수행합니다.
- 배포 버전은 tag/release로 관리합니다.
- `main`은 GitHub CI와 독립 Merge Auditor 승인을 통과해야 합니다.
- 역사적 snapshot은 `docs/history/version-lineage.md`와 archive refs에서 추적합니다.

자세한 통합 방향은 `docs/modernization/UNIFICATION_AND_OPTIMIZATION_PLAN.md`을 참고하세요.
