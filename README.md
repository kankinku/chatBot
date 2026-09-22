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
src/chatbot/knowledge/evidence/ extraction-to-relation provenance + evidence scoring
src/chatbot/knowledge/ingestion/ raw-file inventory + source-hash selective ingestion
src/chatbot/knowledge/projection/ deterministic scenario/regime derived views
src/chatbot/retrieval/         hybrid retrieval + selective derived vector refresh
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
python3 scripts/knowledge_ingest.py --help
python3 scripts/knowledge_replay.py --help
python3 scripts/knowledge_project.py --help
```

Knowledge ingestion은 기존 JSONL 입력과 raw file directory 입력을 모두 지원합니다.

```bash
python3 scripts/knowledge_ingest.py --input documents.jsonl
python3 scripts/knowledge_ingest.py --source-dir ./data/pdfs
```

raw file 모드는 SHA-256 byte inventory를 먼저 확인하므로 변경되지 않은 PDF/TXT/MD의 text extraction 자체를 건너뜁니다. Retrieval vector index도 deterministic chunk ID와 derived manifest를 사용해 새로 생기거나 내용이 바뀐 chunk만 embedding하고, 순서/metadata 변경은 재embedding 없이 갱신합니다. 성공적으로 커밋된 Knowledge Core 상태는 immutable replay chain에도 기록되므로 `knowledge_replay.py as-of`로 특정 commit time 기준의 evidence/provenance 상태를 다시 열 수 있습니다. Phase 12 scenario/regime projection은 이 canonical 또는 replay state 위에서 가정을 read-only derived view로 계산하며 evidence/replay/live graph/Chroma를 수정하지 않습니다. 재현 가능한 snapshot-backed projection만 `knowledge_project.py project ... --persist`로 `knowledge-workspace/projections/`에 저장할 수 있습니다.

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
