# chatBot 통합·최적화 및 시스템 개선 계획

작성일: 2026-09-18
기준 저장소: `kankinku/chatBot`
계획 브랜치: `plan/unified-chatbot-architecture`

## 0. 목표

이 계획의 목적은 다음 다섯 가지를 동시에 해결하는 것이다.

1. `Chatbot_v1~v6`, `ontology_v9~v13` 식 snapshot 폴더 체계를 종료한다.
2. 역사적 발전 내용은 잃지 않고 Git history/tag/archive로 보존한다.
3. 현재 RAG core, production web stack, ontology reasoning을 하나의 유지 가능한 제품 구조로 통합한다.
4. Graft, Aider, Sourcegraph, Continue에서 검증된 context/indexing 메커니즘을 개발 환경과 제품 knowledge layer에 적용한다.
5. "성능이 좋아 보이는 코드"가 아니라 benchmark, provenance, freshness, regression gate가 있는 시스템으로 전환한다.

핵심 원칙:

> Source는 하나, 파생물은 재생성 가능, 변경 영향은 추적 가능, 품질은 증거로 승인한다.

---

# 1. 제안하는 최종 저장소 구조

권장 구조는 아래와 같다.

```text
chatBot/
├─ apps/
│  └─ web/                         # React UI
│
├─ services/
│  ├─ gateway/                     # Django public edge
│  │  ├─ auth/
│  │  ├─ conversations/
│  │  ├─ ownership/
│  │  └─ telemetry/
│  │
│  └─ inference/                   # FastAPI internal-only inference API
│     ├─ api/
│     └─ runtime/
│
├─ src/
│  └─ chatbot/
│     ├─ ingestion/
│     │  ├─ document_loader.py
│     │  ├─ structural_parser.py
│     │  ├─ chunker.py
│     │  └─ manifest.py
│     │
│     ├─ retrieval/
│     │  ├─ lexical.py
│     │  ├─ vector.py
│     │  ├─ graph.py
│     │  ├─ fusion.py
│     │  └─ context_budget.py
│     │
│     ├─ generation/
│     │  ├─ prompt_builder.py
│     │  ├─ llm_gateway.py
│     │  └─ answer_generator.py
│     │
│     ├─ quality/
│     │  ├─ evidence_checker.py
│     │  ├─ numeric_unit_checker.py
│     │  ├─ answer_alignment.py
│     │  └─ fallback.py
│     │
│     ├─ knowledge/
│     │  ├─ extraction/
│     │  ├─ validation/
│     │  ├─ graph/
│     │  ├─ reasoning/
│     │  └─ provenance/
│     │
│     ├─ orchestration/
│     │  ├─ query_router.py
│     │  ├─ retrieval_plan.py
│     │  └─ rag_pipeline.py
│     │
│     └─ observability/
│        ├─ traces.py
│        ├─ metrics.py
│        └─ audit.py
│
├─ config/
│  ├─ models/
│  ├─ retrieval/
│  ├─ domain/
│  ├─ ontology/
│  └─ environments/
│
├─ data/
│  ├─ fixtures/                    # 작은 테스트 fixture만 Git 추적
│  ├─ schemas/
│  └─ README.md
│
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ contracts/
│  ├─ security/
│  └─ evaluation/
│
├─ benchmarks/
│  ├─ datasets/
│  ├─ runners/
│  ├─ thresholds/
│  └─ results/                     # 필요 시 artifact 저장, raw 대용량은 외부
│
├─ docs/
│  ├─ architecture/
│  ├─ adr/
│  ├─ history/
│  └─ modernization/
│
├─ deploy/
│  ├─ docker/
│  └─ compose/
│
├─ scripts/
├─ pyproject.toml
├─ package.json
└─ README.md
```

## 왜 이렇게 나누는가

### `services/gateway`

Django는 공개 진입점으로 유지한다.

책임:

- 사용자 인증
- conversation ownership
- operator 권한
- DB persistence
- public-safe error contract
- telemetry

`origin/codex/r1-reproducibility`에서 구현된 보안 경계와 잘 맞는다.

### `services/inference`

FastAPI는 외부에 직접 노출하지 않고 internal inference service로 둔다.

책임:

- RAG pipeline 호출
- corpus/index 관리 API
- inference health
- 모델 runtime

### `src/chatbot`

실제 비즈니스 로직이다.

FastAPI/Django framework와 최대한 분리한다.

이렇게 해야 테스트가 Docker/DB/Ollama 없이도 상당 부분 실행 가능하다.

---

# 2. 버전 폴더 문제 해결 전략

## 2.1 원칙

앞으로 새 버전은 폴더로 만들지 않는다.

금지:

```text
Chatbot_v7/
Chatbot_v8/
ontology_system_v14/
```

버전 관리는 Git으로 한다.

- branch = 진행 중 변화
- commit = 변화 단위
- tag/release = 배포 버전
- `docs/history` = 사람이 읽는 발전 기록

## 2.2 삭제 전에 보존

통합 작업 시작 시 다음을 만든다.

### archive branch

```text
archive/pre-unification-2026-09-18
```

이 branch는 현재 snapshot 폴더 전체를 보존한다.

### annotated tag

예:

```text
archive/pre-unification-e05aca3
```

### version lineage 문서

`docs/history/version-lineage.md`에 다음을 남긴다.

| 기존 경로 | 성격 | 핵심 기능 | 통합 위치 | 상태 |
|---|---|---|---|---|
| Chatbot_v1 | 대형 초기 모놀리스 | legal/query/domain RAG | 일부 mechanism만 | archive |
| Chatbot_v2 | minimal RAG | OCR/benchmark | ingestion/eval | migrated |
| Chatbot_v3 | v1 계열 개선 | legal/query | optional | archive |
| Chatbot_v4 | web integration | React/Django/FastAPI | apps/services | migrated |
| Chatbot_v5.final | product RAG | hybrid/rerank/recovery | src/chatbot | migrated |
| Chatbot_v6 | modular core | current RAG/eval | canonical baseline | active |
| ontology_v12 | maximal experiment | evidence/replay/incremental | selective | mechanism source |
| ontology_v13 | minimal KG core | extraction/validation/reasoning | knowledge core | active |

## 2.3 최종 제거

기능 provenance 검토가 끝난 뒤 active main에서는 old snapshot directory를 제거한다.

Git history/archive branch에서 언제든 복구 가능하므로 source tree에 중복 보관하지 않는다.

---

# 3. 통합 기준선

## 3.1 RAG

`Chatbot_v6`를 baseline으로 한다.

이유:

- 책임별 모듈 분리
- config centralization
- BM25 + Vector hybrid
- reranking
- filtering
- generation
- quality
- metrics
- unit/integration test structure

## 3.2 Production hardening

`origin/codex/r1-reproducibility`을 별도 review 후 통합한다.

중요 기능:

- fail-closed production settings
- secret validation
- proxy auth
- conversation ownership
- operator isolation
- internal service isolation
- deterministic fixtures
- contract/security tests
- public health vs protected status

권장 순서:

1. integration branch 생성
2. branch 전체 diff review
3. 현재 main 기준 tests 재실행
4. security changes를 baseline에 먼저 반영
5. 그 다음 directory restructure

경로를 먼저 이동한 뒤 security commit을 cherry-pick하면 불필요한 conflict가 커진다.

## 3.3 Knowledge graph

`ontology_system_v13`을 baseline으로 한다.

직접 가져올 core:

- extraction
- validation
- domain update
- graph repository
- path reasoning
- optional LLM gateway
- transaction boundary

## 3.4 v12에서 선택적으로 되살릴 것

초기 통합에서는 v12 전체를 사용하지 않는다.

Phase별 선택:

### 먼저 가져올 가치가 높은 것

- Evidence Binder / Evidence Store 개념
- dependency graph
- incremental update
- snapshot/replay contract

### 나중에 필요할 때

- personal KG
- policy learner
- scenario inference
- regime detector
- shock simulator

---

# 4. Graft 메커니즘 적용

Graft의 핵심은 "LLM에게 repo 전체를 다시 읽게 하지 않는 것"이다.

이 프로젝트에는 두 층으로 적용한다.

---

## 4.1 개발 시스템: 실제 Graft 사용

버전 폴더 정리가 끝난 뒤 repository에 Graft를 연결할 수 있다.

권장:

1. canonical tree 완성
2. `graft init --dry-run`
3. deterministic graph build
4. Moon/Codex가 Graft MCP/context graph를 보조적으로 사용

현재 구조에서 바로 Graft를 적용하지 않는 이유:

- v1~v6가 동시에 source로 보임
- 동일 class/function 개념이 여러 snapshot에 존재
- agent가 obsolete implementation을 current로 오인할 수 있음
- blast radius와 graph centrality가 history snapshot 때문에 왜곡됨

Graft는 graph 자체를 local regenerable cache로 취급하고 Git에 넣지 않는 구조를 사용한다. 또한 query 전에 working tree freshness를 cheap fingerprint로 검사하고, 변경이 있으면 deterministic structural layer만 갱신한다. 이 방식은 현재 프로젝트의 index 관리에도 매우 유용하다.

참고:
- https://github.com/trailhq/Graft
- https://github.com/trailhq/Graft/blob/main/README.md

---

## 4.2 제품 시스템: Graft 원리를 Knowledge Index에 적용

Graft 자체를 PDF RAG engine으로 사용하는 것은 권장하지 않는다.

대신 메커니즘을 이식한다.

### Tier 0: Source Manifest

각 source document에 안정적인 identity와 hash를 부여한다.

예:

```json
{
  "document_id": "water-manual-2025",
  "source_uri": "...",
  "sha256": "...",
  "parser_version": "2",
  "updated_at": "...",
  "status": "active"
}
```

### Tier 1: deterministic structural index

LLM 호출 없이 생성한다.

- document
- page
- section
- table
- list
- numeric value
- unit
- heading hierarchy
- deterministic entities
- source offsets

변경 감지와 freshness는 이 층에서 처리한다.

### Tier 2: semantic enrichment

비용이 드는 파생 계층이다.

- embeddings
- LLM summary
- entity resolution
- relation extraction
- claim extraction
- reranker metadata

각 결과는 source/chunk content hash 기준으로 cache한다.

원문이 바뀌지 않으면 다시 생성하지 않는다.

---

# 5. Incremental indexing

현재 방식에서 가장 큰 구조 개선 중 하나다.

## 기존 위험

문서가 조금 변경되어도 전체 corpus/index를 다시 만들 가능성이 높다.

## 개선

```text
Source changed?
  │
  ├─ No → 기존 derived artifacts 재사용
  │
  └─ Yes
      ├─ 변경 document 식별
      ├─ 변경 section/chunk만 재생성
      ├─ 해당 embedding만 갱신
      ├─ 해당 ontology relation만 재검증
      ├─ dependency cache invalidation
      └─ affected evaluation subset 실행
```

Manifest에는 최소 다음을 저장한다.

- source hash
- parser version
- chunker version
- embedding model/version
- ontology extractor version
- config fingerprint
- derived artifact hash

---

# 6. Freshness gate

Graft는 query path에서 graph freshness를 검사한다.

이 원리를 RAG에도 적용한다.

단, 큰 PDF 시스템에서는 query 중 LLM enrichment를 실행하면 안 된다.

## 권장 정책

### Cheap check

query 시작 전:

- source manifest version
- active index generation
- config fingerprint

만 확인한다.

### stale 상태인 경우

환경별 정책:

#### development

- 작은 deterministic 변경은 자동 refresh 가능

#### production strict

- last-known-good index 사용
- response metadata에 stale state 기록
- background incremental rebuild enqueue
- critical schema mismatch는 fail closed

즉:

> freshness check는 synchronous, 비싼 rebuild는 asynchronous

---

# 7. Typed Knowledge Graph

Graft의 typed relation concept를 ontology에 적극 적용한다.

추천 node:

- Document
- Section
- Chunk
- Entity
- Equipment
- Process
- Metric
- Quantity
- Procedure
- Rule
- Claim
- Evidence

추천 edge:

- `contains`
- `mentions`
- `defines`
- `part_of`
- `affects`
- `causes`
- `depends_on`
- `before`
- `after`
- `supports`
- `contradicts`
- `evidence_for`
- `derived_from`
- `supersedes`

모든 semantic edge는 provenance를 가진다.

예:

```text
relation
 ├─ source_document_id
 ├─ source_chunk_id
 ├─ source_hash
 ├─ extraction_method
 ├─ confidence
 ├─ validator_results
 └─ created_at
```

---

# 8. Retrieval 개선

현재 v6의 BM25 + Vector는 유지한다.

여기에 graph channel을 추가한다.

## Query Router

질문을 먼저 retrieval plan으로 바꾼다.

예:

### 단순 사실

```text
BM25 + Vector
```

### 수치/단위

```text
BM25 + numeric structural index + Vector
```

### 원인/영향 관계

```text
Vector + Knowledge Graph path
```

### 절차

```text
section/sequence-aware retrieval + BM25
```

### 비교/복합 질문

```text
Vector + Graph + multi-section context
```

---

# 9. GraphRank + token budget

Aider의 repository map은 전체 graph 중 현재 작업과 관계 높은 symbol을 graph ranking으로 선택하고 token budget에 맞춘다.

이 아이디어를 answer context에도 적용한다.

현재 context selector를 다음 형태로 발전시킨다.

```text
Candidate score =
    lexical score
  + semantic score
  + reranker score
  + graph relevance
  + evidence confidence
  + structural importance
  - redundancy penalty
  - stale penalty
```

그 뒤 token budget 안에서 context set을 구성한다.

목표는 top-k가 아니라 **evidence coverage / token**을 최적화하는 것이다.

참고:
- https://aider.chat/docs/repomap.html

---

# 10. Precise identity / provenance

Sourcegraph SCIP의 중요한 원칙은 symbol을 문자열 search 결과가 아니라 stable identity와 reference relationship으로 다루는 것이다.

이를 문서 지식에도 적용한다.

나쁜 예:

```text
"탁도" 문자열이 같은 chunk
```

좋은 예:

```text
entity_id = water_quality.turbidity
metric_definition_id = ...
source_section_id = ...
relation_id = ...
```

alias:

```text
탁도
turbidity
NTU
```

는 하나의 canonical entity에 연결한다.

참고:
- https://sourcegraph.com/docs/code-navigation
- https://sourcegraph.com/docs/code-navigation/precise-code-navigation

---

# 11. Hybrid semantic retrieval

Continue의 codebase indexing은 embedding retrieval을 keyword retrieval과 함께 사용한다.

현재 v6가 이미 이 방향이므로 유지한다.

개선점:

- vector-only fallback 금지
- lexical evidence 보존
- semantic rerank
- graph relevance 추가
- document/source diversity 보장
- exact numeric/unit hit boost
- provenance completeness boost

참고:
- https://docs.continue.dev/guides/custom-code-rag

---

# 12. Blast Radius를 Knowledge 변경에 적용

Graft의 강력한 기능 중 하나는 변경 symbol에서 dependent symbol을 역추적하는 blast radius다.

같은 원리를 knowledge update에 적용한다.

문서 변경 시:

```text
Changed source section
   ↓
Changed chunks
   ↓
Changed entities / relations
   ↓
Affected graph paths
   ↓
Affected cached answers
   ↓
Affected QA / regression cases
```

이를 이용하면 전체 evaluation을 매번 돌리지 않고 영향받은 test subset을 우선 실행할 수 있다.

---

# 13. Derived artifact 정책

Git에 넣지 않아야 할 것:

- embeddings
- FAISS/HNSW binary index
- generated graph cache
- model cache
- runtime DB
- benchmark temporary output
- LLM enrichment cache

Git에 넣어야 할 것:

- schema
- config
- small test fixtures
- gold QA
- migration
- source code
- thresholds
- ADR
- benchmark runner

원칙:

> 데이터 source와 재생성 로직은 보존하고, 재생성 가능한 index는 source-of-truth로 취급하지 않는다.

---

# 14. Evaluation architecture

## 14.1 Retrieval

필수:

- Recall@K
- MRR
- nDCG
- Context Precision
- source diversity

## 14.2 Answer

필수:

- faithfulness
- citation/evidence coverage
- answer correctness
- numeric accuracy
- unit accuracy
- unsupported-claim rate

## 14.3 Runtime

필수:

- p50
- p95
- p99
- error rate
- timeout rate
- concurrency
- CPU
- memory
- model latency
- retrieval latency
- generation latency

## 14.4 Index

추가:

- stale query count
- incremental update latency
- documents reprocessed
- chunks reused
- embeddings reused
- graph nodes invalidated
- cache hit ratio

---

# 15. CI Gate

최소 pipeline:

```text
static
  ↓
unit
  ↓
contract
  ↓
security
  ↓
offline-rag-regression
  ↓
integration
  ↓
benchmark-smoke
```

release candidate:

```text
+ full QA regression
+ load test
+ index migration test
+ stale/freshness test
+ failure recovery
```

성능 문서에 숫자를 적는 것으로 끝내지 않고 threshold를 코드화한다.

예:

```yaml
retrieval:
  recall_at_10_min: 0.90

answer:
  numeric_accuracy_min: 0.95
  unsupported_claim_rate_max: 0.02

runtime:
  p95_ms_max: 3000
  error_rate_max: 0.01
```

---

# 16. 보안 아키텍처

`codex/r1-reproducibility` 방향을 유지한다.

외부:

```text
Browser
  ↓
Django Gateway
  ├─ auth
  ├─ ownership
  ├─ rate limit
  ├─ telemetry
  └─ public-safe contracts
       ↓ internal network
FastAPI inference
       ↓
Ollama / indexes / graph store
```

금지:

- FastAPI inference를 public expose
- default production secret
- wildcard host/CORS
- anonymous operator endpoint
- session id만으로 conversation access

---

# 17. 단계별 실행 계획

## Phase 0 — 안전장치 / baseline

목표: 무엇도 잃지 않은 상태에서 작업을 시작한다.

작업:

1. `archive/pre-unification-2026-09-18` branch 생성
2. archive tag 생성
3. 현재 main 및 remote branch inventory
4. `codex/r1-reproducibility` 독립 review
5. main/v6 baseline test
6. feature provenance matrix 완성

완료 조건:

- 모든 old snapshot 복구 경로 존재
- current baseline SHA 기록
- security branch 판단 기록

---

## Phase 1 — security/reproducibility baseline 통합

작업:

1. integration branch 생성
2. R1/R2 보안 commit 통합
3. unit/contract/security test
4. Django/FastAPI boundary 검증
5. Docker config fail-closed 검증

완료 조건:

- 보안 패치가 이후 구조 이동의 기준 코드가 됨

---

## Phase 2 — canonical tree 생성

작업:

1. v6 modules를 `src/chatbot`으로 이동
2. FastAPI를 `services/inference`로 이동
3. Django를 `services/gateway`로 이동
4. React를 `apps/web`으로 이동
5. root config/deploy/test 정리
6. import path 정규화
7. root package/pyproject 정의

중요:

이 단계는 기능 변경 없이 **move/refactor only**로 제한한다.

완료 조건:

- 이전과 동일한 기능
- tests green
- 한 개 canonical runtime

---

## Phase 3 — old version 제거

작업:

1. v1~v5 unique feature inventory 검토
2. 빠진 필수 mechanism만 issue로 남김
3. old Chatbot folders 삭제
4. ontology v9/v11/v12 snapshot 삭제
5. test_chatbot는 fixture/benchmark로 필요한 것만 migration
6. `docs/history/version-lineage.md` 작성

완료 조건:

- root에서 버전 폴더 없음
- history는 Git/archive/docs로 접근 가능

---

## Phase 4 — Knowledge Core 통합

작업:

1. ontology v13 extraction을 `src/chatbot/knowledge/extraction`로 이동
2. validation 이동
3. graph repository 이동
4. reasoning 이동
5. 정수처리 entity/relation schema 작성
6. chunk ↔ entity ↔ relation provenance 연결

처음에는 graph retrieval을 feature flag로 둔다.

완료 조건:

- 기존 vector-only path와 graph-assisted path를 A/B 가능

---

## Phase 5 — Graft-style incremental knowledge index

작업:

1. source manifest
2. content hash
3. parser fingerprint
4. chunk fingerprint
5. derived artifact registry
6. changed document detector
7. incremental embedding
8. incremental graph update
9. stale state
10. background rebuild worker

완료 조건:

- 1개 문서 수정 시 전체 corpus 재처리하지 않음
- 변경하지 않은 embeddings 재사용

---

## Phase 6 — Context Graph / ranking

작업:

1. typed graph relations
2. graph retrieval channel
3. graph centrality/relevance
4. lexical + vector + graph fusion
5. token-budgeted context packing
6. redundancy/diversity constraint
7. evidence coverage optimization

완료 조건:

- graph 질문에서 RAG baseline 대비 개선을 regression dataset으로 입증

---

## Phase 7 — Blast radius / selective validation

작업:

1. source → chunk dependency
2. chunk → graph edge dependency
3. edge → evaluation case dependency
4. changed-source impact calculation
5. affected QA subset
6. cache invalidation

완료 조건:

- 데이터 변경의 영향 범위를 설명 가능
- 관련 regression만 빠르게 실행 가능

---

## Phase 8 — Production evidence

작업:

1. load generator
2. concurrency 1/4/8/16
3. cold/warm cache 분리
4. repeated baseline/candidate runs
5. p50/p95/p99
6. CPU/memory
7. Ollama failure injection
8. recovery/MTTR
9. CI performance thresholds

완료 조건:

- README 성능 수치가 raw evidence로 재현 가능

---

# 18. 구현 우선순위

## 반드시 먼저

1. archive
2. security branch 검토/통합
3. canonical tree
4. old version removal
5. reproducible test baseline

## 그 다음

6. ontology v13 core
7. incremental manifest
8. provenance
9. graph-assisted retrieval

## 마지막

10. replay
11. learning
12. scenario
13. automatic policy optimization

이 순서를 뒤집으면 다시 v12처럼 시스템이 커진 뒤 핵심이 불명확해질 가능성이 높다.

---

# 19. Graft를 그대로 복제하지 않는 이유

Graft는 source code graph에 최적화돼 있다.

chatBot domain knowledge는 다음이 다르다.

- PDF가 크다.
- OCR 오류가 있다.
- 표/숫자/단위가 중요하다.
- source가 Git file만이 아니다.
- semantic enrichment 비용이 크다.
- production query가 index rebuild를 오래 기다릴 수 없다.

따라서 메커니즘을 다음처럼 변형한다.

| Graft | chatBot 적용 |
|---|---|
| tree-sitter graph | deterministic document structure graph |
| symbol | section/chunk/entity/claim |
| code call edge | semantic/domain relation |
| fingerprint | document/chunk/config hash |
| incremental graph build | incremental corpus/index build |
| repo map | knowledge map |
| GraphRank | retrieval/context rank |
| blast radius | knowledge update impact |
| local graph cache | derived index/cache |
| deep LLM enrichment | optional semantic enrichment |
| pre-query freshness | manifest freshness check |

---

# 20. 기대되는 최종 시스템

최종적으로 사용자는 질문만 한다.

시스템 내부:

```text
Question
   ↓
Query Analysis
   ↓
Retrieval Plan
   ├─ lexical
   ├─ vector
   ├─ structural
   └─ graph
        ↓
Evidence Fusion / Rerank
        ↓
Token-budget Context Pack
        ↓
LLM Generation
        ↓
Evidence / Numeric / Unit Validation
        ↓
Fallback or Final Answer
        ↓
Trace + Metrics
```

지식 변경:

```text
New/Changed Document
   ↓
Fingerprint
   ↓
Only changed sections
   ↓
Chunks / Embeddings
   ↓
Entity / Relation Validation
   ↓
Graph Update
   ↓
Blast Radius
   ↓
Selective Cache + Evaluation Invalidation
```

---

# 21. 다음 실제 작업

이 계획 승인 후 첫 구현 PR은 기능 추가가 아니다.

**PR 1: Repository Canonicalization Foundation**

범위:

1. archive branch/tag 생성
2. security branch review 결과 반영
3. target directory skeleton 생성
4. current v6 code를 기능 변경 없이 이동
5. imports/tests/docker path 수정
6. old version lineage 문서 생성
7. CI baseline

그 PR이 안정화된 뒤 old version directory 제거 PR을 별도로 진행한다.

이렇게 분리해야 대규모 이동과 기능 변경을 한 PR에서 섞지 않는다.
