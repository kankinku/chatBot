# chatBot 발전 과정 및 설계 의도 분석

작성일: 2026-09-18
분석 기준: `kankinku/chatBot` main `e05aca37b091bc79afa949e6d85b597d9eb89ad0` 및 원격 `codex/r1-reproducibility` 브랜치
목적: 버전 폴더를 통합하기 전에 각 시도의 의미와 살아남아야 할 설계 의도를 복원한다.

> 상태 메모: 이 문서는 통합 전 `e05aca3` 시점의 역사 분석이다. 문서에서 “현재 저장소/active tree”라고 부르는 snapshot 중 retired source는 canonicalization 이후 active tree에서 제거되었으며, 원본은 archive ref와 `docs/history/retired-snapshot-inventory.md`에서 추적한다.

## 1. 결론

이 저장소는 단순히 "챗봇 v1 → v6"로 기능이 누적된 프로젝트가 아니다.

실제 변화는 다음 네 축이 반복적으로 강화된 과정이다.

1. **검색 정확도**: PDF 단순 검색에서 도메인 특화 청킹, BM25 + Vector, reranking, 적응형 context selection으로 발전했다.
2. **근거성과 신뢰성**: 키워드 테스트, 수치·단위 보존, hallucination 방지, answer/context alignment, validation/fallback으로 발전했다.
3. **제품화·운영성**: CLI에서 FastAPI, Django proxy, React, DB, Docker, metrics, auth/ownership 경계로 발전했다.
4. **구조적 추론**: 단순 RAG에서 entity/relation extraction, validation, knowledge graph, path reasoning, evidence/replay 실험으로 확장했다.

가장 중요한 패턴은 **기능을 계속 누적하는 것이 아니라, 크게 실험한 뒤 핵심만 남기는 방향**이다.

특히 `ontology_system_v12`에서 약 27K LOC 수준까지 확장한 뒤 `ontology_system_v13`에서 대략 25K LOC 이상을 제거하고 Extraction → Validation → Domain Update → Reasoning의 코어만 남긴 변화가 이를 잘 보여준다.

따라서 통합 작업의 목표는 "모든 과거 코드를 한곳에 합치기"가 아니라 다음이어야 한다.

> 과거 버전에서 검증된 메커니즘을 추출하고, 현재 필요한 최소 코어에 다시 배치한다.

---

## 2. 저장소 진화 타임라인

### 2.1 2025-08: PDF 구조화에서 시작

초기 `PDF_module`은 정수처리 문서를 단순 임베딩 대상으로만 취급하지 않고, 문서 구조를 추출하여 JSON으로 나누고 분류하는 방향에서 출발했다.

주요 의도:

- 정수처리 문서의 구조를 보존한다.
- 섭취, 약품, 응집, 침전, 여과, 소독 등 도메인 단위를 명시적으로 분류한다.
- 범용 QA보다 "정수처리 도메인에서 틀리지 않는 답"을 우선한다.

이 시기의 핵심 자산은 **도메인 구조화 사고방식**이다.

### 2.2 2025-08 말: 신뢰성 최적화

`1차 최적화 성공 (신뢰성 부분)` 단계에서 다음이 강화됐다.

- keyword enhancement
- 질문 분석
- vector store 개선
- answer generator 개선
- 실제 PDF keyword test
- QA check script

즉 성능 최적화보다 먼저 **답의 신뢰성을 테스트 가능하게 만드는 것**에 초점이 있었다.

살려야 할 의도:

- 데이터/질문 유형마다 검색 방식을 달리한다.
- 검색이 잘 됐는지 테스트한다.
- 생성 모델의 답을 그대로 신뢰하지 않는다.

### 2.3 웹서비스화

`웹서비스용 모듈 (mk3)`에서 FastAPI endpoint, Django client, TypeScript client, conversation logger가 추가된다.

이 단계에서 프로젝트는 "로컬 RAG 스크립트"에서 "사용자가 실제로 호출할 수 있는 서비스"로 전환된다.

살려야 할 의도:

- inference core와 서비스 계층 분리
- API contract 명시
- 대화 기록 및 운영 로그
- 다른 프론트/백엔드가 RAG 엔진을 호출할 수 있는 구조

### 2.4 Chatbot_v1~v3: 기능 폭 확대

현재 저장소의 `Chatbot_v1`, `Chatbot_v3`에는 다음이 함께 존재한다.

- PDF RAG
- SQL query
- legal search
- query routing
- dynamic query expansion
- hallucination prevention
- cache/memory optimization
- quality loop
- domain-specific chunker/reranker

이 시기는 **가능한 기능을 넓게 탐색한 시기**로 보는 것이 적절하다.

그러나 이후 v6에 legal/SQL 전체가 그대로 남지 않았다는 점에서, 모든 기능이 장기 코어로 선택된 것은 아니다.

### 2.5 Chatbot_v2: 작고 실행 가능한 RAG

v2는 상대적으로 작고 다음에 집중한다.

- OCR
- PDF corpus 생성
- Vector index
- 한국어 QA
- benchmark
- CPU fallback

이 버전은 거대한 기능 집합보다 **최소 실행 가능 RAG와 평가 루프**를 실험한 흔적이다.

### 2.6 Chatbot_v4~v5: 제품화

v4/v5에서 구조가 다음처럼 변한다.

- React frontend
- Django backend/proxy
- FastAPI/Ollama chatbot server
- MySQL
- Docker Compose
- GPU 실행
- 세션/대화 로그

v5 문서에는 RAG 파이프라인이 훨씬 구체적으로 정리된다.

- question classification
- hybrid retrieval
- BM25 + Vector merge
- Cross-Encoder reranking
- adaptive filtering
- dynamic K
- guardrail
- question-specific prompt
- generation retry
- answer validation
- number preservation
- context alignment
- fallback/recovery
- metrics

여기서 현재 RAG 엔진의 설계 철학이 거의 완성된다.

### 2.7 Chatbot_v6: 구조 재작성

v6의 핵심 변화는 기능 추가보다 **책임 분리**다.

현재 주요 모듈:

- `analysis/`
- `preprocessing/`
- `chunking/`
- `embedding/`
- `retrieval/`
- `reranking/`
- `filtering/`
- `generation/`
- `quality/`
- `monitoring/`
- `pipeline/`

README에 명시된 원칙도 다음과 같다.

- One Source of Truth
- 단일 책임
- 구조화된 예외 처리
- JSON logging

즉 v6는 현재 RAG 계열에서 가장 적절한 **통합 기준선**이다.

### 2.8 v6 성능 최적화에서 얻은 교훈

최적화 문서에는 다음의 의미 있는 변화가 있다.

- 사전 vector normalization
- question analysis cache
- embedding query cache
- batch tuning
- logging overhead 감소
- 과도한 ThreadPool 제거

중요한 점은 병렬화가 항상 빨라지는 것이 아니었다는 사실을 문서에 남겼다는 것이다.

이것은 향후 최적화 원칙으로 유지해야 한다.

> 최적화는 구현 아이디어가 아니라 반복 가능한 benchmark evidence로 승인한다.

### 2.9 2026-08 점검: 성능 주장보다 증거

2026-08-31 inspection report는 과거의 "성능 향상" 문서를 그대로 신뢰하지 않고 다음 문제를 지적했다.

- 테스트 39 pass / 2 fail
- p50/p95/p99 없음
- concurrency/peak/soak 없음
- resource metrics 없음
- cache 조건 통제 부족
- 반복 A/B 결과 부족
- 장애 복구 검증 없음
- CI performance gate 없음
- production secret/debug 문제

이 시점부터 프로젝트의 발전 방향이 **기능 개발 → 검증 가능한 엔지니어링**으로 이동한 것으로 판단된다.

### 2.10 2026-09 미병합 브랜치: 재현성·보안 강화

`origin/codex/r1-reproducibility`는 main 이후의 중요한 발전이다.

확인된 변화:

- unit/contract test baseline
- CLI import/entrypoint 재현성
- fail-closed production configuration
- predictable secret 거부
- proxy authentication
- owner isolation
- operator-only route
- internal service isolation
- public health와 상세 status 분리
- Django settings/security contract test
- deterministic fixture

이 브랜치는 "실험 브랜치"로 버리면 안 된다.

통합 시 **Chatbot_v6의 production hardening 후보**로 별도 검토 후 흡수해야 한다.

---

## 3. Ontology 계열의 의미

### 3.1 v9

v9는 시장 데이터, pair trading, market indices 등을 포함한 금융 성격의 knowledge graph 실험이다.

현재 정수처리 챗봇에 코드를 직접 병합할 대상은 아니다.

가져올 것은 다음 메커니즘이다.

- entity/relation graph
- graph service
- relation reasoning
- 시각화/graph API 사고방식

### 3.2 v11

v11에서는 아키텍처가 6 Layer로 정리된다.

1. Extraction
2. Validation
3. Domain
4. Personal
5. Reasoning
6. Learning/Evolution

여기서 중요한 발전은 단순 graph 저장이 아니라 **입력 → 검증 → 반영 → 추론 → 학습** lifecycle을 정의했다는 것이다.

### 3.3 v12

v12는 이 아이디어를 극단적으로 확장한 버전이다.

추가된 주요 메커니즘:

- append/delta ingestion
- idempotency
- evidence layer
- feature dependency
- incremental orchestrator
- regime
- replay/backtest
- scenario/shock
- policy learning
- snapshot
- learning/deployment
- domain/personal 분리

정수처리 챗봇에 전부 넣으면 과설계가 된다.

그러나 다음 네 메커니즘은 장기적으로 가치가 높다.

1. Evidence binding
2. Incremental update
3. Dependency invalidation
4. Replay/evaluation

### 3.4 v13

v13은 v12를 대폭 축소했다.

남은 핵심:

- Extraction
- Validation
- Domain Update
- Reasoning
- Graph Repository
- Optional LLM Adapter

설정도 domain-agnostic으로 바뀌었다.

이것은 `ontology_system_v13`이 현재 통합에 가장 적절한 **knowledge core 후보**라는 강한 신호다.

---

## 4. 사용자의 장기 의도 추론

코드와 변경 이력에서 반복적으로 나타나는 의도는 다음과 같다.

### 의도 A. 범용 챗봇보다 도메인 전문가

도메인 dictionary, 숫자/단위 보존, 정수처리 전용 chunker/reranker, 질문 유형 분류가 반복된다.

목표는 "무엇이든 대답하는 챗봇"이 아니라 **정수처리 문서에서 정확하고 근거 있게 답하는 assistant**다.

### 의도 B. 검색과 생성의 분리

BM25, vector, reranking, context selection, generator가 명시적으로 분리된다.

따라서 향후에도 LLM 하나에 retrieval과 reasoning을 몰아넣으면 안 된다.

### 의도 C. 생성 결과를 검증

hallucination prevention, answer quality checker, numerical validation, context alignment, confidence, recovery가 여러 버전에 반복된다.

즉 생성 모델은 최종 권위가 아니다.

### 의도 D. 구조화된 지식으로 확장

Ontology 계열은 문서 chunk 검색만으로 해결하지 못하는 "관계"와 "경로"를 다루기 위한 시도다.

최종 방향은 다음 조합으로 보는 것이 적절하다.

> RAG = 문서 근거 검색
> Knowledge Graph = 구조적 관계 추론
> LLM = 언어 해석·합성

### 의도 E. 자동 개선은 검증 이후

v12의 Learning/Evolution, replay, policy learning과 2026-08/09의 검증 강화는 공통된 방향을 갖는다.

자동 변경은 가능하되:

- evidence
- replay
- regression
- deployment gate

를 거쳐야 한다.

### 의도 F. 로컬/자체 운영 가능성

Ollama, local SBERT, FAISS/HNSW, Docker 기반이 지속적으로 사용된다.

외부 API 의존성을 최소화하면서도 필요 시 provider를 교체할 수 있는 구조가 적합하다.

---

## 5. 현재 버전 폴더 구조의 문제

현재 active tree에는 다음 snapshot이 함께 있다.

- Chatbot_v1
- Chatbot_v2
- Chatbot_v3
- Chatbot_v4
- Chatbot_v5.final
- Chatbot_v6
- ontology v9/v11/v12/v13
- test_chatbot

실측 결과:

- exact duplicate hash group: **171개**
- 해당 duplicate group에 포함된 파일: **401개**

문제는 디스크 사용량보다 source-of-truth가 여러 개라는 점이다.

### 발생하는 실제 문제

1. 어떤 버그를 어디에서 고쳐야 하는지 불명확
2. 보안 패치가 한 버전에만 적용됨
3. 같은 기능의 서로 다른 구현이 검색/AI context를 오염
4. dependency/version이 여러 폴더에 중복
5. 테스트가 어떤 production code를 검증하는지 불명확
6. 오래된 문서의 성능 주장이 최신 상태처럼 노출됨
7. CI를 전체 repo에 걸기 어려움
8. Graft 같은 code graph를 만들 때 역사 snapshot이 현재 코드와 함께 graph에 들어감
9. refactor blast radius가 왜곡됨
10. 신규 개발자가 "current"를 찾는 데 비용을 지불함

---

## 6. 통합 시 보존/폐기 기준

### Production baseline

- Chatbot_v6
- + 검증 후 `origin/codex/r1-reproducibility`의 production hardening

### Knowledge core baseline

- ontology_system_v13

### Mechanism source only

- ontology_system_v12
- Chatbot_v1~v5

### Historical / benchmark reference

- test_chatbot
- old optimization reports
- old full snapshot directories

---

## 7. 통합에서 하지 말아야 할 것

1. v1~v6를 하나의 Python path에 그대로 병합
2. v12 전체를 v6에 import
3. legal/SQL/finance 기능을 사용 여부 검증 없이 복원
4. 오래된 benchmark 수치를 현재 성능 baseline으로 사용
5. vector DB/derived cache를 Git source로 취급
6. 온톨로지와 vector retrieval을 하나의 거대한 pipeline class에 결합
7. query 시 비싼 LLM enrichment를 자동 실행
8. 모든 데이터 변경을 전체 re-index로 처리

---

## 8. 최종 판단

현재 프로젝트의 가장 자연스러운 다음 단계는 새 v14나 v7 폴더를 만드는 것이 아니다.

**버전 폴더 체계를 종료하고 하나의 canonical product tree로 전환해야 한다.**

권장 기준:

- RAG core: Chatbot_v6
- production/security: codex/r1-reproducibility 검토 후 흡수
- graph reasoning: ontology_system_v13
- advanced mechanisms: v12에서 선택적으로 재구현
- history: Git branch/tag + docs/history
- derived indexes: regenerable cache
- 평가: reproducible evidence + CI gate

이 기준으로 다음 문서 `UNIFICATION_AND_OPTIMIZATION_PLAN.md`에서 실제 단계별 마이그레이션 구조를 정의한다.
