# Evidence Provenance Chain

Phase 6는 Knowledge Core의 extraction/validation/domain 결과를 관계 그래프의 근거 계층으로 투영한다.

## 목표

관계 하나만 저장하는 대신 다음 계보를 유지한다.

    Document
      -> Fragment
      -> Assertion
      -> Domain Relation

실제 graph edge 방향은 recomputation impact를 계산하기 쉽게 구성한다.

- Document --produces--> Fragment
- Fragment --produces--> Assertion
- DomainRelation --supported_by--> Assertion
- DomainRelation --contradicted_by--> Assertion
- Assertion --depends_on--> Entity
- DomainRelation --depends_on--> Entity
- Entity --domain_relation/semantic_type--> Entity

## Source provenance

ExtractionPipeline은 모든 문서에 다음 값을 남긴다.

- source_uri: logical source identity
- source_hash: raw input UTF-8 bytes의 SHA-256

source_uri가 지정되지 않으면 document:<doc_id>를 사용한다.

Fragment/Assertion ID는 runtime UUID를 직접 사용하지 않는다. 원문 hash, fragment 위치/내용, canonical entity identity, relation semantics를 기반으로 결정적으로 계산한다. 따라서 동일한 원문이 재처리되어 runtime fragment/raw-edge UUID가 달라져도 같은 provenance projection을 만들 수 있다.

## Assertion과 Domain Relation의 분리

Assertion은 한 source가 주장한 개별 관계 레코드다. validation 결과와 domain 판정은 assertion 속성에 보존한다.

Domain Relation은 여러 assertion이 공유할 수 있는 semantic relation identity다. 동일 관계를 여러 문서가 지지하면 하나의 Domain Relation node에 여러 supported_by edge와 SourceRef가 합쳐진다.

상충 evidence는 semantic relation을 즉시 승격하지 않고 contradicted_by로 보존한다.

## Invalidation

EvidenceLedger는 source_uri당 현재 projection 하나를 가진다.

source hash 또는 projection이 바뀌면 이전 document node에서 blast-radius를 계산한다.

- 이전 Document / Fragment / Assertion은 invalidation 대상
- 관련 Domain Relation은 affected/recompute 대상
- Entity 및 다른 source의 Document/Fragment/Assertion은 건드리지 않음

동일 Domain Relation을 다른 source도 지지하는 경우 한 source를 제거해도 relation은 merged graph에 남는다.

## Persistence

EvidenceLedger는 derived cache로 JSON 저장/복구할 수 있다.

예시 위치:

    knowledge-workspace/evidence-ledger.json

이 파일은 authoritative knowledge가 아니다. canonical source와 pipeline 결과에서 다시 생성할 수 있어야 한다.

## End-to-end API

EvidenceProvenancePipeline은 기존 Knowledge Core를 그대로 조합한다.

    extraction
      -> validation
      -> domain update
      -> evidence projection

기존 ExtractionPipeline, ValidationPipeline, DomainPipeline의 책임을 합치거나 재작성하지 않는다.

## Determinism boundary

provenance identity에는 다음과 같은 runtime/aggregate 상태를 넣지 않는다.

- generated fragment/raw edge/candidate UUID
- created_at timestamp
- dynamic relation runtime ID
- 누적 evidence_count
- 누적 domain_conf
- CREATE_NEW / UPDATE_EXISTING 같은 저장소 현재 상태

이 값들은 semantic evidence identity가 아니므로 동일 원문 재처리의 결정성을 깨뜨린다.

## 현재 확장 상태

Phase 7에서 selective ingestion이 연결됐다.

- source hash + processor stamp 기반 skip/reprocess
- non-mutating domain evaluation
- source-scoped evidence replacement
- affected Domain Relation deterministic reconciliation
- ingestion-state derived persistence
- transaction rollback/compensation

세부 구조는 docs/architecture/selective-ingestion.md를 참고한다.

## 현재 확장 상태

Phase 8에서 supporting/contradicting evidence score aggregation을 고도화했다.

- source별 최대 assertion quality 사용
- independent source diversity 반영
- strong conflict penalty
- deterministic current-ledger recomputation
- persisted scoring trace/version

세부 정책은 docs/architecture/evidence-scoring.md를 참고한다.

## 현재 확장 상태

Phase 9에서 원시 PDF/TXT/MD byte hash inventory와 extracted-text cache를 연결했다. 변경되지 않은 source는 text extraction 자체를 건너뛴다. 세부 구조는 docs/architecture/raw-file-inventory.md를 참고한다.

Phase 10에서 deterministic chunk ID, content hash, embedder stamp를 기반으로 embedding/vector index selective refresh를 추가했다. 변경되지 않은 chunk는 embedding을 다시 계산하지 않고, metadata/order 변경만 있을 때는 vector를 재사용한다. 세부 구조는 docs/architecture/selective-vector-refresh.md를 참고한다.

## 현재 확장 상태

Phase 11에서 성공적으로 커밋된 IngestionState/EvidenceLedger를 immutable chain으로 보존하는 as-of replay를 추가했다. commit-time 기준 point-in-time 상태를 read-only로 복원하며, live relation/vector state는 historical authority에 포함하지 않는다. 세부 구조는 docs/architecture/as-of-replay.md를 참고한다.

## 다음 단계

1. scenario / regime projection
