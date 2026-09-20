# Selective Ingestion

Phase 7은 Knowledge Core 전체 재처리를 피하고 source별 변경만 재처리하는 계층이다.

## 입력 경계

이 계층은 PDF/OCR 같은 원시 파일 parsing 자체를 담당하지 않는다. upstream에서 텍스트 추출이 끝난 다음 다음 단위로 입력한다.

- doc_id
- source_uri
- text

source_uri는 논리적인 source identity이고 text의 SHA-256이 content identity다.

## 처리 흐름

    SourceDocument
      -> source hash / processor stamp 비교
      -> unchanged: skip
      -> added/changed/stale:
           extraction
           validation
           non-mutating domain evaluation
           evidence projection replacement
      -> removed source pruning
      -> affected Domain Relation recomputation
      -> one ingestion-state persistence

기존 EvidenceProvenancePipeline의 기본 동작은 그대로 domain update를 수행한다. SelectiveIngestionManager만 apply_domain_updates=False를 사용해 중복 누적을 피한다.

## 왜 Domain을 직접 증가시키지 않는가

기존 DynamicDomainUpdate는 evidence를 받을 때마다 evidence_count와 domain_conf를 누적한다. 수정된 같은 source를 다시 처리하면 동일 source를 새 evidence처럼 중복 가산할 수 있다.

Selective ingestion은 먼저 source별 EvidenceLedger를 교체한 뒤 현재 전체 ledger에서 affected relation만 다시 계산한다.

예:

- source A + source B가 같은 relation을 지지: evidence_count=2
- source A 내용 수정 후 재처리: 여전히 evidence_count=2
- source A 삭제: evidence_count=1
- source B도 삭제: evidence-managed relation 삭제

## Processor stamp

skip 조건은 source hash만 보지 않는다.

다음 범주의 Python 코드와 config/ontology YAML 전체를 hash하여 processor stamp를 만든다.

- shared
- extraction
- validation
- domain
- evidence
- ingestion
- llm
- workspace

source text가 같아도 processor stamp가 달라지면 REPROCESSED 처리한다.

## Relation reconciliation

affected Domain Relation은 현재 EvidenceLedger에서 다시 계산한다.

- support count
- conflict count
- majority polarity
- semantic tags
- domain_conf

domain_conf는 기존 DynamicDomainUpdate의 기본 강화/약화율과 호환되는 결정적 계산을 사용한다. 시간 decay와 처리 순서에는 의존하지 않는다.

Selective ingestion이 만든 relation은 origin=evidence_reconciled로 표시한다. 다른 origin의 외부/수동 relation은 덮어쓰거나 삭제하지 않는다.

## Transaction safety

relation recomputation은 KG transaction 하나에서 적용한다.

기존 KG adapter도 transaction 중 기존 entity/relation을 CREATE로 기록하던 문제를 수정해 실제 existing record는 UPDATE로 기록한다. TransactionManager는 before_state를 deep copy하여 in-memory mutation 이후에도 rollback snapshot을 보존한다.

state persistence가 relation commit 이후 실패하면 이전 EvidenceLedger를 deterministic하게 replay해 compensating reconciliation을 시도한다.

## Derived state

기본 상태 파일:

    knowledge-workspace/ingestion-state.json

이 파일에는 source별:

- source_hash
- projection_hash
- processor_stamp
- EvidenceLedger projection

이 저장된다. canonical source-of-truth가 아니며 재생성 가능한 derived state다.

## CLI

입력 JSONL:

    {"doc_id":"doc-1","source_uri":"file://docs/a.txt","text":"..."}
    {"doc_id":"doc-2","source_uri":"file://docs/b.txt","text":"..."}

실행:

    python3 scripts/knowledge_ingest.py --input documents.jsonl

기본적으로 입력에서 사라진 기존 source는 prune한다. 부분 batch라면:

    python3 scripts/knowledge_ingest.py --input changed.jsonl --no-prune

LLM 경로를 사용하려면:

    python3 scripts/knowledge_ingest.py --input documents.jsonl --use-llm

## 실패 의미

개별 source extraction/validation 실패는 기존 projection을 유지하고 FAILED로 보고한다. 다른 source는 계속 처리한다.

relation transaction 실패 시 새 ingestion state를 저장하지 않는다. state 저장 실패가 relation commit 뒤 발생하면 이전 ledger를 이용한 compensating reconciliation을 수행한다.

## 다음 단계

Phase 8 후보:

1. PDF/file inventory에서 source byte hash를 먼저 비교하여 text extraction 자체도 skip
2. embedding/vector index selective refresh
3. evidence score aggregation 정책
4. as-of snapshot/replay
5. scenario/regime projection
