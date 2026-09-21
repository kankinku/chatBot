# As-of Knowledge Core replay

Phase 11은 현재 상태만 유지하던 selective ingestion 위에 **불변 historical
knowledge-state chain**을 추가한다. 목적은 "현재 파일을 과거 시각으로
필터링"하는 것이 아니라, **그 시점까지 시스템에 성공적으로 커밋되어 있던
Knowledge Core 상태**를 다시 여는 것이다.

## 시간 의미

Phase 11의 `as_of`는 system/knowledge commit time이다.

> requested time 이하에서 가장 최근에 성공적으로 커밋된 snapshot

문서 본문에 등장하는 사건 시각이나 실제 세계 event time은 이 계층의
`as_of` 의미가 아니다. event-time/bitemporal 모델은 별도 확장 대상이다.

모든 snapshot 시각은 timezone-aware 값만 허용하고 UTC로 정규화한다.
naive datetime은 추정하지 않고 거부한다.

같은 시각에 여러 상태가 커밋될 수 있으므로 선택 순서는 다음과 같다.

1. `committed_at`
2. monotonic `sequence`

## Snapshot 내용

한 snapshot은 성공적으로 커밋된 `IngestionState` 전체를 보존한다.

- ingestion records
- source-scoped `EvidenceLedger`
- 각 `EvidenceProjection`의 deterministic graph
- fragment text와 provenance
- processor stamp

따라서 source가 나중에 변경되거나 삭제되어도 이전 snapshot에는 기존
projection과 fragment text가 남는다.

반대로 다음은 historical authority로 snapshot하지 않는다.

- live Neo4j / in-memory DynamicRelation rows
- Chroma vector index
- vector manifest
- extracted-text cache
- 원본 PDF/TXT/MD byte
- transient LLM/runtime cache

이 값들은 replay snapshot의 EvidenceLedger 또는 원본 입력에서 파생되는
runtime/derived state다.

## Snapshot identity

현재 state digest는 canonical JSON 형태의 `IngestionState.to_dict()`를
SHA-256으로 fingerprint한다.

snapshot ID는 timestamp가 아니라 다음 chain identity로 계산한다.

```text
ksnap_SHA256({
  parent_snapshot_id,
  state_digest
})
```

이 규칙의 결과:

- 동일한 최신 state를 다시 sync해도 snapshot을 추가하지 않는다.
- state가 바뀌었다가 나중에 같은 내용으로 돌아와도 parent가 다르므로 새로운
  역사 노드가 된다.
- timestamp 충돌이 snapshot identity를 깨뜨리지 않는다.

## 저장 구조

기본 runtime 위치:

```text
knowledge-workspace/replay/
├─ index.json
└─ snapshots/
   ├─ ksnap_<hash>.json
   └─ ...
```

`knowledge-workspace/`는 Git source tree가 아니라 runtime workspace다.

`index.json`에는 순서/chain/time/digest와 compact change summary만 저장한다.
각 snapshot blob은 full `IngestionState`를 포함한다.

blob은 immutable이다. 같은 ID에 다른 chain/state payload가 이미 존재하면
overwrite하지 않고 integrity error로 실패한다. snapshot blob은 먼저 atomic
write하고 index를 atomic replace한다. index write 전 프로세스가 끊겨 orphan
blob이 남은 경우에는 동일 chain/state인지 검증한 뒤 해당 blob을 재사용할 수
있다.

## 무결성 검증

history load와 `verify`는 다음을 fail-closed로 검사한다.

1. replay schema version
2. sequence가 1부터 연속인지
3. 첫 parent가 null인지
4. 이후 parent가 바로 앞 snapshot을 가리키는지
5. committed time이 뒤로 이동하지 않는지
6. index가 가리키는 blob이 존재하는지
7. index metadata와 blob metadata가 일치하는지
8. `IngestionState.from_dict()` validation
9. EvidenceProjection graph/projection hash validation
10. state digest 재계산 일치
11. snapshot ID 재계산 일치
12. duplicate snapshot ID 부재

손상된 index나 blob에서 history를 추측해 재구축하지 않는다.

## Selective ingestion commit

Replay recording을 사용하는 ingestion commit은 다음 순서다.

```text
target IngestionState 계산
        ↓
target state validate
        ↓
affected live relation reconcile
        ↓
current ingestion-state.json atomic save
        ↓
immutable replay snapshot record
        ↓
report 반환
```

snapshot recording 실패가 current-state save 뒤 발생하면:

1. 이전 EvidenceLedger로 managed relation을 compensating reconcile
2. 이전 `IngestionState`를 다시 저장
3. 원래 오류를 호출자에게 전달

보상 복구 자체가 실패하면 double-failure를 명시적으로 보고한다.

current-state save 자체가 실패한 경우에는 atomic state store가 기존 파일을
보존하므로 이전 relation state만 보상 복구한다.

## Add / update / remove

별도 tombstone event log 없이 full-state snapshot으로 as-of를 재현한다.

- add: 새 snapshot부터 source가 존재
- update: 이전 snapshot은 old projection, 새 snapshot은 replacement projection
- remove: 이전 snapshot에는 존재하고 새 snapshot에서는 사라짐
- processor reprocess: record/projection 상태가 달라지면 새 state digest와 snapshot
- 완전 no-op: snapshot 추가 없음

`change_summary`는 감사/가독성용 metadata이며 state reconstruction의 authority가
아니다.

## Replay API

`KnowledgeReplayStore`:

```python
record(state, committed_at=..., change_summary=..., origin=...)
latest()
get(snapshot_id)
as_of(datetime)
list()
verify()
```

`KnowledgeReplayService`:

```python
state_as_of(datetime)
state_by_snapshot(snapshot_id)
```

`ReplayState.state`는 defensive copy를 반환한다. replay caller가 반환 객체를
수정해도 immutable history blob이나 live state는 변하지 않는다.

`ReplayState.merged_graph()`는 당시 EvidenceLedger에서 deterministic merged
graph를 다시 만든다.

## CLI

일반 ingestion은 기본적으로 replay history도 기록한다.

```bash
python3 scripts/knowledge_ingest.py --input documents.jsonl
python3 scripts/knowledge_ingest.py --source-dir ./data/pdfs
```

history 확인:

```bash
python3 scripts/knowledge_replay.py list
python3 scripts/knowledge_replay.py show --snapshot ksnap_...
python3 scripts/knowledge_replay.py as-of --at 2026-09-21T10:00:00+00:00
python3 scripts/knowledge_replay.py verify
```

Phase 11 도입 전에 존재하던 현재 ingestion state를 history 시작점으로 채택하려면:

```bash
python3 scripts/knowledge_replay.py bootstrap \
  --state knowledge-workspace/ingestion-state.json
```

bootstrap은 replay history가 비어 있을 때만 허용되며, 지정한 current-state
파일이 실제로 존재해야 한다. 누락된 경로를 빈 state로 간주해 baseline을
생성하지 않는다. 현재 상태 이전의 과거를 추정하거나 생성하지 않으므로 첫
snapshot보다 이른 `as_of`는 "history unavailable"로 처리된다.

## Derived 관계와 vector

Replay history는 live DynamicRelation row를 source of truth로 사용하지 않는다.
과거 relation evidence와 score는 당시 snapshot의 EvidenceLedger에서 다시 계산할
수 있다.

Phase 10 Chroma/vector index는 계속 current-corpus derived state다. Phase 11은
historical vector search를 제공하지 않는다. 이 경계를 유지해야 과거 knowledge
state가 특정 Chroma 버전이나 embedding model에 묶이지 않는다.

## v12 replay에서 가져온 것과 버린 것

가져온 개념:

- point-in-time replay context
- read-only snapshot view
- requested time 이하의 최신 snapshot 선택

계승하지 않은 동작:

- 과거 `as_of` snapshot을 만들면서 current graph/latest evidence를 읽는 방식
- `SNAP_YYYYMMDD_HHMMSS` 시간 기반 identity
- snapshot UPSERT
- 무결성 digest 없는 저장
- parent chain 없는 snapshot 집합
- naive datetime과 replay time 혼용

## 다음 단계

1. scenario / regime projection
