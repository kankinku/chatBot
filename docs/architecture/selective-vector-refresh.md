# Selective vector-index refresh

Phase 10은 Phase 9의 raw-file inventory와 selective ingestion 뒤에 있는
embedding/vector index를 **derived state**로 유지하면서 변경된 chunk만
갱신한다.

## 문제

기존 `VectorRetriever`는 Chroma collection의 record 수와 현재 chunk 수가
다르면 collection 전체를 다시 만들었다. 또한 vector ID가 실행 시 생성되는
UUID를 포함했기 때문에 다음 실행에서 동일 chunk를 안정적으로 식별할 수
없었다.

이 방식은 원본 PDF 한 개가 바뀌어도 전체 corpus를 다시 embedding할 수 있고,
변경되지 않은 chunk의 계산 비용을 재사용할 수 없다.

## 설계 원칙

1. source text와 evidence ledger가 source of truth다.
2. vector index와 manifest는 언제든 다시 만들 수 있는 derived cache다.
3. vector ID는 list 순서와 실행 시점에 의존하지 않는다.
4. text가 바뀐 chunk만 embedding을 다시 계산한다.
5. list 순서나 metadata만 바뀐 경우 embedding은 재사용한다.
6. embedding model의 의미가 바뀌면 현재 chunk 전체를 다시 embedding한다.
7. collection mutation이 중간 실패하면 manifest를 전진시키지 않는다.

## deterministic chunk identity

`chunk_vector_id()`는 다음 identity를 SHA-256으로 fingerprint한다.

- `source_uri`가 있으면 source identity로 사용하고, 없으면 `doc_id`
- `doc_id`
- `filename`
- `page`
- `start_offset`

chunk text 자체는 ID에 포함하지 않는다. 따라서 같은 위치의 chunk 내용이
바뀌어도 vector ID는 유지되고 해당 record만 upsert할 수 있다.

text는 별도의 `content_hash`로 추적한다.

## derived manifest

기본 manifest 위치:

```text
<vector_store_dir>/<collection_name>.manifest.json
```

각 record는 다음 값만 저장한다.

- deterministic `chunk_id`
- `content_hash`
- `metadata_hash`

manifest 상단에는 collection name과 embedder stamp를 기록한다.

embedder stamp는 최소한 다음 embedding semantics를 포함한다.

- embedder implementation class
- model name
- embedding dimension
- normalization 여부(지원하는 embedder)

manifest에는 `embedding_dim`도 별도로 기록한다. Chroma collection은 최초
embedding의 dimension을 고정하므로 dimension이 달라지면 같은 collection에
upsert하지 않고 derived collection을 재생성한 뒤 현재 chunk를 다시
embedding한다. manifest가 없거나 손상됐는데 기존 collection records가 남아
있는 legacy 상태도 동일하게 hard rebuild한다.

batch size, progress 출력, device 같은 실행 배치 옵션은 embedding 의미가
같으므로 stamp에 포함하지 않는다.

## refresh decision

현재 chunk와 이전 manifest, 실제 Chroma ID를 비교한다.

| 상태 | 동작 |
| --- | --- |
| 새 chunk | embedding + upsert |
| 동일 ID, text 변경 | embedding + upsert |
| 실제 Chroma record 유실 | embedding + upsert |
| text 동일, metadata/order 변경 | metadata update only |
| 완전 동일 | no-op |
| 현재 corpus에서 사라진 ID | stale vector 제거 |
| embedder stamp 변경, dimension 동일 | 현재 chunk 전체 재embedding |
| embedding dimension 변경 | collection 재생성 + 현재 chunk 전체 재embedding |
| manifest 누락/손상 + 기존 records 존재 | collection 재생성 후 derived index recovery |

list reorder는 `chunk_index` metadata만 바뀌므로 embedding 호출 없이 metadata만
갱신한다. 검색 결과는 항상 현재 `chunks` list의 index로 변환된다.

## failure semantics

각 embedding batch는 해당 collection mutation 직전에 계산한다.

mutation 순서는 다음과 같다.

1. changed/new chunk를 bounded batch로 embedding + upsert
2. metadata-only update
3. stale vector removal
4. manifest atomic save

manifest는 모든 collection mutation이 성공한 뒤에만 저장한다. 중간 단계에서
실패하면 이전 manifest가 남고 다음 실행에서 필요한 작업을 다시 계산한다.
embedding/upsert는 기본 100개 단위로 제한해 초기 구축이나 model refresh에서도
전체 corpus embedding을 한 번에 메모리에 올리지 않는다.

Chroma와 JSON manifest를 하나의 트랜잭션으로 묶을 수는 없으므로 완전한
원자성 대신 **replay 가능한 derived-state reconciliation**을 사용한다.

## 기존 Phase와 연결

```text
raw file byte hash
    ↓
selective text extraction
    ↓
source text hash / evidence projection
    ↓
selective relation reconciliation
    ↓
chunk content hash
    ↓
selective embedding/vector refresh
```

Phase 10은 Knowledge Core의 evidence/provenance truth를 변경하지 않는다.
vector index는 retrieval 가속용 파생 표현일 뿐이다.

## 현재 확장 상태

Phase 11에서 immutable IngestionState/EvidenceLedger snapshot chain과 commit-time as-of replay를 추가했다. historical Chroma/live relation row는 보존하지 않고 evidence state에서 파생하도록 경계를 유지한다. 세부 구조는 docs/architecture/as-of-replay.md를 참고한다.

## 다음 후보

1. scenario / regime projection
