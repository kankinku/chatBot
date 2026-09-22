# Raw File Inventory

Phase 9는 PDF/TXT/MD 원본을 Knowledge Core에 넣기 전에 **원시 파일 bytes를 먼저 fingerprint**하여, 변경되지 않은 파일의 text extraction 자체를 건너뛴다.

## 계층

기존 Phase 7의 SelectiveIngestionManager는 여전히 text 입력 계층이다.

    raw file
      -> byte SHA-256 / extractor stamp
      -> SelectiveFileIngestionManager
           unchanged -> cached extracted text reuse
           changed   -> text extraction
      -> SourceDocument
      -> SelectiveIngestionManager
           text hash / processor stamp
           extraction / validation / evidence / relation reconciliation

즉 두 단계 skip이 존재한다.

1. file bytes가 같으면 PDF/text extraction skip
2. extracted text와 Knowledge Core processor가 같으면 knowledge processing skip

PDF의 metadata만 바뀌어 byte hash가 달라졌지만 추출 text가 동일한 경우에는 PDF extraction은 다시 수행하지만 Knowledge Core는 두 번째 단계에서 skip할 수 있다.

## File identity

한 file-inventory state는 하나의 source root에 고정된다.

source root 아래 상대 경로가:

    reports/daily/risk.pdf

이면 논리 source URI는:

    file:reports/daily/risk.pdf

이고 doc_id는:

    reports/daily/risk

이다.

source root가 달라진 기존 inventory state를 재사용하면 fail-closed 한다. 서로 다른 root를 관리하려면 inventory state 파일도 분리한다.

## Byte hash

source bytes는 SHA-256으로 streaming hash한다. mtime/size는 freshness 기준으로 사용하지 않는다.

따라서:

- mtime만 변경: extraction skip
- bytes 변경: extraction 수행
- bytes 원복: 해당 bytes/extractor 조합에 cache가 남아 있으면 재사용 가능

현재 cache GC는 최신 inventory에서 참조되지 않는 text cache를 정리하므로 오래된 버전 text는 보존하지 않는다.

## Extractor stamp

동일 bytes라도 extractor 동작이 바뀌면 재추출해야 한다.

extractor stamp에는 다음이 들어간다.

- ingestion/file_inventory.py hash
- preprocessing/pdf_extractor.py hash
- 설치된 PyMuPDF 버전

PyMuPDF가 설치되지 않은 환경은 dependency marker를 missing으로 fingerprint한다. 실제 PDF extraction은 기존 PDFExtractor 계약에 따라 실패한다.

## Extracted text cache

기본 위치:

    knowledge-workspace/extracted-text/<cache-key>.txt

cache key는 byte_hash와 extractor_stamp의 canonical JSON SHA-256으로 계산한다.

inventory record는 byte_hash, extractor_stamp, text_hash, cache_key를 모두 보존한다. load 시 cache key를 다시 계산하고, cache read 시 extracted text SHA-256을 다시 계산한다.

cache가 없거나 손상된 경우 원본 file에서 재추출한다.

## Derived state

기본 inventory:

    knowledge-workspace/file-inventory.json

기본 Knowledge Core state:

    knowledge-workspace/ingestion-state.json

둘 다 canonical source-of-truth가 아니다.

.gitignore로 Git에서 제외되며, .ignore에서도 inventory/state/extracted text를 제외해 agent search 결과를 오염시키지 않는다.

## Failure policy

새 file extraction 실패:
- FAILED 보고
- Knowledge Core에는 추가하지 않음

기존 file이 변경됐는데 extraction 실패:
- 이전 extracted text cache가 정상이라면 이전 SourceDocument를 downstream에 전달
- 기존 evidence projection 유지
- inventory record도 이전 byte hash 상태 유지
- 다음 실행에서 변경된 file을 다시 시도

이전 cache까지 손상되어 복구할 수 없으면 해당 실행에서는 pruning을 비활성화한다. 추출 장애가 기존 evidence 삭제로 확대되는 것을 막기 위한 fail-safe다.

## Scoped pruning

파일 모드와 JSONL/connector 모드가 같은 ingestion-state를 사용할 수 있다.

그래서 SelectiveIngestionManager에 prune_scope를 추가했다.

File manager는 previous file inventory sources UNION current file inventory sources만 prune_scope로 전달한다.

따라서 file directory에서 문서가 사라져도 그 file evidence만 제거되며, JSONL이나 connector가 소유한 unrelated source는 삭제하지 않는다.

기존 JSONL CLI는 prune_scope를 지정하지 않으므로 기존 full-batch semantics를 그대로 유지한다.

## CLI

기존 JSONL 경로:

    python3 scripts/knowledge_ingest.py --input documents.jsonl

원시 file directory 경로:

    python3 scripts/knowledge_ingest.py --source-dir ./data/pdfs

기본 file pattern:

    **/*.pdf
    **/*.txt
    **/*.md

패턴을 직접 지정하려면 반복한다.

    python3 scripts/knowledge_ingest.py --source-dir ./data --pattern '**/*.pdf' --pattern '**/*.md'

부분 inventory로 삭제를 판단하면 안 되는 경우:

    python3 scripts/knowledge_ingest.py --source-dir ./data --no-prune

## Path safety

scan된 path는 resolve 후 source root 내부인지 확인한다.

source root 밖을 가리키는 symlink는 거부한다. 동일 실제 파일이 여러 경로로 resolve되는 alias도 거부해 source identity 중복을 방지한다.

## 현재 확장 상태

Phase 10에서 raw-file/text selective refresh 이후의 retrieval vector index도 selective refresh로 연결했다. deterministic chunk ID와 content hash를 사용하므로 원본 파일 일부 변경이 전체 corpus re-embedding으로 이어지지 않는다. 세부 구조는 docs/architecture/selective-vector-refresh.md를 참고한다.

## 현재 확장 상태

Phase 11에서 성공적으로 커밋된 IngestionState/EvidenceLedger를 immutable chain으로 보존하는 as-of replay를 추가했다. commit-time 기준 point-in-time 상태를 read-only로 복원하며, live relation/vector state는 historical authority에 포함하지 않는다. 세부 구조는 docs/architecture/as-of-replay.md를 참고한다.

## 다음 단계

1. scenario / regime projection
