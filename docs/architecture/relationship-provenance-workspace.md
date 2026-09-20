# Relationship & Provenance Workspace

이 계층은 canonical Knowledge Core의 source-of-truth를 대체하지 않는다. 관계 graph, cards, fingerprint, parse cache는 모두 다시 만들 수 있는 local derived state다.

## 목적

- entity / relation / config / source 사이의 관계를 agent가 빠르게 탐색한다.
- 모든 derived node/edge에 source path + SHA-256 provenance를 남긴다.
- source 변경 시 바뀐 파일만 재파싱한다.
- cold build와 incremental build가 동일한 graph를 생성하게 한다.
- dependency edge를 따라 변경 영향 범위(blast radius)를 계산한다.

## Source of truth

기본 입력은 다음 두 canonical tree다.

- config/ontology
- data/ontology

config/knowledge_workspace.yaml 자체도 fingerprint에 포함된다. 생성물은 knowledge-workspace/에 저장하며 Git에는 커밋하지 않는다.

## Graph model

meta relation은 폐쇄형 vocabulary를 사용한다.

- part_of
- depends_on
- produces
- configures
- validates
- derived_from
- supported_by
- contradicted_by
- domain_relation

Affect, Cause, DependOn, TemporalBefore 같은 domain relation 이름은 meta relation을 늘리지 않고 domain_relation edge의 semantic_type으로 저장한다.

원본 relation record는 assertion node로 별도 보존한다. 따라서 같은 의미 관계에 여러 evidence source가 추가되어도 개별 주장과 provenance를 잃지 않는다.

## Freshness

각 source file은 SHA-256으로 fingerprint한다. mtime만으로 freshness를 판단하지 않는다.

- hash unchanged: cached parse 재사용
- hash changed: 해당 source만 재파싱
- file add/remove: manifest와 graph에서 반영
- generator code changed: cache 무효화
- final graph: stable sort + invariant validation

## CLI

    python3 scripts/knowledge_workspace.py build
    python3 scripts/knowledge_workspace.py check
    python3 scripts/knowledge_workspace.py impact source:config/ontology/entity_types.yaml

강제 cold rebuild:

    python3 scripts/knowledge_workspace.py build --cold

impact는 stale workspace에서 기본적으로 실패한다. 먼저 build해 현재 source hash와 graph를 일치시키는 것이 원칙이다.

## Graft에서 가져온 원칙

Graft의 구현에서 다음 원칙만 차용한다.

1. graph는 local regenerable cache다.
2. provenance와 freshness는 content hash에 묶는다.
3. 관계 vocabulary는 제한하고 의미가 애매한 edge를 남발하지 않는다.
4. incremental 결과는 cold rebuild와 동일해야 한다.
5. structural/semantic relation과 recomputation dependency relation을 구분한다.

코드 AST graph 자체를 복제하지 않는다. chatBot에서는 domain knowledge와 evidence provenance에 맞는 schema로 재구성한다.

## 현재 확장 상태

Phase 6에서 다음 기반을 추가했다.

1. extraction 결과의 source_uri/source_hash 보존
2. Document -> Fragment -> Assertion provenance projection
3. supported_by / contradicted_by evidence 연결
4. source-scoped EvidenceLedger와 relation recompute 대상 계산
5. ledger derived-cache persistence

세부 구조는 docs/architecture/evidence-provenance.md를 참고한다.

## Phase 7 확장

Selective ingestion 계층이 추가되어 source hash와 processor stamp를 기준으로 변경 source만 extraction/validation/evidence replacement하고 affected relation만 재계산한다.

세부 구조는 docs/architecture/selective-ingestion.md를 참고한다.

## 다음 단계

1. raw file inventory 단계 selective extraction
2. vector/embedding index selective refresh
3. evidence score aggregation 고도화
4. as-of snapshot / replay
5. scenario / regime projection
