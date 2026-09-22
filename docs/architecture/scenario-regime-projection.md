# Scenario / Regime Projection

Phase 12는 canonical Knowledge Core 위에 **가정 기반 read-only projection**을 추가한다.

핵심 원칙은 다음과 같다.

> Evidence는 사실 근거이고, Scenario/Regime은 그 근거의 적용 조건을 바꾸는 파생 가정이다.

따라서 scenario 실행은 canonical evidence, replay history, live relation repository,
Chroma/vector index를 수정하지 않는다.

## 데이터 흐름

```text
IngestionState / Replay Snapshot
          ↓
EvidenceLedger merged graph
          ↓
RelationEvidenceViewBuilder
          ↓
Evidence-backed relation view
          ↓
Scenario assumptions + optional RegimeSpec
          ↓
ProjectedRelation graph
          ↓
Scenario shocks
          ↓
ProjectedImpact / sensitivity / provenance trace
```

## Evidence 경계

`EvidenceScoreSummary`는 scenario/regime 실행 중 변경하지 않는다.

Evidence-backed relation의 projection weight:

```text
projected_weight =
  clamp(
    evidence_score
    × regime_multiplier
    × scenario_multiplier,
    0,
    1
  )
```

여기서:

- `evidence_score`: canonical evidence로 계산된 base score
- `regime_multiplier`: 명시적 regime applicability
- `scenario_multiplier`: hypothetical scenario modifier
- `projected_weight`: 현재 가정에서 reasoning에 사용할 applicability weight

`projected_weight`는 probability도 아니고 새로운 evidence도 아니다.

## 공통 relation evidence view

Phase 12는 기존 `EvidenceRelationReconciler` 내부의 evidence derivation을
`RelationEvidenceViewBuilder`로 분리한다.

이 builder가 계산하는 값:

- relation stable key
- base sign
- EvidenceScoreSummary
- support/conflict assertion IDs
- source refs
- semantic tags
- entity display names

따라서 live reconciliation과 replay/scenario projection이 동일한 evidence
semantics를 사용한다.

## ScenarioSpec

지원되는 assumption:

### RelationScaleAssumption

기존 relation의 scenario applicability를 `0..2` multiplier로 조정한다.

### RelationDisableAssumption

relation을 projection에서 inactive 처리한다. relation 자체는 trace에 남고
`projected_weight=0`으로 표시된다.

### RelationSignOverrideAssumption

가정된 sign을 projection에 적용한다.

base evidence sign은 별도로 보존한다.

### InjectRelationAssumption

base graph에 이미 존재하는 entity 사이에 hypothetical relation을 추가한다.

- 새 entity 생성은 지원하지 않는다.
- evidence assertion을 만들지 않는다.
- evidence count를 증가시키지 않는다.
- `origin=hypothesis`
- `evidence_score=None`
- 명시적인 `assumed_weight`를 사용한다.

기존 evidence relation과 같은 stable key를 주입하면 거부한다.

## RegimeSpec

Phase 12의 regime은 자동 탐지 결과가 아니라 **명시적인 relation applicability
profile**이다.

각 `RegimeRule`은 selector와 `0..2` multiplier를 가진다.

여러 rule이 같은 relation에 적용되면 deterministic rule ID 순서로 적용하고
곱한 뒤 `0..2`로 bound한다.

Phase 12는 다음을 제공하지 않는다.

- risk_on / risk_off 같은 finance-specific 기본 preset
- VIX/CPI 등을 읽는 자동 detector
- regime policy learning

향후 detector가 필요하면 동일한 `RegimeSpec` 계약을 출력하도록 추가할 수 있다.

## Deterministic identity

모든 semantic identity는 canonical hash 기반이다.

```text
scenario_spec_id = scn_<hash(semantic scenario fields)>
regime_spec_id   = rgm_<hash(semantic regime fields)>

projection_id =
  proj_<hash(
    engine_version,
    base state identity,
    scenario_spec_id,
    regime_spec_id
  )>
```

label, description, runtime timing은 semantic identity에 포함하지 않는다.

base state, scenario, regime, engine version이 같으면 projection ID와 output digest도
같다.

## Base state

세 가지 실행 기준을 지원한다.

### Snapshot

`KnowledgeReplayService.state_by_snapshot()`을 사용한다.

### As-of

`KnowledgeReplayService.state_as_of()`을 사용한다.

요청 시각 이전 snapshot이 없으면 추정하지 않고 실패한다.

### Current

현재 `IngestionState`를 사용한다.

현재 state digest가 latest replay snapshot과 같으면 snapshot identity로
normalize한다. 그렇지 않으면 transient `current_state` base로 처리한다.

snapshot identity가 없는 transient current projection은 기본 cache에 저장할 수
없다.

## Shock propagation

`ScenarioShock`:

- existing entity ID
- direction: `+` / `-`
- magnitude: `0..1`

전파 규칙:

1. active projected relation만 사용
2. sign이 `+` 또는 `-`인 edge만 사용
3. path 내부 cycle 금지
4. `max_depth` 적용
5. `max_paths` 적용
6. path sign은 shock sign과 edge sign의 곱
7. impact는 shock magnitude와 projected edge weights의 곱

```text
impact =
  shock_magnitude
  × edge_1.projected_weight
  × edge_2.projected_weight
  × ...
```

숨겨진 finance relation map이나 추가 hop decay 상수는 사용하지 않는다.

## Correlated path 처리

같은 entity로 여러 경로가 도달해도 모두 독립 근거처럼 합산하지 않는다.

node summary는:

- strongest positive path
- strongest negative path
- net = strongest_positive - strongest_negative

를 보존한다.

## Sensitivity

임의 threshold를 자동 생성하지 않는다.

사용자가 `ScenarioSpec.sensitivity_thresholds`에 명시한 조건만 평가한다.

지원 operator:

- `<`
- `<=`
- `>`
- `>=`

실제 적용된 assumption/regime/shock path는 `ProjectionDependency`로 별도
기록한다.

## Read-only reasoning integration

`GraphRetrieval`은 기존 live domain object 자체가 아니라
`get_all_relations()` read contract를 사용할 수 있게 확장한다.

Phase 12는:

- `DynamicRelationProvider`
- `ProjectedRelationProvider`

를 제공한다.

Projected relation을 Neo4j나 live DomainKGAdapter에 upsert하지 않는다.

## Vector / RAG 경계

Scenario projection은 Chroma를 수정하지 않는다.

- projected relation을 embedding하지 않는다.
- scenario assumption으로 vector manifest를 변경하지 않는다.
- historical snapshot projection에 current vector search를 자동 혼합하지 않는다.

historical vector search는 Phase 12 범위 밖이다.

## Derived cache

재현 가능한 snapshot-backed projection은 optional cache로 저장할 수 있다.

```text
knowledge-workspace/projections/
└─ proj_<hash>.json
```

특성:

- content-addressed
- immutable
- 삭제 후 재생성 가능
- 동일 내용은 idempotent
- 동일 ID의 다른 내용은 integrity failure
- unique temp file + atomic create
- semantic ID / output digest / trace metadata 재검증

mutable projection index는 두지 않는다.

## CLI

```bash
python3 scripts/knowledge_project.py project \
  --scenario scenario.json \
  --snapshot ksnap_...

python3 scripts/knowledge_project.py project \
  --scenario scenario.json \
  --regime regime.json \
  --as-of 2026-09-22T00:00:00+00:00 \
  --persist

python3 scripts/knowledge_project.py project \
  --scenario scenario.json \
  --current

python3 scripts/knowledge_project.py show --projection proj_...
python3 scripts/knowledge_project.py verify --projection proj_...
```

`--persist`는 reproducible snapshot-backed projection에만 허용된다. `--current`
실행도 현재 state digest가 latest replay snapshot과 동일해 snapshot identity로
정규화된 경우에만 저장할 수 있다. replay와 다른 transient current state는
projection 결과를 출력할 수 있지만 cache에 영구 저장하지 않는다.

최소 scenario JSON:

```json
{
  "assumptions": [],
  "shocks": [],
  "sensitivity_thresholds": [],
  "max_depth": 4,
  "max_paths": 100
}
```

초기 interchange format은 JSON만 사용한다.

## v12에서 가져온 개념

가져온 것:

- normalized shock spec
- as-of pinned context
- regime relation applicability
- break/sensitivity concept
- trace/provenance

계승하지 않은 것:

- finance-specific hard-coded relation map
- `SIM_<timestamp>` ID
- UUID scenario identity
- fabricated snapshot reference
- current data와 historical data 혼합
- regime modifier로 evidence score 자체 변경
- 자동 policy learning
- outcome verification/backtesting

## 완료 이후

Phase 12까지 병합되면 계획된 modernization 핵심 architecture phase는 완료된다.

이후 작업은 필수 architecture phase가 아니라 optional hardening으로 분류한다.

현재 알려진 후보:

- ingestion/replay inter-process locking
- local `Chatbot_v6` residue에 영향을 받지 않는 validation layout 판별 강화
