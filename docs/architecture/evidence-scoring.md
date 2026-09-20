# Evidence Score Aggregation

Phase 8은 selective ingestion이 만든 evidence graph에서 Domain Relation confidence를 결정적으로 계산한다.

## 문제

기존 reconciler는 support assertion 개수와 conflict assertion 개수만 사용했다.

이 방식은 다음 문제가 있다.

- 한 문서가 같은 주장을 여러 번 반복하면 독립 evidence처럼 과대계산될 수 있다.
- validation confidence 0.9와 0.2가 같은 한 표로 계산된다.
- source 다양성이 반영되지 않는다.
- 최종 domain_conf의 근거를 설명하기 어렵다.

## 원칙

1. assertion 개수와 독립 source 수를 구분한다.
2. 동일 source 안에서는 같은 relation에 대한 가장 강한 assertion 하나만 scoring mass에 반영한다.
3. assertion quality는 combined_conf를 우선 사용한다.
4. combined_conf가 없으면 semantic_conf / sign_score / student_conf 평균을 사용한다.
5. 모든 confidence field가 없으면 정책의 fallback quality를 사용한다.
6. support와 conflict를 각각 포화 함수로 정규화한다.
7. 입력 순서에 따라 결과가 달라지지 않는다.
8. scoring policy 변경은 processor fingerprint에 포함되어 unchanged source도 재평가된다.

## 계산

source별 최대 quality를 q_i라고 할 때:

    support_strength = sum(max_quality_per_support_source)
    conflict_strength = sum(max_quality_per_conflict_source)

포화 신호:

    support_score = 1 - exp(-support_strength)
    conflict_score = 1 - exp(-conflict_strength)

기본 정책:

    baseline = 0.50
    support_gain = 0.40
    conflict_penalty = 0.45

최종 confidence:

    domain_conf =
        clamp(
            baseline
            + support_gain * support_score
            - conflict_penalty * conflict_score,
            0.10,
            0.95
        )

conflict penalty를 support gain보다 약간 크게 두어 강한 상충 evidence가 존재할 때 관계 confidence가 과도하게 유지되지 않게 한다.

## 저장 필드

DynamicRelation에는 기존 필드를 유지하면서 다음 trace를 추가한다.

- support_score
- conflict_score
- support_source_count
- conflict_source_count
- evidence_score_version

기존 evidence_count / conflict_count는 raw assertion 개수다. source_count와 의미가 다르다.

## Source diversity

동일 source에서 assertion이 5개 반복되어도 scoring mass에는 source의 최대 quality 한 번만 들어간다.

반대로 독립 source 두 개가 각각 같은 relation을 지지하면 두 source quality가 모두 반영되어 confidence가 증가한다.

이 방식은 source_uri가 독립 evidence 단위라는 현재 ingestion 계약을 따른다.

## External relation

origin이 evidence_reconciled가 아닌 수동/외부 relation은 기존과 동일하게 overwrite/delete하지 않는다.

reconciliation report에는 현재 evidence의 support/conflict score trace를 기록하지만 외부 relation의 domain_conf는 보존한다.

## 결정성

다음은 score에 영향을 주지 않는다.

- assertion 입력 순서
- runtime UUID
- ingestion 처리 순서
- relation created_at / last_update
- 기존 relation의 누적 domain_conf

현재 EvidenceLedger 전체 상태에서 매번 다시 계산하므로 동일 ledger는 동일 score를 만든다.

## v12와의 관계

v12 EvidenceAccumulator의 EMA/변동성 개념은 그대로 복구하지 않는다.

현재 시스템은 source replacement가 가능한 derived ledger이므로 과거 실행 순서에 의존하는 EMA보다 현재 evidence set에서 confidence를 재계산하는 방식이 더 적합하다.

향후 time-series feature evidence가 들어오면 as-of/replay 계층 위에서 temporal aggregation을 별도 추가한다.

## 다음 단계

1. PDF/file byte inventory에서 text extraction 이전 skip
2. embedding/vector index selective refresh
3. as-of snapshot / replay
4. scenario/regime projection
