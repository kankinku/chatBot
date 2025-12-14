# 조언자(Advisor) 시스템 구축 로드맵

> **목표:** "실시간 업데이트 + 증거기반 검증 + 과거사례 재현 + 시나리오"를 갖춘 조언자 시스템
> **전제:** 초기 1회 풀 수집 후, 이후는 **delta만** 수집

---

## 현재 상태 분석 (As-Is)

### ✅ 이미 구현된 것

| 계층 | 모듈 | 설명 |
|------|------|------|
| **Extraction** | FragmentExtractor, NERStudent, EntityResolver, RelationExtractor | 텍스트→엔티티/관계 추출 |
| **Validation** | SchemaValidator, SignValidator, SemanticValidator, ConfidenceFilter | RawEdge 타당성 검증 |
| **Domain** | DomainCandidateIntake, StaticDomainGuard, DynamicDomainUpdate, ConflictAnalyzer, DriftDetector | 불변 지식 관리 |
| **Personal** | PersonalCandidateIntake, PCSClassifier, PersonalKGUpdate, PersonalDriftAnalyzer | 개인 지식 저장 |
| **Reasoning** | QueryParser, GraphRetrieval, EdgeWeightFusion(EES), PathReasoningEngine, ConclusionSynthesizer | 그래프 기반 추론 |
| **Learning** | TrainingDatasetBuilder, GoldsetManager, Trainer, PolicyLearner, Deployment, Dashboard | 시스템 개선 |
| **Storage** | GraphRepository (InMemory/Neo4j), VectorStore, DocumentStore | 저장소 추상화 |

### ❌ 조언자 시스템에 부족한 것

| 우선순위 | 결함 | 영향도 | 현재 상태 |
|---------|------|-------|----------|
| **1** | Evidence 결합 부재 | 🔴 치명적 | Edge가 "주장"일 뿐 "검증된 주장"이 아님 |
| **2** | 시계열/버전/레짐 모델 부족 | 🔴 치명적 | 업데이트/드리프트/충돌이 "설명"이 아닌 "구조"로 존재해야 함 |
| **3** | 증분 갱신 설계 부족 | 🟡 중요 | delta 수집해도 전체 재계산이면 운영 불가 |
| **4** | 리플레이(과거 재현) 부재 | 🟡 중요 | 과거 사례검증 불가 → 학습/정책 업데이트 공허 |
| **5** | 시나리오 Shock 모델 부재 | 🟠 유용 | 방향/강도 전파를 정량 입력으로 제어 불가 |

---

## 0단계: 스키마/계약(Contract) 확정 (1~2일)

> **목표:** 이후 모든 단계가 의존할 "공통 언어" 정의

### 0-1. 관계 타입 메타데이터 확장

현재 `config/relation_types.yaml`에 다음 필드 추가:

```yaml
relation_types:
  Affect:
    description: "A가 B에 영향을 미침"
    has_polarity: true
    # === 신규 필드 ===
    meta:
      default_lag_days: [0, 30, 90]        # 지연 효과 후보
      typical_decay_halflife: 60           # 신뢰도 반감기(일)
      regime_applicability:                # 레짐별 적용 강도
        risk_on: 1.0
        risk_off: 0.6
        inflation_up: 0.8
```

### 0-2. 핵심 스키마 정의

| 개념 | 설명 | 저장 위치 |
|------|------|----------|
| **Fact** | 불변 정의 (예: "금리↑ → 성장주↓") | `data/domain/facts.json` |
| **Edge** | 관계 인스턴스 (confidence, evidence_count 포함) | GraphRepository |
| **Observation** | 시계열 수치 (series_id, timestamp, value, as_of) | TimeSeriesRepository (신규) |
| **Event** | 비정형 사건 (source_id, external_id, content, occurred_at) | EventRepository (신규) |

### 0-3. 산출물

- [x] `config/relation_types.yaml` 확장 (lag, decay, regime 필드) ✅ 완료
- [x] `src/shared/schemas.py` 신규: `Observation`, `Event`, `EvidenceSpec` Pydantic 모델 ✅ 완료
- [x] `docs/SCHEMA_CONTRACT.md`: 스키마 계약 문서 ✅ 완료

---

## 1단계: Delta Ingestion + Append-only 저장소 (3~5일)

> **목표:** "업데이트만 수집"이 실제로 성립하는 상태

### 1-1. 신규 모듈

```
src/ingestion/
├── __init__.py
├── source_registry.py      # SourceRegistry: 소스별 delta 방식 정의
├── fetch_state_store.py    # FetchStateStore: 수집 위치 저장
├── delta_fetcher.py        # DeltaFetcher: 신규/수정 데이터만 수집
├── normalizer.py           # Normalizer: 단위/타임존/결측 처리
└── idempotency_guard.py    # IdempotencyGuard: 중복 방지
```

| 모듈 | 책임 | 의존성 |
|------|------|-------|
| **SourceRegistry** | 소스별 delta 방식 정의 (since_timestamp / cursor / etag / hash-diff) | `config/sources.yaml` |
| **FetchStateStore** | (source, stream)별 마지막 수집 위치 저장 | SQLite |
| **DeltaFetcher** | state 기반 신규/수정 데이터만 수집 + 재시도 | SourceRegistry, FetchStateStore |
| **Normalizer** | 단위/타임존/결측 처리 → `Observation`/`Event`로 변환 | shared/schemas |
| **IdempotencyGuard** | 중복 방지 (시계열: series_id+timestamp+as_of / 이벤트: source_id+external_id) | - |

### 1-2. 신규 저장소

```
src/storage/
├── timeseries_repository.py  # TimeSeriesRepository
└── event_repository.py       # EventRepository
```

| 저장소 | 특성 | 핵심 API |
|--------|------|---------|
| **TimeSeriesRepository** | Append-only + as_of 버전 | `append_batch()`, `get_range()`, `get_last()` |
| **EventRepository** | Append-only + revision | `append()`, `get_by_external_id()`, `get_range()` |

### 1-3. 수용 기준

- [x] 같은 배치를 2번 넣어도 결과 동일 (멱등성) ✅ IdempotencyGuard 구현
- [x] 수정 데이터가 들어와도 과거 raw 보존 (as_of 분리) ✅ TimeSeriesRepository 구현
- [x] "초기 1회 이후 delta만"으로 계속 최신 유지 가능 ✅ DeltaFetcher 구현

---

## 2단계: Feature 계산(증분) + 의존성 인덱스 (2~3일)

> **목표:** 원시 데이터를 "검증 가능한 지표"로 바꾸고, 바뀐 부분만 재계산

### 2-1. 신규 모듈

```
src/features/
├── __init__.py
├── feature_spec_registry.py    # FeatureSpecRegistry
├── feature_dependency_index.py # FeatureDependencyIndex
├── feature_builder.py          # FeatureBuilder (증분)
└── feature_store.py            # FeatureStore
```

| 모듈 | 책임 |
|------|------|
| **FeatureSpecRegistry** | Feature 정의 (스프레드, ROC, zscore, YoY 등) 스펙 보관 |
| **FeatureDependencyIndex** | `series → feature[]` 의존성 맵 (역인덱스) |
| **FeatureBuilder** | Window 기반 지표 계산 (윈도우 길이만큼 과거 포함) |
| **FeatureStore** | Feature 값 저장 (series_id, feature_type, timestamp, value) |

### 2-2. 의존성 예시

```python
# config/features.yaml
features:
  SOFR_EFFR_SPREAD:
    type: spread
    inputs: [SOFR, EFFR]
    window_days: 0  # 당일
    
  SOFR_ZSCORE_30D:
    type: zscore
    inputs: [SOFR]
    window_days: 30
```

### 2-3. 수용 기준

- [x] Series 하나 업데이트 시 관련 feature만 재계산 ✅ FeatureDependencyIndex 구현
- [x] 하루치 업데이트로 전체 feature 재계산 발생 안 함 ✅ compute_affected() 메서드 구현

---

## 3단계: Evidence Layer (관계 ↔ 지표 매핑 + 점수화) (5~7일) ⭐핵심

> **목표:** KG의 edge를 "실데이터로 검증되는 주장"으로 승격

### 3-1. 신규 모듈

```
src/evidence/
├── __init__.py
├── evidence_spec_registry.py   # EdgeEvidenceSpecRegistry
├── evidence_binder.py          # EvidenceBinder
├── evidence_accumulator.py     # EvidenceAccumulator
└── evidence_store.py           # EvidenceStore
```

| 모듈 | 책임 | 핵심 로직 |
|------|------|----------|
| **EdgeEvidenceSpecRegistry** | 관계 타입별 필요 feature, 점수 규칙, lag, decay, regime 정의 | `config/evidence_specs.yaml` |
| **EvidenceBinder** | Feature 값 읽어 `pro_score / con_score / total_score` 산출 + trace 생성 | FeatureStore 의존 |
| **EvidenceAccumulator** | 단발성 score를 누적/평활화 → edge confidence 업데이트 값 생성 | EMA/EWMA |
| **EvidenceStore** | Edge별 evidence 시계열 저장 (리플레이/설명에 필수) | Append-only |

### 3-2. Evidence Spec 예시

```yaml
# config/evidence_specs.yaml
edge_evidence:
  # 관계: 금리 ↑ → 성장주 ↓
  - edge_pattern:
      head_type: "Indicator"
      head_name_contains: ["금리", "interest_rate", "SOFR"]
      tail_type: "Asset"
      tail_name_contains: ["성장주", "growth_stock", "NASDAQ"]
      relation_type: "Affect"
      polarity: "-"
    
    evidence_features:
      - feature: "SOFR_ROC_30D"        # 30일 변화율
        direction: "positive"          # feature ↑ → pro
        weight: 0.4
      - feature: "NASDAQ_ROC_30D"      # 30일 변화율
        direction: "negative"          # feature ↓ → pro
        weight: 0.4
      - feature: "SOFR_NASDAQ_CORR_90D"
        direction: "negative"          # 음의 상관 → pro
        weight: 0.2
    
    thresholds:
      strong_pro: 0.7
      weak_pro: 0.3
      neutral: [-0.3, 0.3]
      weak_con: -0.3
      strong_con: -0.7
    
    lag_days: [0, 30, 60]  # 시차 효과 탐색
    regime_applicability:
      risk_on: 0.8
      risk_off: 1.2
```

### 3-3. 기존 모듈 수정

| 모듈 | 수정 내용 |
|------|----------|
| **EdgeWeightFusion** | `W = W_D × evidence_score × regime_applicability × decay` |
| **DomainDriftDetector** | "반대 증거 누적" vs "레짐 전환" 분리 → 오탐 감소 |
| **ConflictAnalyzer** | 텍스트 부호와 evidence 부호 충돌 시 즉시 뒤집지 말고 flag + 약화 |

### 3-4. 수용 기준

- [x] 어떤 결론이든 **숫자 근거(trace)**가 항상 따라붙음 ✅ EvidenceScore.trace 필드
- [x] Evidence가 약하면 결론도 자동으로 약해짐 (과장 방지) ✅ EdgeWeightFusion v2.0
- [x] Evidence trace가 ReasoningConclusion에 포함됨 ✅ EvidenceStore 구현

---

## 4단계: Regime Layer (국면) (3~4일)

> **목표:** 레짐 변화로 관계가 뒤집히는 금융 고유 실패 방지

### 4-1. 신규 모듈

```
src/regime/
├── __init__.py
├── regime_spec.py       # RegimeSpec: 레짐 정의
├── regime_detector.py   # RegimeDetector: 현재 레짐 산출
└── regime_store.py      # RegimeStore: 레짐 결과 저장
```

### 4-2. 레짐 정의 (규칙 기반 시작)

```yaml
# config/regimes.yaml
regimes:
  - name: risk_on
    conditions:
      - feature: VIX
        operator: "<"
        threshold: 20
      - feature: SPY_ROC_20D
        operator: ">"
        threshold: 0
    priority: 1
    
  - name: risk_off
    conditions:
      - feature: VIX
        operator: ">="
        threshold: 25
    priority: 2
    
  - name: inflation_up
    conditions:
      - feature: CPI_YOY
        operator: ">"
        threshold: 0.03
      - feature: CPI_YOY_DELTA_3M
        operator: ">"
        threshold: 0
    priority: 1
    
  - name: disinflation
    conditions:
      - feature: CPI_YOY_DELTA_3M
        operator: "<"
        threshold: 0
    priority: 2
```

### 4-3. 기존 모듈 수정

| 모듈 | 수정 내용 |
|------|----------|
| **EdgeWeightFusion** | `weight = base_conf × evidence_score × regime_applicability × decay` |
| **PathReasoningEngine** | 레짐 컨텍스트를 경로 해석에 반영 |
| **ConflictAnalyzer** | 충돌이 레짐 차이로 설명되면 drift 오판 방지 |

### 4-4. 수용 기준

- [x] 레짐 변경 시 동일 질문에 "관계/근거"가 합리적으로 변함 ✅ RegimeDetector + EdgeWeightFusion 연동
- [x] 충돌이 레짐 차이로 설명 가능하면 drift로 오판하지 않음 ✅ regime_applicability 반영

---

## 5단계: Incremental Update Orchestrator (3~4일)

> **목표:** "delta 수집"의 이점을 실제 운영 성능으로 연결

### 5-1. 신규 모듈

```
src/orchestration/
├── __init__.py
├── dependency_graph_manager.py  # DependencyGraphManager
├── cache_invalidator.py         # CacheInvalidator
└── incremental_orchestrator.py  # IncrementalUpdateOrchestrator
```

| 모듈 | 책임 |
|------|------|
| **DependencyGraphManager** | `series→feature→evidence→edge` 역인덱스 유지 |
| **CacheInvalidator** | 업데이트된 입력 따라가며 캐시/계산 무효화 |
| **IncrementalUpdateOrchestrator** | Ingestion 결과 → 영향 feature만 재계산 → 영향 evidence만 재계산 → 변동 큰 edge만 KG 갱신 |

### 5-2. 처리 흐름

```
[Delta Ingestion]
       ↓
[DependencyGraphManager: 영향 범위 산출]
       ↓
       ├─→ [FeatureBuilder: 해당 feature만 재계산]
       ↓
       ├─→ [EvidenceBinder: 해당 evidence만 재계산]
       ↓
       └─→ [DynamicDomainUpdate: 변동 edge만 갱신]
```

### 5-3. 수용 기준

- [x] 하루 업데이트로 전체 KG 재구축 없음 ✅ IncrementalUpdateOrchestrator 구현
- [x] 지표 수 증가 시 선형 확장 (부분 재계산) ✅ DependencyGraphManager 구현

---

## 6단계: Replay/Backtest (과거 재현) (4~5일)

> **목표:** 과거 특정 날짜로 돌아가 "그때 시스템이 뭐라고 했어야 하는지" 재현

### 6-1. 신규 모듈

```
src/replay/
├── __init__.py
├── snapshot_manager.py    # SnapshotManager
├── replay_runner.py       # ReplayRunner
└── metrics.py             # Metrics 계산
```

| 모듈 | 책임 |
|------|------|
| **SnapshotManager** | as_of 기준 스냅샷화 (KG edge 상태, feature/evidence 최신값, regime 상태) |
| **ReplayRunner** | 기간 순회하며 그날 delta만 반영 → 조언/결론 생성 → 실제 결과와 비교 |
| **Metrics** | calibration, risk_control, stability 3종 이상 계산 |

### 6-2. 메트릭 정의

| 메트릭 | 설명 |
|--------|------|
| **Calibration** | 시스템 자신감과 실제 성과의 정합성 |
| **Risk Control** | 경고가 DD/Vol 상승 전에 나왔는지 |
| **Stability** | 조건 동일할 때 결론 흔들림 여부 |

### 6-3. 수용 기준

- [x] 특정 날짜 지정 시 동일 결과 재현 (결정적) ✅ SnapshotManager 구현
- [x] 수정 데이터(as_of revision) 있어도 당시 스냅샷 기준 재현 ✅ ReplayRunner 구현

---

## 7단계: Scenario (Shock 주입) (3~4일)

> **목표:** "말"이 아니라 "정량 Shock"으로 시나리오 제어

### 7-1. 신규 모듈

```
src/scenario/
├── __init__.py
├── shock_spec_registry.py   # ShockSpecRegistry
└── scenario_simulator.py    # ScenarioSimulator
```

| 모듈 | 책임 |
|------|------|
| **ShockSpecRegistry** | Shock 입력 표준화 (예: 10Y +25bp, credit +50bp, liquidity -1σ) |
| **ScenarioSimulator** | Shock를 노드/feature에 주입 → 경로추론/부호전파로 영향 전파 |

### 7-2. 출력 형식

- **초기:** "방향(+/-) + 상대강도(약/중/강)" 중심 (수치 예측 과장 방지)
- **Break Conditions:** 어떤 지표가 특정 임계값 넘으면 시나리오 무효

### 7-3. 수용 기준

- [x] 동일 shock에 결과 일관적 ✅ ScenarioSimulator 구현
- [x] Break conditions 함께 출력 ✅ BreakConditionResult 구현

---

## 8단계: Learning/Policy 연결 강화 (2~3일)

> **목표:** Learning Layer를 "진짜로" 작동시키기

### 8-1. TrainingDatasetBuilder 확장

```python
# 필수 포함 정보
class ExtendedTrainingSample(BaseModel):
    # 기존
    sample_id: str
    task_type: TaskType
    
    # === 신규 ===
    evidence_trace: Dict[str, float]  # 당시 evidence
    regime_snapshot: Dict[str, Any]   # 당시 regime
    conclusion_label: str             # 당시 조언 결론
    outcome_metrics: Dict[str, float] # 이후 실제 결과
```

### 8-2. PolicyLearner 확장

탐색 후보 확대:

| 기존 | 확장 |
|------|------|
| EES weights | + evidence 가중치 |
| PCS weights | + regime 적용 강도 |
| Thresholds | + 결론 임계값 |

### 8-3. 수용 기준

- [x] Dataset에 evidence trace, regime 포함 ✅ ExtendedTrainingSample 구현
- [x] Policy 최적화에 확장 후보 포함 ✅ ExtendedPolicyLearner 구현

---

## 실행 순서 요약

```
┌─────────────────────────────────────────────────────────────────┐
│ 0단계: 스키마/계약 확정 (1~2일)                                    │
│   → relation_types.yaml 확장, Observation/Event 스키마 정의       │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 1단계: Delta Ingestion + Append-only 저장소 (3~5일)               │
│   → SourceRegistry, FetchStateStore, TimeSeriesRepository 등     │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2단계: Feature 계산(증분) + 의존성 인덱스 (2~3일)                  │
│   → FeatureSpecRegistry, FeatureDependencyIndex, FeatureBuilder │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3단계: Evidence Layer ⭐ (5~7일) - MVP 핵심                       │
│   → EdgeEvidenceSpecRegistry, EvidenceBinder, EvidenceStore     │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4단계: Regime Layer (3~4일) - 금융 필수                           │
│   → RegimeSpec, RegimeDetector, EdgeWeightFusion 수정           │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
╔═════════════════════════════════════════════════════════════════╗
║ 🎯 여기까지 = "조언자 MVP" (약 14~21일)                           ║
║    증거 기반 + 레짐 인지 + 설명 가능한 추론                        ║
╚════════════════════════┬════════════════════════════════════════╝
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5단계: Incremental Update Orchestrator (3~4일)                   │
│   → 운영비용 절감, 확장성 확보                                     │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6단계: Replay/Backtest (4~5일)                                   │
│   → "검증 가능한 시스템" 도달                                      │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7단계: Scenario (3~4일)                                          │
│   → "What-if" 분석 가능                                           │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8단계: Learning/Policy 연결 강화 (2~3일)                          │
│   → 지속적 개선 체계 완성                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 디렉토리 구조 변화 (To-Be)

```
src/
├── extraction/      # 기존 유지
├── validation/      # 기존 유지
├── domain/          # 기존 + 수정
├── personal/        # 기존 유지
├── reasoning/       # 기존 + 수정 (EdgeWeightFusion 등)
├── learning/        # 기존 + 확장
├── llm/             # 기존 유지
├── shared/          # 기존 + 확장 (schemas.py)
├── storage/         # 기존 + 확장
│   ├── timeseries_repository.py  # 신규
│   └── event_repository.py       # 신규
├── stores/          # 기존 유지
│
│ === 신규 디렉토리 ===
├── ingestion/       # 1단계: Delta Ingestion
├── features/        # 2단계: Feature 계산
├── evidence/        # 3단계: Evidence Layer
├── regime/          # 4단계: Regime Layer
├── orchestration/   # 5단계: Incremental Update
├── replay/          # 6단계: Replay/Backtest
└── scenario/        # 7단계: Scenario
```

---

## 위험 요소 및 대응

| 위험 | 확률 | 영향 | 대응 |
|------|------|------|------|
| Evidence spec 설계 실수 | 높음 | 높음 | 3단계에 충분한 시간 투자, 프로토타입 먼저 |
| 외부 데이터 소스 불안정 | 중간 | 중간 | 1단계에 robust한 재시도/폴백 설계 |
| 레짐 정의 불명확 | 중간 | 높음 | 규칙 기반으로 시작, 점진적 개선 |
| 성능 병목 (전체 재계산) | 중간 | 중간 | 5단계 증분 업데이트로 해결 |

---

## 진행 상태

### ✅ 전체 완료! (0-8단계)

| 단계 | 상태 | 완료일 |
|------|------|--------|
| 0단계: 스키마/계약 | ✅ 완료 | 2024-12-14 |
| 1단계: Delta Ingestion | ✅ 완료 | 2024-12-14 |
| 2단계: Feature 계산 | ✅ 완료 | 2024-12-14 |
| 3단계: Evidence Layer | ✅ 완료 | 2024-12-14 |
| 4단계: Regime Layer | ✅ 온료 | 2024-12-14 |
| 5단계: Incremental Orchestrator | ✅ 완료 | 2024-12-14 |
| 6단계: Replay/Backtest | ✅ 완료 | 2024-12-14 |
| 7단계: Scenario Simulation | ✅ 완료 | 2024-12-14 |
| 8단계: Extended Learning | ✅ 완료 | 2024-12-14 |

### 📋 생성된 파일 목록

**Config 파일:**
- `config/sources.yaml` - 데이터 소스 설정
- `config/features.yaml` - Feature 정의
- `config/evidence_specs.yaml` - Evidence 스펙
- `config/regimes.yaml` - 레짐 정의

**Core 모듈 (0-4단계):**
- `src/shared/schemas.py` - 핵심 스키마 정의
- `src/ingestion/` - Delta Ingestion 패키지
- `src/features/` - Feature 계산 패키지
- `src/evidence/` - Evidence Layer 패키지
- `src/regime/` - Regime Layer 패키지
- `src/storage/timeseries_repository.py`
- `src/storage/event_repository.py`

**Core 모듈 (5-8단계):**
- `src/orchestration/` - 증분 업데이트 오케스트레이션
  - `dependency_graph_manager.py`
  - `cache_invalidator.py`
  - `incremental_orchestrator.py`
- `src/replay/` - 과거 재현/백테스트
  - `snapshot_manager.py`
  - `replay_runner.py`
  - `metrics.py`
- `src/scenario/` - 시나리오 시뮬레이션
  - `shock_spec_registry.py`
  - `scenario_simulator.py`
- `src/learning/` - 확장 학습 모듈
  - `extended_models.py`
  - `extended_policy_learner.py`
  - `extended_dataset_builder.py`

**문서:**
- `docs/SCHEMA_CONTRACT.md` - 스키마 계약 문서

**테스트:**
- `tests/test_advisor_step0_4.py` - 0-4단계 통합 테스트 (18개 통과)
- `tests/test_advisor_step5_8.py` - 5-8단계 통합 테스트 (13개 통과)

## 다음 액션 (배포 및 운영)

1. **실 데이터 연동**: FRED, Yahoo Finance API 연동
2. **엔드투엔드 테스트**: 전체 파이프라인 통합 테스트
3. **UI/API**: 조언자 시스템 API 엔드포인트 구현
4. **모니터링**: 시스템 상태 대시보드

---

*Generated: 2024-12-14*
*Updated: 2024-12-14 (Step 0-8 전체 완료)*
*Version: 3.0*
