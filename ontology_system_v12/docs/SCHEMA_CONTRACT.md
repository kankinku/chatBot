# Schema Contract - 스키마 계약 문서

## 개요

이 문서는 Advisor System의 핵심 데이터 스키마에 대한 계약을 정의합니다.
모든 모듈은 이 스키마를 준수해야 하며, 스키마 변경 시 반드시 버전을 올려야 합니다.

**버전**: 1.0
**최종 수정**: 2024-12-14

---

## 1. 데이터 계층 정의

```
┌─────────────────────────────────────────────────────────────────┐
│                        Raw Data Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Observation  │  │    Event     │  │   Document   │          │
│  │  (시계열)    │  │   (이벤트)   │  │   (문서)     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼─────────────────┼─────────────────┼──────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Feature Layer                              │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                   FeatureValue                          │    │
│  │  (계산된 지표: ROC, ZScore, Spread, Correlation 등)     │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Evidence Layer                             │
│  ┌──────────────┐  ┌────────────────────┐  ┌───────────────┐   │
│  │EvidenceScore │  │AccumulatedEvidence │  │ RegimeResult  │   │
│  │  (개별 점수) │  │   (누적 점수)      │  │  (레짐 상태)  │   │
│  └──────────────┘  └────────────────────┘  └───────────────┘   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Knowledge Graph                             │
│  ┌──────────────┐  ┌────────────────────────────────────────┐  │
│  │   Entity     │  │            Edge (Relation)              │  │
│  │              │  │  - confidence = f(text, evidence, regime)│  │
│  └──────────────┘  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 핵심 스키마 정의

### 2.1 Observation (시계열 관측값)

**위치**: `src/shared/schemas.py`

| 필드 | 타입 | 설명 | 필수 |
|------|------|------|------|
| `observation_id` | str | 고유 ID | ✅ |
| `series_id` | str | 시계열 ID (예: SOFR, VIX) | ✅ |
| `timestamp` | datetime | 관측 시점 | ✅ |
| `as_of` | datetime | 수집/수정 시점 (버전 관리) | ✅ |
| `value` | float | 관측값 | ✅ |
| `unit` | str | 단위 | ❌ |
| `source_id` | str | 데이터 소스 ID | ✅ |
| `is_revision` | bool | 수정 데이터 여부 | ✅ |
| `quality_flag` | str | 품질 플래그 | ✅ |

**Primary Key**: `(series_id, timestamp, as_of)`

**불변 규칙**:
- 한번 저장된 Observation은 수정/삭제 불가 (Append-only)
- 수정이 필요하면 새 `as_of`로 새 레코드 추가
- `is_revision=true`로 표시

---

### 2.2 Event (비정형 이벤트)

| 필드 | 타입 | 설명 | 필수 |
|------|------|------|------|
| `event_id` | str | 고유 ID | ✅ |
| `source_id` | str | 데이터 소스 ID | ✅ |
| `external_id` | str | 원본 시스템 ID | ❌ |
| `content_hash` | str | 콘텐츠 해시 | ✅ |
| `occurred_at` | datetime | 이벤트 발생 시점 | ✅ |
| `collected_at` | datetime | 수집 시점 | ✅ |
| `title` | str | 제목 | ✅ |
| `content` | str | 내용 | ❌ |
| `event_type` | str | 이벤트 유형 | ✅ |
| `entities` | List[str] | 관련 엔티티 | ❌ |
| `sentiment` | float | 감성 점수 (-1~1) | ❌ |

**Primary Key**: `(source_id, external_id)` 또는 `content_hash`

---

### 2.3 FeatureValue (계산된 지표)

| 필드 | 타입 | 설명 | 필수 |
|------|------|------|------|
| `feature_value_id` | str | 고유 ID | ✅ |
| `feature_id` | str | Feature 정의 ID | ✅ |
| `timestamp` | datetime | 계산 기준 시점 | ✅ |
| `as_of` | datetime | 계산 시점 | ✅ |
| `value` | float | 계산된 값 | ✅ |

**Primary Key**: `(feature_id, timestamp, as_of)`

---

### 2.4 EvidenceScore (Edge 검증 점수)

| 필드 | 타입 | 설명 | 필수 |
|------|------|------|------|
| `evidence_id` | str | 고유 ID | ✅ |
| `edge_id` | str | 대상 Edge ID | ✅ |
| `head_id` | str | Head 엔티티 ID | ✅ |
| `tail_id` | str | Tail 엔티티 ID | ✅ |
| `relation_type` | str | 관계 타입 | ✅ |
| `pro_score` | float | 지지 점수 (0~1) | ✅ |
| `con_score` | float | 반박 점수 (0~1) | ✅ |
| `total_score` | float | 종합 점수 (-1~1) | ✅ |
| `timestamp` | datetime | 점수 계산 시점 | ✅ |
| `regime` | RegimeType | 레짐 컨텍스트 | ❌ |
| `trace` | List[Dict] | 추적 정보 | ✅ |
| `confidence` | float | 신뢰도 (0~1) | ✅ |

**중요**: `trace` 필드는 항상 채워져야 함 (설명 가능성 필수)

---

### 2.5 RegimeDetectionResult (레짐 탐지 결과)

| 필드 | 타입 | 설명 | 필수 |
|------|------|------|------|
| `detection_id` | str | 고유 ID | ✅ |
| `timestamp` | datetime | 탐지 시점 | ✅ |
| `detected_regimes` | Dict[RegimeType, float] | 레짐별 확률 | ✅ |
| `primary_regime` | RegimeType | 주요 레짐 | ❌ |
| `primary_probability` | float | 주요 레짐 확률 | ✅ |
| `uncertainty` | float | 불확실성 (0~1) | ✅ |
| `feature_snapshot` | Dict[str, float] | 사용된 Feature 값 | ✅ |

---

## 3. 관계 타입 메타데이터

**위치**: `config/relation_types.yaml`

모든 관계 타입에는 다음 메타데이터가 필수:

| 필드 | 설명 | 기본값 |
|------|------|--------|
| `lag_days` | 지연 효과 후보 (일) | [0, 30, 90] |
| `decay_halflife_days` | 신뢰도 반감기 (일) | 90 |
| `regime_applicability` | 레짐별 적용 강도 | 모두 1.0 |

### 레짐 타입 목록

| RegimeType | 설명 |
|------------|------|
| `risk_on` | 위험 선호 |
| `risk_off` | 위험 회피 |
| `inflation_up` | 인플레 상승 |
| `disinflation` | 인플레 하락 |
| `growth_up` | 성장 상승 |
| `growth_down` | 성장 하락 |
| `liquidity_abundant` | 유동성 풍부 |
| `liquidity_tight` | 유동성 긴축 |

---

## 4. 의존성 규칙

### 4.1 데이터 흐름

```
Source → Observation/Event → Feature → Evidence → Edge Confidence
                                ↓
                             Regime
```

### 4.2 증분 업데이트 규칙

1. **Observation 업데이트 시**:
   - 해당 series를 입력으로 하는 Feature만 재계산
   - 해당 Feature를 사용하는 Evidence만 재계산
   - 해당 Evidence가 적용되는 Edge만 갱신

2. **Event 업데이트 시**:
   - 관련 엔티티의 Evidence 재계산
   - 영향 범위 Edge 갱신

3. **Regime 변경 시**:
   - 모든 Evidence의 `regime_adjustment` 재계산
   - 영향받는 Edge 갱신

---

## 5. ID 규약

| 스키마 | ID Prefix | 예시 |
|--------|-----------|------|
| Observation | OBS | OBS_a1b2c3d4e5f6 |
| Event | EVT | EVT_a1b2c3d4e5f6 |
| FeatureValue | FV | FV_a1b2c3d4e5f6 |
| EvidenceScore | EV | EV_a1b2c3d4e5f6 |
| EvidenceSpec | ESPEC | ESPEC_a1b2c3d4e5f6 |
| RegimeDetection | RD | RD_a1b2c3d4e5f6 |
| RegimeSpec | REG | REG_a1b2c3d4e5f6 |

---

## 6. 버전 관리 규칙

### 6.1 스키마 버전

- **Major**: 호환성 깨지는 변경 (필드 삭제, 타입 변경)
- **Minor**: 후방 호환 변경 (필드 추가, 옵션 변경)

### 6.2 데이터 버전 (as_of)

- 모든 시계열 데이터는 `as_of` 필드로 버전 관리
- 과거 시점 재현 시 `as_of <= target_date` 조건으로 조회
- 수정 데이터 식별 시 `is_revision=true` 사용

---

## 7. 유효성 검사 규칙

### 7.1 Observation

- `value`는 NaN/Inf 불가
- `timestamp` < `as_of`
- `series_id`는 등록된 시계열이어야 함

### 7.2 EvidenceScore

- `0 <= pro_score <= 1`
- `0 <= con_score <= 1`
- `-1 <= total_score <= 1`
- `trace`는 비어있으면 안 됨

### 7.3 RegimeDetectionResult

- `sum(detected_regimes.values()) <= len(RegimeType)` (정규화 불필요, 중복 가능)
- `0 <= uncertainty <= 1`

---

## 8. 저장소별 스키마 매핑

| 스키마 | 저장소 | 테이블/컬렉션 |
|--------|--------|---------------|
| Observation | TimeSeriesRepository | observations |
| Event | EventRepository | events |
| FeatureValue | FeatureStore | feature_values |
| EvidenceScore | EvidenceStore | evidence_scores |
| RegimeDetectionResult | RegimeStore | regime_detections |
| Edge | GraphRepository | relations |

---

## 부록: 샘플 데이터

### Observation 예시

```json
{
  "observation_id": "OBS_a1b2c3d4e5f6",
  "series_id": "SOFR",
  "timestamp": "2024-12-13T00:00:00Z",
  "as_of": "2024-12-14T06:00:00Z",
  "value": 4.33,
  "unit": "%",
  "source_id": "fred",
  "is_revision": false,
  "quality_flag": "ok"
}
```

### EvidenceScore 예시

```json
{
  "evidence_id": "EV_x1y2z3",
  "edge_id": "EDGE_abc123",
  "head_id": "interest_rate",
  "tail_id": "growth_stock",
  "relation_type": "Affect",
  "pro_score": 0.72,
  "con_score": 0.15,
  "total_score": 0.57,
  "timestamp": "2024-12-14T00:00:00Z",
  "regime": "risk_off",
  "regime_adjustment": 1.2,
  "trace": [
    {"feature": "SOFR_ROC_30D", "value": 0.12, "contribution": 0.35},
    {"feature": "NASDAQ_ROC_30D", "value": -0.08, "contribution": 0.37}
  ],
  "best_lag_days": 30,
  "confidence": 0.75
}
```

---

*이 문서는 시스템의 Single Source of Truth입니다.*
*스키마 변경 시 반드시 이 문서를 먼저 업데이트하세요.*
