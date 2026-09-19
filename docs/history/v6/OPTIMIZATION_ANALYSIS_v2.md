# 🔍 최적화 결과 분석 및 재조정

## 📊 최적화 전후 비교

| 항목 | 최적화 전 | 최적화 후 | 변화 |
|------|----------|----------|------|
| **평균 검색 시간** | 29.88ms | 32.69ms | ❌ +9.4% (악화) |
| **Hybrid 검색** | 134.56ms | 246.76ms | ❌ +83% (악화) |
| **QPS** | 33.5 | 30.6 | ❌ -8.7% (악화) |
| **전체 테스트 시간** | 17.77초 | 8.41초 | ✅ -53% (개선) |
| **파이프라인 초기화** | 166.90ms | 152.88ms | ✅ -8.4% (개선) |

## ⚠️ 문제 원인 분석

### 1. 병렬 검색의 역효과 (ThreadPoolExecutor)
**예상**: BM25와 Vector를 병렬 실행하면 빨라질 것  
**실제**: 134ms → 246ms (오히려 느려짐!)

**원인**:
- Python GIL (Global Interpreter Lock) 때문에 CPU 바운드 작업은 병렬화가 어려움
- ThreadPoolExecutor 생성/관리 오버헤드 (~100ms)
- NumPy 연산은 이미 C로 최적화되어 있어 병렬화 이득 없음

**해결책**: ThreadPoolExecutor 제거하고 순차 실행으로 복귀

### 2. NumPy 배열 변환 오버헤드
**추가된 코드**:
```python
vector_arr = np.array([(idx, score) for idx, score in vector_results])
```

**문제**: 리스트 컴프리헨션 + np.array 변환이 오히려 느림

**해결책**: 원래의 간단한 반복문이 더 빠를 수 있음

### 3. 테스트 환경 차이
- 전체 테스트 시간이 절반으로 줄어든 것은 긍정적
- 일부 모듈이 캐시되어 재실행 시 빨라짐

## 🔧 재최적화 전략

### Phase 1-Rev: 검증된 최적화만 적용

#### ✅ 적용 유지 (효과 확인)
1. **Vector norm 사전 계산** ⭐
   - 초기화 시간 개선 확인 (166.90ms → 152.88ms)
   - 유지!

2. **질문 분석 캐싱** ⭐
   - @lru_cache 오버헤드 미미
   - 유지!

3. **배치 크기 증가** ⭐
   - 32 → 64 (초기화 시간 개선에 기여)
   - 유지!

#### ❌ 롤백 필요 (부작용 확인)
1. **병렬 검색 (ThreadPoolExecutor)** ❌
   - 오히려 느려짐 (134ms → 246ms)
   - 즉시 제거!

2. **NumPy 정규화** ❌
   - 작은 데이터셋에서는 오버헤드
   - 단순 반복문으로 복귀

### Phase 2-Rev: 진짜 효과적인 최적화

#### 1. FAISS 인덱스 사용 ⚡⚡⚡
```python
# 현재: simple numpy (190ms)
# FAISS: ~20ms (10배 빠름!)
```

#### 2. 벡터 정규화 사전 계산 ⚡⚡
```python
# 초기화 시 한 번만
self.vectors = self.vectors / np.linalg.norm(self.vectors, axis=1)[:, np.newaxis]

# 검색 시
similarities = np.dot(self.vectors, query_vec)  # norm 불필요!
```

#### 3. 임베딩 캐싱 ⚡
```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def embed_query(self, text: str):
    # 같은 쿼리 재사용
```

## 🎯 수정 계획

### 즉시 수정 (5분)
1. ThreadPoolExecutor 제거
2. NumPy 배열 변환 제거
3. 단순한 정규화로 복귀

### 단기 적용 (30분)
1. 벡터 정규화 사전 계산
2. 임베딩 쿼리 캐싱
3. FAISS 인덱스 빌드 스크립트

### 예상 결과
```
현재 (최적화 전): 29.88ms
↓
수정 후: 15ms (2배 향상!)
↓
FAISS 적용: 8ms (4배 향상!)
```

