# 벤치마크 가이드

qa.json을 사용한 RAG 시스템 성능 평가 가이드입니다.

## 📋 qa.json 구조

```json
[
  {
    "id": 1,
    "question": "질문 내용",
    "answer": "정답 내용",
    "accepted_keywords": ["키워드1", "키워드2", "키워드3"]
  }
]
```

- 총 30개의 질문-답변 쌍
- 정수장 AI 플랫폼 관련 질문
- 각 질문마다 정답과 핵심 키워드 포함

## 🚀 실행 순서

### 1단계: 샘플 테스트 (빠른 확인)

```bash
python test_sample.py
```

- 처음 3개 질문만 테스트
- 각 질문에 대한 상세 평가 점수 출력
- 전체 벤치마크 전 시스템 동작 확인용

```bash
python test_sample.py 5
```

- 5개 질문으로 테스트 (개수 조정 가능)

### 2단계: 전체 벤치마크

```bash
python benchmark.py
```

- 30개 전체 질문 테스트
- 결과를 `benchmark_results.json`에 저장
- 소요 시간: 약 5-10분 (LLM 속도에 따라 다름)

### 3단계: 결과 확인

```bash
python view_benchmark_results.py
```

대화형 메뉴:
1. 전체 결과 보기
2. 낮은 점수 항목만 보기 (50% 미만)
3. 실패한 항목만 보기
4. 종료

## 📊 평가 지표 상세

### 1. 기본 Score (v5) - 실무 중심 점수

**계산 방식:**
```
기본 Score = (키워드 정확도 × 0.35) + 
             (토큰 F1 × 0.25) + 
             (숫자 정확도 × 0.20) + 
             (ROUGE-L × 0.15) + 
             (BLEU-2 × 0.05)
```

**특징:**
- 실무에서 중요한 지표들의 가중 평균
- 키워드와 숫자 정확도에 높은 가중치
- 정수장 도메인 특성 반영

**해석:**
- 80% 이상: 우수
- 50-80%: 보통
- 50% 미만: 개선 필요

### 2. 도메인 특화 종합 점수

**계산 방식:**
```
도메인 특화 = (숫자 정확도 × 0.5) + 
              (단위 정확도 × 0.3) + 
              (키워드 정확도 × 0.2)
```

**특징:**
- 정수장 도메인에 특화된 평가
- 숫자, 단위, 전문용어 중심
- 기술 문서 평가에 최적화

### 3. RAG 핵심 지표

#### 3.1 Faithfulness (충실성)

**계산 방식:**
```
Faithfulness = 답변 토큰 중 컨텍스트에 포함된 비율
```

**특징:**
- 답변이 검색된 문서에 충실한지 평가
- 환각(Hallucination) 방지
- 높을수록 근거 기반 답변

#### 3.2 Answer Correctness (정확도)

**계산 방식:**
```
Answer Correctness = (Token F1 + ROUGE-L) / 2
```

**특징:**
- 답변이 정답과 얼마나 일치하는지
- 의미적 정확도 평가
- 토큰과 구조 모두 고려

#### 3.3 Context Precision (컨텍스트 정밀도)

**계산 방식:**
```
Context Precision = 관련있는 컨텍스트 수 / 전체 검색된 컨텍스트 수
```

**특징:**
- 검색 품질 평가
- 불필요한 문서 검색 방지
- 높을수록 정밀한 검색

### 4. 키워드 정확도 (Keyword Accuracy)

**계산 방식:**
```
키워드 정확도 = 매칭된 키워드 수 / 전체 키워드 수
```

**특징:**
- 도메인 전문 용어 매칭 평가
- 대소문자 구분 없음
- 부분 문자열 매칭

**예시:**
```
키워드: ["한국수자원공사", "AI 자율 운영"]
답변: "한국수자원공사에서 AI 자율 운영 시스템을..."
결과: 2/2 = 100%
```

### 5. 토큰 오버랩 (Token Overlap)

**계산 방식:**
```
Precision = 공통 토큰 수 / 생성된 답변 토큰 수
Recall = 공통 토큰 수 / 정답 토큰 수
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**특징:**
- 단어 수준 일치도 평가
- 정확도와 재현율 균형 고려
- F1 점수를 주요 지표로 사용

### 6. 숫자 & 단위 정확도

#### 6.1 숫자 정확도 (Numeric Accuracy)

**계산 방식:**
```
숫자 정확도 = 매칭된 숫자 수 / 정답의 숫자 수
```

**특징:**
- 날짜, 버전, 수량, IP 주소 등 추출
- 정수장 운영 정보에 중요
- 예: "2025.02.17", "10.103.11.112", "80%"

#### 6.2 단위 정확도 (Unit Accuracy)

**계산 방식:**
```
단위 정확도 = 매칭된 단위 수 / 정답의 단위 수
```

**특징:**
- kWh, kg, %, ℃ 등 단위 매칭
- 정수장 운영 지표에 중요
- 예: "전력 100 kWh", "탄소 50 kg"

### 7. BLEU-2 (Bi-gram BLEU)

**특징:**
- 기계번역 평가에서 유래
- 연속된 2개 단어(bi-gram) 일치도
- 문장 유창성 평가

### 8. ROUGE-L (Longest Common Subsequence)

**특징:**
- 요약 평가에서 유래
- 최장 공통 부분수열 기반
- 순서를 고려한 유사도

## 📈 결과 해석

### 출력 예시

```
================================================================================
📊 평가 결과 요약:

1️⃣  기본 Score (v5):          90.5%
2️⃣  도메인 특화 종합:          98.0%
    - 숫자 정확도:            26.7%
    - 단위 정확도:             0.0%
3️⃣  RAG 핵심 지표:
    - Faithfulness:           56.4%
    - Answer Correctness:     75.7%
    - Context Precision:       2.9%
4️⃣  학술 표준:
    - Token F1:               40.5%
    - ROUGE-L:                38.0%

⏱️  평균 응답 시간:          9.94초
================================================================================
```

### 점수별 분류

```
높음 (80% 이상)  → 답변 품질 우수
중간 (50-80%)    → 답변 품질 보통
낮음 (50% 미만)  → 개선 필요
```

### 평가 지표별 의미

1. **기본 Score가 높으면:**
   - 실무에서 바로 사용 가능한 품질
   - 키워드, 숫자, 텍스트 유사도 모두 우수

2. **도메인 특화 점수가 높으면:**
   - 정수장 전문 지식 잘 반영
   - 숫자와 단위가 정확
   - 기술 문서로 활용 가능

3. **RAG 지표가 높으면:**
   - Faithfulness 높음: 환각 없이 문서 기반 답변
   - Answer Correctness 높음: 정답과 높은 일치도
   - Context Precision 높음: 정밀한 문서 검색

4. **학술 표준이 높으면:**
   - 논문 발표 가능 수준
   - 객관적 성능 검증
   - 다른 시스템과 비교 가능

### 점수가 낮은 경우 원인 분석

1. **키워드 정확도가 낮은 경우:**
   - PDF 문서에 관련 정보가 없음
   - 검색(Retrieval) 성능 문제
   - 청크 크기 조정 필요

2. **토큰 F1이 낮은 경우:**
   - LLM 답변 생성 품질 문제
   - 프롬프트 개선 필요
   - 다른 LLM 모델 시도

3. **숫자 정확도가 낮은 경우:**
   - 숫자 정보 추출 실패
   - PDF OCR 품질 문제
   - 정밀한 정보 검색 필요

## 🔧 성능 개선 방법

### 검색 성능 개선
```python
# pdf_to_vectordb.py의 청크 크기 조정
def split_text(text, chunk_size=500):  # 기본 500
    # → 300으로 줄이면: 더 정밀한 검색
    # → 1000으로 늘리면: 더 많은 컨텍스트
```

### 검색 개수 조정
```python
# rag_query.py에서 top_k 조정
results = self.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=top_k  # 기본 3, 5~10으로 증가 시도
)
```

### LLM 프롬프트 개선
```python
# rag_query.py의 프롬프트 수정
prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.
정확한 숫자, 날짜, 고유명사를 포함하여 답변하세요.

참고 문서:
{context}

질문: {query}

답변:"""
```

## 📁 결과 파일 구조

### benchmark_results.json

```json
{
  "summary": {
    "timestamp": "2025-10-12T23:45:00",
    "total_questions": 30,
    "successful_answers": 28,
    "failed_answers": 2,
    "success_rate": 93.3,
    "total_time": 456.78,
    "average_time_per_question": 15.23,
    "average_scores": {
      "composite_score": 0.72,
      "keyword_accuracy": 0.68,
      "token_f1": 0.54,
      "numeric_accuracy": 0.81,
      "bleu_2": 0.32,
      "rouge_l": 0.45
    }
  },
  "results": [
    {
      "id": 1,
      "question": "...",
      "expected_answer": "...",
      "generated_answer": "...",
      "keywords": [...],
      "elapsed_time": 12.34,
      "retrieved_sources": ["파일1.pdf", "파일2.pdf"],
      "success": true,
      "evaluation": {
        "composite_score": 0.85,
        "keyword_accuracy": 0.9,
        "keyword_matched": ["키워드1", "키워드2"],
        "token_f1": 0.67,
        "token_precision": 0.72,
        "token_recall": 0.63,
        "numeric_accuracy": 1.0,
        "numeric_matched": ["2025", "17"],
        "bleu_2": 0.45,
        "rouge_l": 0.58,
        "exact_match": 0.0,
        "contains_match": 1.0
      }
    }
  ]
}
```

## 💡 팁

### 1. 단계적 테스트
```bash
# 1단계: 샘플로 빠른 확인
python test_sample.py

# 2단계: 문제 없으면 전체 실행
python benchmark.py

# 3단계: 결과 분석
python view_benchmark_results.py
```

### 2. 개선 사이클
1. 벤치마크 실행
2. 낮은 점수 항목 분석
3. 파라미터 조정 (청크 크기, top_k 등)
4. 재실행 및 비교

### 3. 결과 비교
```bash
# 버전별로 결과 저장
mv benchmark_results.json benchmark_v1.json
# 파라미터 조정 후 재실행
python benchmark.py
# 두 결과 비교
```

## ⚠️ 주의사항

1. **Ollama 실행 확인**
   - 벤치마크 전에 `ollama serve`가 실행 중인지 확인
   - 모델 `llama3.1:8b-instruct-q4_K_M`이 설치되어 있는지 확인

2. **시간 소요**
   - 전체 벤치마크는 5-10분 소요
   - 느린 경우 LLM 응답 속도 확인

3. **재현성**
   - LLM은 비결정적이므로 실행마다 결과가 약간 다를 수 있음
   - 여러 번 실행하여 평균을 확인하는 것이 좋음

4. **메모리 사용**
   - 30개 질문 처리 시 메모리 사용량 증가
   - 문제 발생 시 청크 수 줄이기

