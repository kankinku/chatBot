# 평가 시스템 완벽 가이드

RAG 챗봇의 모든 평가 지표를 이해하고 활용하는 방법

---

## 📋 목차

1. [평가 시스템 개요](#평가-시스템-개요)
2. [4가지 평가 체계](#4가지-평가-체계)
3. [통합 평가 모듈 사용법](#통합-평가-모듈-사용법)
4. [각 지표의 의미와 활용](#각-지표의-의미와-활용)
5. [실전 예시](#실전-예시)

---

## 평가 시스템 개요

이 프로젝트는 **4가지 독립적인 평가 체계**를 제공합니다:

| 평가 체계 | 목적 | 주요 지표 | 활용 분야 |
|---------|------|----------|----------|
| **1. 기본 Score (v5)** | 도메인 실무 평가 | 가중치 기반 종합 점수 | 실무 성능 측정 |
| **2. 도메인 특화** | 정수장 특화 평가 | 숫자/단위 정확도 | 기술 정보 정확도 |
| **3. RAG 핵심 3대** | RAG 시스템 평가 | Faithfulness, Correctness, Precision | 논문/학술 연구 |
| **4. 학술 표준** | 범용 NLP 평가 | F1, ROUGE, BLEU, EM | 타 시스템 비교 |

---

## 4가지 평가 체계

### 1️⃣ 기본 Score (v5 방식) - 도메인 가중치 평가

**파일**: `scripts/run_qa_benchmark.py` (46-127줄)

**특징**: 
- 숫자, 단위, 키워드에 차등 가중치 적용
- 실무에서 중요한 정보를 강조
- v5에서 검증된 평가 방식

**가중치**:
- 숫자: **1.5** (가장 중요)
- 단위: **1.3**
- 키워드: **1.0**

**예시**:
```python
score = benchmark.score_answer(
    prediction="수질 기준은 25℃ 입니다.",
    gold_answer="수질 기준은 25℃입니다.",
    keywords=["수질", "기준", "25", "℃"]
)
# score: 1.0 (완벽 일치)
```

**언제 사용**:
- ✅ 실무 성능 측정할 때
- ✅ v5와 성능 비교할 때
- ✅ 도메인 특화 성능을 강조할 때

---

### 2️⃣ 도메인 특화 평가 - 정수장 특화

**파일**: `scripts/enhanced_scoring.py`

**특징**:
- 숫자/단위 정확도 별도 측정
- 세부 분석 정보 제공
- 도메인 동의어 처리 (mg/L ↔ ppm)

**주요 메서드**:
```python
from scripts.enhanced_scoring import DomainSpecificScoring

scorer = DomainSpecificScoring()

# 1. 종합 평가 (v5 스타일)
result = scorer.score_answer_v5_style(pred, gold, keywords)
# {
#     'total_score': 0.95,
#     'numeric_score': 1.0,
#     'unit_score': 1.0,
#     'keyword_score': 0.85,
#     'details': {...}
# }

# 2. 숫자만 평가
numeric_acc = scorer.score_numeric_accuracy(pred, gold)
# 0.0 ~ 1.0

# 3. 단위만 평가
unit_acc = scorer.score_unit_accuracy(pred, gold)
# 0.0 ~ 1.0
```

**언제 사용**:
- ✅ 숫자 정확도를 상세히 분석할 때
- ✅ 단위 변환 정확도를 확인할 때
- ✅ 도메인 특화 성능을 강조할 때

---

### 3️⃣ RAG 핵심 3대 지표 - 학술 연구용

**파일**: `scripts/rag_core_metrics.py`

**특징**:
- RAGAs Framework 기반 (Es et al., 2023)
- 논문 인용 가능한 표준 지표
- RAG 시스템의 핵심 성능 측정

**3대 지표**:

#### 📌 Faithfulness (충실성) - 환각 방지
```python
from scripts.rag_core_metrics import RAGCoreMetrics

faith = RAGCoreMetrics.faithfulness(answer, contexts)
# {
#     'score': 0.85,
#     'supported_claims': 17,
#     'total_claims': 20,
#     'support_ratio': 0.85
# }
```

**평가 질문**: "답변이 자료 밖의 거짓말을 했나?"
- 높을수록 좋음 (환각 없음)
- 논문 핵심 지표 1순위

#### 📌 Answer Correctness (답변 정확도) - 사실적 일치
```python
correctness = RAGCoreMetrics.answer_correctness(answer, ground_truth)
# {
#     'score': 0.92,
#     'semantic_similarity': 0.88,
#     'factual_correctness': 0.95
# }
```

**평가 질문**: "답변이 정답과 사실상 동일한가?"
- 의미 유사도(40%) + 사실 정확도(60%)
- 숫자/단위 일치를 강조

#### 📌 Context Precision (문맥 정밀도) - 검색 효율성
```python
precision = RAGCoreMetrics.context_precision(
    question, contexts, answer, ground_truth
)
# {
#     'score': 0.75,
#     'relevant_contexts': 3,
#     'total_contexts': 4,
#     'precision': 0.75
# }
```

**평가 질문**: "엉뚱한 자료를 가져와서 헷갈리지 않았나?"
- 검색된 자료의 효율성 측정
- 불필요한 자료 비율 확인

**종합 평가**:
```python
results = RAGCoreMetrics.evaluate_all(
    question, answer, ground_truth, contexts
)
# {
#     'faithfulness': {...},
#     'answer_correctness': {...},
#     'context_precision': {...},
#     'overall_score': 0.84  # 가중 평균
# }
```

**가중치**: Faithfulness(40%) + Correctness(40%) + Precision(20%)

**언제 사용**:
- ✅ 논문 작성할 때
- ✅ RAG 시스템을 평가할 때
- ✅ 환각(hallucination) 문제를 분석할 때

**논문 인용 예시**:
```
본 연구의 RAG 챗봇 시스템을 20개 질문으로 평가한 결과,
Faithfulness 85.3%, Answer Correctness 92.1%, Context Precision 75.4%를 
달성하였다 (Es et al., 2023).

참고문헌:
Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023).
RAGAS: Automated Evaluation of Retrieval Augmented Generation.
arXiv preprint arXiv:2309.15217.
```

---

### 4️⃣ 학술 표준 지표 - 범용 NLP 평가

**파일**: `scripts/academic_metrics.py`

**특징**:
- SQuAD, ROUGE, BLEU 등 표준 지표
- 다른 시스템과 비교 가능
- 논문 인용 가능

**4가지 지표**:

#### 📌 Exact Match (완전 일치)
```python
from scripts.academic_metrics import AcademicMetrics

em = AcademicMetrics.exact_match(pred, gold)
# 1.0 (일치) or 0.0 (불일치)
```

**출처**: Rajpurkar et al. (2016), SQuAD, EMNLP 2016

#### 📌 Token F1 Score
```python
f1_result = AcademicMetrics.token_f1_score(pred, gold)
# {
#     'precision': 0.92,
#     'recall': 0.88,
#     'f1': 0.90
# }
```

**출처**: Rajpurkar et al. (2016), SQuAD, EMNLP 2016
- SQuAD의 주요 평가 지표

#### 📌 ROUGE-L (요약 평가)
```python
rouge = AcademicMetrics.rouge_l(pred, gold)
# {
#     'precision': 0.85,
#     'recall': 0.82,
#     'f1': 0.83
# }
```

**출처**: Lin (2004), ROUGE, ACL Workshop 2004
- 최장 공통 부분수열(LCS) 기반
- 순서를 고려한 유사도

#### 📌 BLEU (기계번역 표준)
```python
bleu1 = AcademicMetrics.bleu_n(pred, gold, n=1)  # BLEU-1
bleu2 = AcademicMetrics.bleu_n(pred, gold, n=2)  # BLEU-2
```

**출처**: Papineni et al. (2002), BLEU, ACL 2002
- n-gram 정밀도 기반

**종합 평가**:
```python
all_metrics = AcademicMetrics.evaluate_all(pred, gold)
# {
#     'exact_match': 0.0,
#     'token_f1': {'f1': 0.90, ...},
#     'rouge_l': {'f1': 0.83, ...},
#     'bleu_1': 0.88,
#     'bleu_2': 0.75
# }
```

**언제 사용**:
- ✅ 타 시스템과 성능 비교할 때
- ✅ 국제 학회 논문 작성할 때
- ✅ 표준 벤치마크와 비교할 때

---

## 통합 평가 모듈 사용법

**파일**: `scripts/unified_evaluation.py` (새로 생성)

**특징**:
- 4가지 평가 체계를 한 번에 실행
- 재사용 가능한 통합 인터페이스
- 다른 프로젝트에서도 사용 가능

### 기본 사용법

#### 1. 단일 평가

```python
from scripts.unified_evaluation import UnifiedEvaluator

evaluator = UnifiedEvaluator()

results = evaluator.evaluate_all(
    question="AI 플랫폼의 기본 관리자 아이디는?",
    prediction="기본 관리자 아이디는 KWATER입니다.",
    ground_truth="기본 관리자 아이디는 KWATER입니다.",
    contexts=["관리자 계정: KWATER", "시스템 접속 정보..."],
    keywords=["KWATER", "관리자", "아이디"]
)

# 결과 출력
evaluator.print_results(results)
```

#### 2. 배치 평가 (여러 질문 한 번에)

```python
qa_pairs = [
    {
        'question': '질문 1',
        'prediction': '답변 1',
        'ground_truth': '정답 1',
        'contexts': [...],
        'keywords': [...]
    },
    {
        'question': '질문 2',
        'prediction': '답변 2',
        'ground_truth': '정답 2',
        'contexts': [...],
        'keywords': [...]
    }
]

batch_results = evaluator.evaluate_batch(qa_pairs)
evaluator.print_results(batch_results)
```

#### 3. 결과 구조

```python
{
    'basic_score': {           # 기본 Score (v5 방식)
        'score': 0.94,
        'numeric_hit': 1.0,
        'unit_hit': 1.0,
        'keyword_hit': 0.85
    },
    'domain_specific': {       # 도메인 특화
        'total_score': 0.95,
        'numeric_score': 1.0,
        'unit_score': 1.0,
        'keyword_score': 0.85
    },
    'rag_metrics': {          # RAG 핵심 3대
        'faithfulness': {'score': 0.85, ...},
        'answer_correctness': {'score': 0.92, ...},
        'context_precision': {'score': 0.75, ...},
        'overall_score': 0.84
    },
    'academic_metrics': {      # 학술 표준
        'exact_match': 0.0,
        'token_f1': {'f1': 0.90, ...},
        'rouge_l': {'f1': 0.83, ...},
        'bleu_1': 0.88,
        'bleu_2': 0.75
    },
    'summary': {              # 주요 점수 요약
        'basic_v5_score': 0.94,
        'domain_total_score': 0.95,
        'numeric_accuracy': 1.0,
        'unit_accuracy': 1.0,
        'faithfulness': 0.85,
        'answer_correctness': 0.92,
        'context_precision': 0.75,
        'token_f1': 0.90,
        'rouge_l': 0.83,
        'bleu_2': 0.75
    }
}
```

---

## 각 지표의 의미와 활용

### 점수 해석 가이드

| 점수 범위 | 평가 | 의미 | 조치 사항 |
|----------|------|------|----------|
| **90~100%** | 우수 ⭐⭐⭐ | 실무 활용 가능 | 유지 |
| **70~90%** | 양호 ⭐⭐ | 준수한 성능 | 세부 개선 |
| **50~70%** | 보통 ⭐ | 개선 필요 | 검색/생성 점검 |
| **50% 미만** | 부족 ⚠️ | 시스템 점검 필요 | 전면 재검토 |

### 지표별 개선 방법

#### 기본 Score (v5) / 도메인 특화 점수가 낮을 때
- **숫자 정확도 ↓**: 
  - BM25 가중치 높이기
  - 숫자 주변 문맥 확장
- **단위 정확도 ↓**: 
  - 단위 동의어 사전 확장
  - 단위 정규화 강화
- **키워드 정확도 ↓**: 
  - 도메인 사전 확장
  - 키워드 추출 로직 개선

#### Faithfulness (충실성)가 낮을 때 (환각 발생)
- Context 품질 개선
- Temperature 낮추기 (0.0 권장)
- Prompt에 "자료 기반 답변" 강조

#### Answer Correctness가 낮을 때
- 검색 품질 개선 (Retrieval)
- 답변 생성 Prompt 개선
- Context 수(k) 조정

#### Context Precision이 낮을 때 (불필요한 자료 많음)
- Confidence Threshold 높이기
- Reranking 강화
- BM25 vs Vector 가중치 조정

#### 학술 지표(F1, ROUGE, BLEU)가 낮을 때
- 표현 스타일 맞추기
- 문장 구조 개선
- 핵심 키워드 보존

---

## 실전 예시

### 예시 1: 단일 질문 평가

```python
from scripts.unified_evaluation import UnifiedEvaluator

evaluator = UnifiedEvaluator()

# 평가 실행
results = evaluator.evaluate_all(
    question="AI 플랫폼의 발주기관은?",
    prediction="발주기관은 한국수자원공사입니다.",
    ground_truth="발주기관은 한국수자원공사입니다.",
    contexts=[
        "발주기관: 한국수자원공사",
        "사업명: 금강 유역 스마트 정수장"
    ],
    keywords=["발주기관", "한국수자원공사"]
)

# 결과 출력
evaluator.print_results(results)
```

**출력**:
```
================================================================================
📊 통합 평가 결과
================================================================================

1️⃣  기본 Score (v5 방식)
   종합 점수: 100.0%

2️⃣  도메인 특화 평가
   종합 점수: 100.0%
   숫자 정확도: 100.0%
   단위 정확도: 100.0%
   키워드 정확도: 100.0%

3️⃣  RAG 핵심 3대 지표
   Faithfulness (충실성): 100.0%
   Answer Correctness (정확도): 100.0%
   Context Precision (정밀도): 100.0%
   RAG 종합 점수: 100.0%

4️⃣  학술 표준 지표
   Token F1: 100.0%
   ROUGE-L: 100.0%
   BLEU-2: 100.0%
   Exact Match: 100.0%
```

---

### 예시 2: 배치 평가 (QA 벤치마크)

```python
import json
from scripts.unified_evaluation import UnifiedEvaluator

# QA 데이터 로드
with open('data/qa.json', 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

# 평가할 데이터 준비 (RAG 파이프라인으로 답변 생성 후)
qa_pairs = []
for item in qa_data:
    # 답변 생성 (예시)
    answer_result = pipeline.ask(item['question'])
    
    qa_pairs.append({
        'question': item['question'],
        'prediction': answer_result.text,
        'ground_truth': item['answer'],
        'contexts': [src.chunk.text for src in answer_result.sources],
        'keywords': item.get('accepted_keywords', [])
    })

# 배치 평가 실행
evaluator = UnifiedEvaluator()
batch_results = evaluator.evaluate_batch(qa_pairs)

# 결과 출력
evaluator.print_results(batch_results)

# JSON으로 저장
with open('evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(batch_results, f, indent=2, ensure_ascii=False)
```

---

### 예시 3: 특정 지표만 사용

각 평가 모듈을 독립적으로 사용할 수도 있습니다:

```python
# 1. RAG 핵심 지표만 사용
from scripts.rag_core_metrics import RAGCoreMetrics

rag_scores = RAGCoreMetrics.evaluate_all(
    question, answer, ground_truth, contexts
)
print(f"Faithfulness: {rag_scores['faithfulness']['score']}")

# 2. 도메인 특화만 사용
from scripts.enhanced_scoring import DomainSpecificScoring

scorer = DomainSpecificScoring()
numeric_acc = scorer.score_numeric_accuracy(answer, ground_truth)
print(f"숫자 정확도: {numeric_acc}")

# 3. 학술 지표만 사용
from scripts.academic_metrics import AcademicMetrics

academic = AcademicMetrics.evaluate_all(answer, ground_truth)
print(f"Token F1: {academic['token_f1']['f1']}")
```

---

## 평가 결과 활용 시나리오

### 📊 시나리오 1: 논문 작성

```python
# RAG 핵심 3대 지표 + 학술 지표 사용
evaluator = UnifiedEvaluator()
results = evaluator.evaluate_all(...)

summary = results['summary']

print(f"""
논문 초록 예시:
본 연구의 RAG 챗봇 시스템을 20개 질문으로 평가한 결과,
- Faithfulness: {summary['faithfulness']*100:.1f}%
- Answer Correctness: {summary['answer_correctness']*100:.1f}%
- Context Precision: {summary['context_precision']*100:.1f}%
- Token F1: {summary['token_f1']*100:.1f}%
를 달성하였다.
""")
```

---

### 🔧 시나리오 2: 시스템 개선

```python
# 개선 전후 비교
before_results = evaluator.evaluate_all(...)
# ... 시스템 개선 ...
after_results = evaluator.evaluate_all(...)

# 비교 분석
improvements = {
    key: after_results['summary'][key] - before_results['summary'][key]
    for key in after_results['summary'].keys()
}

print("개선 효과:")
for metric, improvement in improvements.items():
    if improvement > 0:
        print(f"  ✅ {metric}: +{improvement*100:.1f}%p")
    elif improvement < 0:
        print(f"  ⚠️ {metric}: {improvement*100:.1f}%p")
```

---

### 📈 시나리오 3: 버전 비교 (v5 vs v6)

```python
# v5 결과 (기본 Score만 있음)
v5_score = 0.870  # 87.0%

# v6 결과 (모든 지표)
v6_results = evaluator.evaluate_all(...)

print(f"""
버전 비교:
- v5 기본 Score: {v5_score*100:.1f}%
- v6 기본 Score: {v6_results['summary']['basic_v5_score']*100:.1f}%
- 개선폭: +{(v6_results['summary']['basic_v5_score'] - v5_score)*100:.1f}%p

v6 추가 지표:
- Faithfulness: {v6_results['summary']['faithfulness']*100:.1f}%
- Answer Correctness: {v6_results['summary']['answer_correctness']*100:.1f}%
- Token F1: {v6_results['summary']['token_f1']*100:.1f}%
""")
```

---

## 요약

### 어떤 평가를 사용해야 할까?

| 목적 | 추천 평가 | 이유 |
|-----|----------|------|
| **실무 성능 확인** | 기본 Score (v5) + 도메인 특화 | 숫자/단위 강조, 실용적 |
| **논문 작성** | RAG 핵심 3대 + 학술 표준 | 인용 가능, 표준 지표 |
| **타 시스템 비교** | 학술 표준 (F1, ROUGE, BLEU) | 국제 표준 |
| **환각 문제 분석** | Faithfulness (RAG) | 자료 기반 답변 검증 |
| **검색 성능 분석** | Context Precision (RAG) | 검색 효율성 측정 |
| **종합 평가** | 통합 모듈 (모두) | 모든 관점에서 분석 |

### 핵심 요약

```python
# 💡 가장 쉬운 방법: 통합 평가 모듈 사용
from scripts.unified_evaluation import UnifiedEvaluator

evaluator = UnifiedEvaluator()
results = evaluator.evaluate_all(
    question, prediction, ground_truth, contexts, keywords
)
evaluator.print_results(results)

# 모든 지표가 한 번에! 🎉
```

---

## 참고 문헌

1. **RAG 평가**:
   - Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.

2. **QA 평가**:
   - Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP 2016.

3. **요약 평가**:
   - Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. ACL Workshop 2004.

4. **번역 평가**:
   - Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. ACL 2002.

---

**작성일**: 2025-10-12  
**버전**: v6  
**문의**: 프로젝트 이슈 트래커

