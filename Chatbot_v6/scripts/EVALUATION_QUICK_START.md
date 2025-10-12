# 평가 시스템 빠른 시작 가이드 ⚡

5분 안에 평가 시스템을 사용할 수 있습니다!

---

## 🚀 가장 빠른 방법

### 1단계: 통합 평가 모듈 import

```python
from scripts.unified_evaluation import UnifiedEvaluator

evaluator = UnifiedEvaluator()
```

### 2단계: 평가 실행

```python
results = evaluator.evaluate_all(
    question="질문",
    prediction="생성된 답변",
    ground_truth="정답",
    contexts=["참고 자료 1", "참고 자료 2"],  # 선택 사항
    keywords=["키워드1", "키워드2"]          # 선택 사항
)
```

### 3단계: 결과 확인

```python
evaluator.print_results(results)
```

끝! 🎉

---

## 📊 4가지 평가 체계

| 평가 | 파일 | 주요 지표 | 사용 시기 |
|-----|------|----------|----------|
| **1. 기본 Score (v5)** | `run_qa_benchmark.py` | 가중치 기반 종합 점수 | 실무 성능 |
| **2. 도메인 특화** | `enhanced_scoring.py` | 숫자/단위 정확도 | 기술 정보 |
| **3. RAG 핵심 3대** | `rag_core_metrics.py` | Faithfulness, Correctness, Precision | 논문 작성 |
| **4. 학술 표준** | `academic_metrics.py` | F1, ROUGE, BLEU, EM | 타 시스템 비교 |

---

## 💡 주요 지표 한눈에 보기

### 1️⃣ 기본 Score (v5 방식)
```python
# 가중치: 숫자(1.5) > 단위(1.3) > 키워드(1.0)
score = benchmark.score_answer(prediction, ground_truth, keywords)
# 결과: 0.0 ~ 1.0
```

### 2️⃣ 도메인 특화
```python
from scripts.enhanced_scoring import DomainSpecificScoring
scorer = DomainSpecificScoring()

# 종합 평가
result = scorer.score_answer_v5_style(pred, gold, keywords)
# {'total_score': 0.95, 'numeric_score': 1.0, 'unit_score': 1.0, ...}

# 숫자만
numeric_acc = scorer.score_numeric_accuracy(pred, gold)  # 0.0 ~ 1.0

# 단위만
unit_acc = scorer.score_unit_accuracy(pred, gold)  # 0.0 ~ 1.0
```

### 3️⃣ RAG 핵심 3대 지표
```python
from scripts.rag_core_metrics import RAGCoreMetrics

results = RAGCoreMetrics.evaluate_all(question, answer, ground_truth, contexts)
# {
#   'faithfulness': {'score': 0.85, ...},        # 충실성 (환각 방지)
#   'answer_correctness': {'score': 0.92, ...},  # 정확도
#   'context_precision': {'score': 0.75, ...},   # 검색 효율성
#   'overall_score': 0.84
# }
```

### 4️⃣ 학술 표준 지표
```python
from scripts.academic_metrics import AcademicMetrics

metrics = AcademicMetrics.evaluate_all(pred, gold)
# {
#   'exact_match': 0.0,
#   'token_f1': {'f1': 0.90, ...},
#   'rouge_l': {'f1': 0.83, ...},
#   'bleu_1': 0.88,
#   'bleu_2': 0.75
# }
```

---

## 🎯 상황별 사용법

### ✅ 실무 성능 측정
```python
evaluator = UnifiedEvaluator()
results = evaluator.evaluate_all(...)
print(f"실무 점수: {results['summary']['basic_v5_score']*100:.1f}%")
```

### ✅ 논문 작성
```python
results = evaluator.evaluate_all(...)
print(f"Faithfulness: {results['summary']['faithfulness']*100:.1f}%")
print(f"Answer Correctness: {results['summary']['answer_correctness']*100:.1f}%")
print(f"Token F1: {results['summary']['token_f1']*100:.1f}%")
```

### ✅ 여러 질문 배치 평가
```python
qa_pairs = [
    {'question': '...', 'prediction': '...', 'ground_truth': '...', ...},
    {'question': '...', 'prediction': '...', 'ground_truth': '...', ...},
]
batch_results = evaluator.evaluate_batch(qa_pairs)
evaluator.print_results(batch_results)
```

### ✅ 두 버전 비교
```python
v5_results = evaluator.evaluate_all(..., prediction=v5_answer)
v6_results = evaluator.evaluate_all(..., prediction=v6_answer)

improvement = (
    v6_results['summary']['basic_v5_score'] - 
    v5_results['summary']['basic_v5_score']
) * 100
print(f"개선폭: +{improvement:.1f}%p")
```

---

## 📝 실행 예제

### 터미널에서 테스트
```bash
# 통합 평가 모듈 테스트
python scripts/unified_evaluation.py

# 예제 스크립트 실행
python scripts/example_evaluation.py
```

### QA 벤치마크 실행
```bash
# 전체 QA 데이터 평가 (모든 지표 포함)
python scripts/run_qa_benchmark.py --qa data/qa.json --corpus data/corpus.jsonl
```

---

## 📖 더 자세한 정보

- **상세 가이드**: `scripts/EVALUATION_GUIDE.md`
- **예제 코드**: `scripts/example_evaluation.py`
- **RAG 평가**: `scripts/rag_core_metrics.py`
- **도메인 평가**: `scripts/enhanced_scoring.py`
- **학술 지표**: `scripts/academic_metrics.py`

---

## 🎓 논문 인용 예시

### RAG 시스템 평가
```
본 연구의 RAG 챗봇 시스템을 20개 질문으로 평가한 결과,
Faithfulness 85.3%, Answer Correctness 92.1%, Context Precision 75.4%를 
달성하였다 (Es et al., 2023).
```

### 참고문헌
```
Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023).
RAGAS: Automated Evaluation of Retrieval Augmented Generation.
arXiv preprint arXiv:2309.15217.
```

---

## ⚡ 핵심 요약

```python
# 💡 이것만 기억하세요!
from scripts.unified_evaluation import UnifiedEvaluator

evaluator = UnifiedEvaluator()
results = evaluator.evaluate_all(
    question, prediction, ground_truth, contexts, keywords
)
evaluator.print_results(results)

# 모든 지표를 한 번에 확인! 🎉
```

---

**작성일**: 2025-10-12  
**버전**: v6

