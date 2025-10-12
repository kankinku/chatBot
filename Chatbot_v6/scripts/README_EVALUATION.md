# 평가 시스템 사용 방법 📊

프로젝트의 평가 시스템을 사용하는 모든 방법을 정리했습니다.

---

## 🚀 빠른 시작 (가장 쉬운 방법)

### qa.json으로 전체 평가 실행

```bash
# 기본 실행 (qa.json 사용, 모든 평가 지표 포함)
python scripts/evaluate_qa_unified.py
```

끝! 이 명령어 하나로:
- ✅ qa.json의 모든 질문에 답변 생성
- ✅ 4가지 평가 체계로 평가 (기본 Score, 도메인, RAG, 학술)
- ✅ 결과를 JSON + 텍스트로 저장
- ✅ 상세 리포트 자동 생성

**결과 파일**:
- `out/benchmarks/qa_unified_result.json` - 상세 JSON
- `out/benchmarks/qa_unified_result_summary.txt` - 요약 텍스트

---

## 📋 평가 스크립트 비교

| 스크립트 | 평가 지표 | 사용 시기 | 출력 |
|---------|----------|----------|------|
| **`evaluate_qa_unified.py`** ⭐ | 모든 지표 (4가지 체계) | 종합 평가 필요 시 | JSON + TXT |
| `run_qa_benchmark.py` | 기본 Score + RAG + 학술 | 기존 방식 유지 | JSON + TXT + REPORT |
| `unified_evaluation.py` | 모든 지표 (라이브러리) | 코드에서 직접 사용 | Python 객체 |

---

## 💡 주요 사용법

### 1️⃣ 전체 QA 평가 (qa.json)

```bash
# 기본 실행
python scripts/evaluate_qa_unified.py

# 다른 QA 파일 사용
python scripts/evaluate_qa_unified.py --qa data/qa_test5.json

# 출력 파일 지정
python scripts/evaluate_qa_unified.py --output out/my_result.json

# 다른 LLM 모델 사용
python scripts/evaluate_qa_unified.py --model qwen2.5:7b-instruct-q4_K_M
```

**옵션**:
- `--qa`: QA 데이터 파일 (기본: `data/qa.json`)
- `--corpus`: Corpus 파일 (기본: `data/corpus.jsonl`)
- `--config`: 설정 파일 (기본: `config/default.yaml`)
- `--output`: 출력 파일 (기본: `out/benchmarks/qa_unified_result.json`)
- `--model`: LLM 모델명 (기본: `llama3.1:8b-instruct-q4_K_M`)
- `--mode`: 실행 모드 - `accuracy` 또는 `speed` (기본: `accuracy`)

---

### 2️⃣ Python 코드에서 직접 사용

```python
from scripts.unified_evaluation import UnifiedEvaluator

# 평가자 생성
evaluator = UnifiedEvaluator()

# 단일 평가
results = evaluator.evaluate_all(
    question="AI 플랫폼의 관리자 계정은?",
    prediction="관리자 계정은 KWATER입니다.",
    ground_truth="관리자 계정은 KWATER입니다.",
    contexts=["계정 정보: KWATER / KWATER"],
    keywords=["KWATER", "관리자"]
)

# 결과 출력
evaluator.print_results(results)

# 점수 확인
print(f"기본 Score: {results['summary']['basic_v5_score']}")
print(f"Faithfulness: {results['summary']['faithfulness']}")
print(f"Token F1: {results['summary']['token_f1']}")
```

---

### 3️⃣ 배치 평가 (여러 질문)

```python
from scripts.unified_evaluation import UnifiedEvaluator

evaluator = UnifiedEvaluator()

# 여러 질문 준비
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

# 배치 평가
batch_results = evaluator.evaluate_batch(qa_pairs)

# 평균 점수 확인
stats = batch_results['aggregated_stats']
print(f"평균 기본 Score: {stats['avg_basic_v5_score']}")
print(f"평균 Faithfulness: {stats['avg_faithfulness']}")
```

---

### 4️⃣ 특정 지표만 사용

각 평가 모듈을 독립적으로 사용할 수 있습니다:

```python
# RAG 핵심 3대 지표만
from scripts.rag_core_metrics import RAGCoreMetrics

rag_scores = RAGCoreMetrics.evaluate_all(
    question, answer, ground_truth, contexts
)
print(f"Faithfulness: {rag_scores['faithfulness']['score']}")

# 도메인 특화만
from scripts.enhanced_scoring import DomainSpecificScoring

scorer = DomainSpecificScoring()
numeric_acc = scorer.score_numeric_accuracy(answer, ground_truth)
print(f"숫자 정확도: {numeric_acc}")

# 학술 지표만
from scripts.academic_metrics import AcademicMetrics

academic = AcademicMetrics.evaluate_all(answer, ground_truth)
print(f"Token F1: {academic['token_f1']['f1']}")
```

---

### 5️⃣ 예제 스크립트 실행

```bash
# 5가지 사용 예제 실행
python scripts/example_evaluation.py

# 테스트 (모든 평가 모듈 작동 확인)
python scripts/unified_evaluation.py
```

---

## 📊 4가지 평가 체계

### 1. 기본 Score (v5 방식) - 실무 성능
- **가중치**: 숫자(1.5) > 단위(1.3) > 키워드(1.0)
- **파일**: `scripts/run_qa_benchmark.py` (46-127줄)
- **사용**: 실무 성능 측정

### 2. 도메인 특화 평가 - 정수장 특화
- **지표**: 숫자 정확도, 단위 정확도
- **파일**: `scripts/enhanced_scoring.py`
- **사용**: 기술 정보 정확도 분석

### 3. RAG 핵심 3대 지표 - 학술 연구용
- **지표**: Faithfulness, Answer Correctness, Context Precision
- **파일**: `scripts/rag_core_metrics.py`
- **사용**: 논문 작성, RAG 시스템 평가

### 4. 학술 표준 지표 - 범용 NLP 평가
- **지표**: Token F1, ROUGE-L, BLEU, Exact Match
- **파일**: `scripts/academic_metrics.py`
- **사용**: 타 시스템 비교

---

## 📖 상세 문서

| 문서 | 내용 |
|-----|------|
| `EVALUATION_QUICK_START.md` | 5분 빠른 시작 가이드 |
| `EVALUATION_GUIDE.md` | 상세 가이드 (14페이지) |
| `example_evaluation.py` | 5가지 실전 예제 |

---

## 🎯 결과 해석

### 점수 범위별 평가

| 점수 | 평가 | 의미 |
|-----|------|------|
| 90~100% | 우수 ⭐⭐⭐ | 실무 활용 가능 |
| 70~90% | 양호 ⭐⭐ | 준수한 성능 |
| 50~70% | 보통 ⭐ | 개선 필요 |
| 50% 미만 | 부족 ⚠️ | 시스템 점검 필요 |

### 주요 지표 의미

#### 기본 Score & 도메인 특화
- **90% 이상**: 실무 활용 가능한 수준
- **숫자/단위 정확도**: 기술 정보의 정확성

#### RAG 핵심 3대 지표
- **Faithfulness (충실성)**: 환각(hallucination) 방지
  - "답변이 자료 밖의 거짓말을 했나?"
- **Answer Correctness (정확도)**: 정답과의 사실적 일치
  - "답변이 정답과 사실상 동일한가?"
- **Context Precision (정밀도)**: 검색 효율성
  - "엉뚱한 자료를 가져와 헷갈리지 않았나?"

#### 학술 표준 지표
- **Token F1**: SQuAD 표준, 다른 QA 시스템과 비교 가능
- **ROUGE-L**: 요약 평가, 순서 고려
- **BLEU**: 기계번역 표준, 유창성 측정
- **Exact Match**: 완전 일치 여부

---

## 🎓 논문 작성 시 활용

### 평가 결과 인용

```bash
# 전체 평가 실행
python scripts/evaluate_qa_unified.py

# 결과 확인
cat out/benchmarks/qa_unified_result_summary.txt
```

### 논문 작성 예시

```
본 연구의 RAG 챗봇 시스템을 20개 질문으로 평가한 결과,
Faithfulness 85.3%, Answer Correctness 92.1%, 
Context Precision 75.4%를 달성하였다 (Es et al., 2023).

도메인 특화 평가에서는 94.3%의 정확도를 기록하였으며,
특히 숫자 정보 97.5%, 단위 정보 96.8%의 정확도로
실무 활용에 적합함을 확인하였다.

학술 표준 지표인 Token F1에서 90.2%, ROUGE-L에서 85.7%를
달성하여 기존 시스템 대비 우수한 성능을 보였다.
```

### 참고문헌

```
Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023).
RAGAS: Automated Evaluation of Retrieval Augmented Generation.
arXiv preprint arXiv:2309.15217.

Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016).
SQuAD: 100,000+ Questions for Machine Comprehension of Text.
EMNLP 2016.

Lin, C. Y. (2004).
ROUGE: A Package for Automatic Evaluation of Summaries.
ACL Workshop 2004.
```

---

## 🔧 문제 해결

### Corpus 파일이 없다는 오류

```bash
# Corpus 생성
python scripts/build_corpus.py --pdf-dir data --output data/corpus.jsonl
```

### LLM 연결 오류

```bash
# Ollama 실행 확인
curl http://localhost:11434/api/tags

# 모델 다운로드
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 메모리 부족

```bash
# 배치 크기 줄이기 (config/default.yaml 수정)
# embedding_batch_size: 32 → 16
```

---

## ✨ 핵심 요약

```bash
# 💡 가장 간단한 방법
python scripts/evaluate_qa_unified.py

# 결과 확인
cat out/benchmarks/qa_unified_result_summary.txt

# 끝! 🎉
```

---

**작성일**: 2025-10-12  
**버전**: v6  
**문의**: 프로젝트 이슈 트래커


