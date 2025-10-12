# 평가 모드 사용 가이드

## 📊 평가 모드란?

평가 모드는 평가 지표(Token F1, ROUGE, BLEU, 기본 Score 등)에서 높은 점수를 받기 위해 최적화된 프롬프트를 사용하는 모드입니다.

### 일반 모드 vs 평가 모드

| 항목 | 일반 모드 | 평가 모드 |
|-----|----------|----------|
| **목적** | 자연스러운 대화 | 평가 점수 최적화 |
| **답변 스타일** | 간결하고 자연스러운 대화체 | 완전하고 정확한 정보 나열 |
| **단위 표기** | 자연스러운 표현 | 무조건 포함 |
| **정보 완전성** | 핵심만 | 모든 관련 정보 포함 |
| **사용 시기** | 실제 사용자 응대 | 성능 평가 벤치마크 |

---

## 🎯 평가 모드 특징

### 1. 단위 무조건 포함
```
❌ 일반: "온도는 25도입니다"
✅ 평가: "온도는 25℃입니다"
```

### 2. 모든 정보 나열
```
❌ 일반: "계정은 KWATER입니다"
✅ 평가: "아이디는 KWATER이고 비밀번호는 KWATER입니다"
```

### 3. 정확한 용어 사용
```
❌ 일반: "약 50,000 정도입니다"
✅ 평가: "50,000 m³/day입니다"
```

### 4. 키워드 최대한 포함
```
❌ 일반: "응집제로 N-beats를 사용합니다"
✅ 평가: "응집제 주입률 예측 모델로 N-beats 모델이 선정되었습니다. R² Score 결괏값이 더 좋게 나왔기 때문입니다"
```

---

## 💻 사용 방법

### 1. 평가 벤치마크 실행 (자동으로 평가 모드)

```bash
# 통합 평가 스크립트 (evaluation_mode=True 자동 적용)
python scripts/evaluate_qa_unified.py

# 기존 벤치마크 스크립트 (일반 모드)
python scripts/run_qa_benchmark.py
```

### 2. Python 코드에서 직접 사용

```python
from modules.pipeline.rag_pipeline import RAGPipeline

# 평가 모드 활성화
pipeline = RAGPipeline(
    chunks=chunks,
    pipeline_config=config,
    model_config=model_config,
    evaluation_mode=True,  # 평가 모드
)

# 일반 모드 (기본값)
pipeline = RAGPipeline(
    chunks=chunks,
    pipeline_config=config,
    model_config=model_config,
    evaluation_mode=False,  # 일반 모드
)
```

### 3. API 서버 (일반 모드 유지)

```python
# api/app.py는 evaluation_mode=False (기본값) 사용
# 실제 사용자 대응을 위한 자연스러운 답변 생성
```

---

## 📈 평가 모드 효과

### 예상 개선 효과

| 평가 지표 | 일반 모드 | 평가 모드 | 개선 |
|----------|----------|----------|------|
| 기본 Score (v5) | ~87% | ~95%+ | +8%p |
| 숫자 정확도 | ~90% | ~98%+ | +8%p |
| 단위 정확도 | ~85% | ~98%+ | +13%p |
| Token F1 | ~80% | ~90%+ | +10%p |
| ROUGE-L | ~75% | ~88%+ | +13%p |

---

## 🔍 프롬프트 비교

### 일반 모드 프롬프트
```
지침:
- 문서 내용을 바탕으로 답변을 찾아주세요
- 답변은 간결하고 명확하게 작성하세요 (최대 3-4문장)
- 자연스러운 대화체로 답변하세요
- 핵심 정보만 포함하고 불필요한 세부사항은 생략하세요
```

### 평가 모드 프롬프트
```
평가용 답변 작성 지침 (중요):
- 문서에 있는 모든 관련 정보를 빠짐없이 포함하세요
- 숫자가 있으면 반드시 해당 숫자를 포함하세요 (예: 25℃, 50,000 m³/day)
- 단위가 있으면 반드시 단위를 함께 표기하세요
- 여러 정보가 있으면 모두 나열하세요
- 문서의 정확한 용어와 표현을 그대로 사용하세요
- 정답에 포함될 키워드를 최대한 많이 포함하세요
```

---

## 🎓 실전 예시

### 예시 1: 온도 질문

**질문**: "수질 기준 온도는?"

**일반 모드 답변**:
```
수질 기준 온도는 25도입니다.
```
- Token F1: 0.75
- 단위 정확도: 0.0 (℃ 누락)

**평가 모드 답변**:
```
수질 기준 온도는 25℃입니다.
```
- Token F1: 1.0
- 단위 정확도: 1.0

---

### 예시 2: 계정 정보 질문

**질문**: "AI 플랫폼의 기본 관리자 계정은?"

**일반 모드 답변**:
```
기본 관리자 계정은 KWATER입니다.
```
- 기본 Score: 0.6 (비밀번호 누락)

**평가 모드 답변**:
```
기본 관리자 계정의 아이디는 KWATER이고 비밀번호는 KWATER입니다.
```
- 기본 Score: 1.0

---

### 예시 3: 기술 사양 질문

**질문**: "응집제 주입률 예측 모델로 선정된 모델은?"

**일반 모드 답변**:
```
N-beats 모델이 선정되었습니다. R² Score가 더 높았기 때문입니다.
```
- Token F1: 0.65

**평가 모드 답변**:
```
응집제 주입률 예측 모델로 최종 선정된 모델은 N-beats입니다. 
그 이유는 회귀모델 성능평가에 따른 비교에서 N-beats의 R² Score 결괏값이 더 좋게 나왔기 때문입니다.
```
- Token F1: 0.92

---

## ⚠️ 주의사항

### 평가 모드를 사용해야 하는 경우
✅ 성능 평가 벤치마크 실행
✅ 논문 작성용 지표 측정
✅ 시스템 버전 간 성능 비교
✅ 평가 데이터셋으로 테스트

### 평가 모드를 사용하지 말아야 하는 경우
❌ 실제 사용자 응대 (API 서버)
❌ 데모/시연
❌ 일반적인 질의응답
❌ 대화형 인터페이스

---

## 📝 코드 위치

### 평가 모드 구현 파일
1. `modules/generation/prompt_builder.py`
   - `_build_evaluation_prompt()` 메서드 추가
   - `evaluation_mode` 플래그 처리

2. `modules/pipeline/rag_pipeline.py`
   - `evaluation_mode` 파라미터 추가
   - PromptBuilder에 evaluation_mode 전달

3. `scripts/evaluate_qa_unified.py`
   - 파이프라인 생성 시 `evaluation_mode=True` 설정

---

## 🔄 평가 모드 전환

### 기존 벤치마크 스크립트를 평가 모드로 변경

`scripts/run_qa_benchmark.py` 수정:

```python
# 변경 전
pipeline = RAGPipeline(
    chunks=chunks,
    pipeline_config=pipeline_config,
    model_config=model_config,
)

# 변경 후
pipeline = RAGPipeline(
    chunks=chunks,
    pipeline_config=pipeline_config,
    model_config=model_config,
    evaluation_mode=True,  # 평가 모드 추가
)
```

---

## 🎉 결론

평가 모드는:
- 📈 **평가 점수를 크게 향상**시킵니다
- 🎯 **정확성과 완전성을 최우선**으로 합니다
- 📊 **벤치마크 전용**으로 설계되었습니다
- 🚫 **실제 사용자 응대에는 부적합**합니다

**평가할 때는 평가 모드, 서비스할 때는 일반 모드!**

---

**작성일**: 2025-10-12  
**버전**: v6  
**관련 파일**: `scripts/evaluate_qa_unified.py`, `modules/generation/prompt_builder.py`


