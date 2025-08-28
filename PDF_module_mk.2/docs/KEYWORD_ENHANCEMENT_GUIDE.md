# 키워드 인식률 향상 가이드

## 개요

이 문서는 PDF QA 시스템의 키워드 인식률을 향상시키기 위한 개선사항과 사용법을 설명합니다.

## 주요 개선사항

### 1. 도메인별 전문 용어 사전

#### 지원 도메인
- **general**: 일반적인 용어
- **technical**: IT/기술 도메인
- **business**: 비즈니스 도메인
- **academic**: 학술 도메인

#### 사용법
```python
from utils.keyword_enhancer import KeywordEnhancer

# 기술 도메인용 키워드 향상기
enhancer = KeywordEnhancer(domain="technical")

# 질문 분석기에 도메인 설정
from core.question_analyzer import QuestionAnalyzer
analyzer = QuestionAnalyzer(domain="technical")
```

### 2. 키워드 확장 기능

#### 동의어 확장
- "방법" → ["방식", "기법", "기술", "수단"]
- "시스템" → ["플랫폼", "솔루션", "도구", "애플리케이션"]

#### 약어 확장
- "API" → "Application Programming Interface"
- "DB" → "데이터베이스"
- "UI" → "사용자 인터페이스"

#### 사용법
```python
# 기본 키워드
basic_keywords = ["시스템", "API", "성능"]

# 향상된 키워드 (동의어, 약어 포함)
enhanced_keywords = enhancer.enhance_keywords(basic_keywords)
print(enhanced_keywords)
# 출력: ['시스템', '플랫폼', '솔루션', '도구', 'API', 'Application Programming Interface', '성능', '효율', '속도', '품질']
```

### 3. 하이브리드 검색

#### 벡터 유사도 + 키워드 매칭
- 벡터 유사도: 60% 가중치
- 키워드 매칭: 40% 가중치

#### 키워드 가중치 계산
```python
# 도메인 키워드 가중치
if keyword in domain_keywords:
    weight *= domain_keywords[keyword]

# 컨텍스트 빈도 가중치
frequency = context.lower().count(keyword.lower())
weight *= (1.0 + frequency * 0.1)

# 키워드 길이 가중치
weight *= (1.0 + len(keyword) * 0.05)
```

### 4. 향상된 신뢰도 계산

#### 신뢰도 구성 요소
1. **관련 청크 유사도** (30%)
2. **키워드 매칭 점수** (25%)
3. **답변 품질 점수** (20%)
4. **컨텍스트 활용도** (15%)
5. **답변 완성도** (10%)

#### 키워드 매칭 점수 계산
```python
# 정확한 키워드 매칭 (50%)
exact_score = exact_matches / total_keywords

# 부분 키워드 매칭 (30%)
partial_score = partial_matches / total_keywords

# 동의어 매칭 (20%)
synonym_score = synonym_matches / total_keywords

total_score = exact_score * 0.5 + partial_score * 0.3 + synonym_score * 0.2
```

## 사용 예시

### 1. 기본 사용법

```python
from core.question_analyzer import QuestionAnalyzer
from core.vector_store import FAISSVectorStore
from core.answer_generator import AnswerGenerator

# 기술 도메인으로 초기화
analyzer = QuestionAnalyzer(domain="technical")
vector_store = FAISSVectorStore()
generator = AnswerGenerator()

# 질문 분석
question = "API 성능을 어떻게 최적화할 수 있나요?"
analyzed = analyzer.analyze_question(question)

print(f"추출된 키워드: {analyzed.keywords}")
# 출력: ['API', 'Application Programming Interface', '성능', '효율', '속도', '최적화', '고도화', '개선']
```

### 2. 키워드 향상 테스트

```python
from utils.keyword_enhancer import KeywordEnhancer

enhancer = KeywordEnhancer(domain="technical")

# 텍스트에서 키워드 추천
text = "시스템 API 성능 최적화를 위해 캐싱 전략을 구현했습니다."
recommended = enhancer.recommend_keywords(text, max_keywords=10)

for keyword, weight in recommended:
    print(f"{keyword}: {weight:.2f}")
```

### 3. 성능 비교

```python
# 개선 전 vs 개선 후 비교
before_keywords = ["시스템", "API", "성능"]
after_keywords = enhancer.enhance_keywords(before_keywords)

print(f"개선 전: {len(before_keywords)}개 키워드")
print(f"개선 후: {len(after_keywords)}개 키워드")
```

## 성능 최적화 팁

### 1. 도메인 선택
- 문서의 주제에 맞는 도메인 선택
- 기술 문서: `technical`
- 비즈니스 문서: `business`
- 학술 문서: `academic`

### 2. 키워드 가중치 조정
```python
# 도메인별 키워드 가중치 수정
enhancer.domain_keywords["API"] = 1.5  # 더 높은 가중치
enhancer.domain_keywords["시스템"] = 1.3
```

### 3. 동의어 사전 확장
```python
# 새로운 동의어 추가
enhancer.synonym_dict["새로운용어"] = ["동의어1", "동의어2", "동의어3"]
```

## 테스트 및 검증

### 1. 테스트 스크립트 실행
```bash
python test_keyword_enhancement.py
```

### 2. 성능 지표
- 키워드 인식률: 30-50% 향상 예상
- 검색 정확도: 20-30% 향상 예상
- 답변 품질: 15-25% 향상 예상

### 3. 모니터링
```python
# 키워드 추출 성능 모니터링
import time

start_time = time.time()
keywords = analyzer.analyze_question(question).keywords
end_time = time.time()

print(f"처리 시간: {(end_time - start_time) * 1000:.2f}ms")
print(f"추출된 키워드 수: {len(keywords)}")
```

## 문제 해결

### 1. 키워드가 너무 많이 추출되는 경우
```python
# 키워드 개수 제한
top_keywords = [kw for kw, _ in keyword_weights[:10]]  # 상위 10개만
```

### 2. 도메인 특화 키워드가 부족한 경우
```python
# 도메인 키워드 사전 확장
enhancer.domain_keywords.update({
    "새로운키워드": 1.2,
    "또다른키워드": 1.1
})
```

### 3. 성능이 느린 경우
```python
# 캐싱 활용
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_enhance_keywords(keywords_tuple):
    return enhancer.enhance_keywords(list(keywords_tuple))
```

## 향후 개선 계획

### 1. 단기 계획
- [ ] 더 많은 도메인 추가 (의료, 법률, 교육 등)
- [ ] 동적 키워드 가중치 조정
- [ ] 사용자 피드백 기반 학습

### 2. 중기 계획
- [ ] 머신러닝 기반 키워드 추출
- [ ] 다국어 지원
- [ ] 실시간 키워드 업데이트

### 3. 장기 계획
- [ ] 딥러닝 기반 키워드 관계 분석
- [ ] 컨텍스트 인식 키워드 추출
- [ ] 자동 도메인 감지

## 결론

키워드 인식률 향상 기능을 통해 PDF QA 시스템의 전반적인 성능이 크게 개선됩니다. 도메인별 전문 용어 사전, 동의어 확장, 하이브리드 검색 등의 기능을 활용하여 더 정확하고 관련성 높은 답변을 생성할 수 있습니다.

정기적인 테스트와 모니터링을 통해 시스템 성능을 지속적으로 개선하시기 바랍니다.
