# 테스트 가이드

## 테스트 실행

### 전체 테스트 실행
```bash
pytest
```

### 단위 테스트만 실행
```bash
pytest tests/unit/
```

### 통합 테스트만 실행
```bash
pytest tests/integration/
```

### 특정 테스트 파일 실행
```bash
pytest tests/unit/test_question_analyzer.py
```

### 특정 테스트 함수 실행
```bash
pytest tests/unit/test_question_analyzer.py::TestQuestionAnalyzer::test_numeric_question_detection
```

### 커버리지 포함
```bash
pytest --cov=modules --cov-report=html
```

### 느린 테스트 제외
```bash
pytest -m "not slow"
```

## 테스트 작성 가이드

### 1. 단위 테스트
- 각 모듈의 개별 함수/메서드 테스트
- Mock 사용하여 외부 의존성 제거
- 빠른 실행 (< 1초)

### 2. 통합 테스트
- 여러 모듈 간 상호작용 테스트
- 실제 의존성 사용
- 실행 시간 허용 (< 10초)

### 3. Fixture 사용
- `conftest.py`에 공통 fixture 정의
- 테스트 데이터 재사용

## 현재 테스트 커버리지

### Unit Tests
- ✅ QuestionAnalyzer
- ✅ ContextFilter
- ✅ BM25Retriever
- ⏳ VectorRetriever (TODO)
- ⏳ HybridRetriever (TODO)
- ⏳ AnswerGenerator (TODO)

### Integration Tests
- ✅ RAGPipeline
- ⏳ API Endpoints (TODO)

## 추가 필요 테스트

1. **VectorRetriever 테스트**
   - 임베딩 생성 및 검색
   - FAISS vs Simple 백엔드

2. **HybridRetriever 테스트**
   - BM25 + Vector 병합
   - 가중치 조정

3. **AnswerGenerator 테스트**
   - 프롬프트 생성
   - LLM 통신 (Mock)
   - Recovery 모드

4. **API 테스트**
   - 엔드포인트 테스트
   - 인증 테스트 (향후)

