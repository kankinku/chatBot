# 빠른 시작 가이드

기본 RAG 챗봇 프로젝트를 빠르게 시작하는 방법입니다.

## ⚡ 5분 안에 시작하기

### 1단계: Ollama 설정 (1분)

```bash
# Ollama가 없다면 설치: https://ollama.ai
ollama pull llama3.1:8b-instruct-q4_K_M
ollama serve
```

### 2단계: 프로젝트 실행 (1분)

```bash
# 의존성 설치 (이미 설치했다면 생략)
pip install -r requirements.txt

# 메인 프로그램 실행
python main.py
```

### 3단계: 질문하기 (3분)

프로그램이 자동으로 PDF를 처리한 후 질문을 받습니다:

```
질문: 고산 정수장 시스템의 발주기관은?
```

종료하려면 `quit` 또는 `exit` 입력

## 📊 성능 평가하기

### 샘플 테스트 (30초)

```bash
python test_sample.py
```

- 3개 질문으로 빠른 테스트
- 각 질문별 상세 평가 점수 확인

### 전체 벤치마크 (5-10분)

```bash
python benchmark.py
```

- 30개 질문 전체 평가
- 결과를 `benchmark_results.json`에 저장

### 결과 확인

```bash
python view_benchmark_results.py
```

대화형 메뉴로 결과 분석

## 📊 평가 결과 예시

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

## 🎯 평가 지표 간단 설명

### 실무용
- **기본 Score**: 종합 점수 (키워드 + 숫자 + 텍스트 유사도)
- **도메인 특화**: 정수장 전문 지식 (숫자 + 단위 + 키워드)

### 연구용
- **Faithfulness**: 환각 없이 문서 기반 답변
- **Answer Correctness**: 정답과의 일치도
- **Context Precision**: 검색 품질
- **Token F1, ROUGE-L**: 학술 표준 지표

## 🔧 문제 해결

### Ollama 연결 오류
```bash
# Ollama가 실행 중인지 확인
ollama list

# 모델이 있는지 확인
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 벡터 DB 재생성
```bash
# PowerShell
Remove-Item -Recurse -Force vectordb

# 다시 실행
python main.py
```

### PDF 추가/변경
1. `data/` 폴더에 PDF 추가
2. `vectordb/` 폴더 삭제
3. `python main.py` 재실행

## 📚 더 알아보기

- **BENCHMARK_GUIDE.md**: 평가 지표 상세 설명
- **README.md**: 전체 프로젝트 문서
- **evaluation_metrics.py**: 평가 로직 구현

## 🚀 다음 단계

### 성능 개선
1. `pdf_to_vectordb.py`의 청크 크기 조정
2. `rag_query.py`의 `top_k` 값 조정
3. LLM 프롬프트 개선

### 비교 실험
1. 벤치마크 결과 백업
2. 파라미터 변경
3. 재실행 후 결과 비교

### 고급 기능
- 하이브리드 검색 (키워드 + 의미론적)
- 리랭킹 (검색 결과 재정렬)
- 청크 최적화 (오버랩, 계층적 분할)

## 💡 팁

1. **처음 실행**: `test_sample.py`로 빠른 확인
2. **정상 동작 확인**: `benchmark.py`로 전체 평가
3. **결과 분석**: `view_benchmark_results.py`로 상세 확인
4. **개선 실험**: 파라미터 조정 후 재평가

## ⚠️ 주의사항

- Ollama가 실행 중이어야 합니다
- 처음 실행 시 PDF 처리에 시간이 걸립니다
- LLM 답변은 실행마다 약간 다를 수 있습니다
- 대용량 PDF는 메모리를 많이 사용합니다

