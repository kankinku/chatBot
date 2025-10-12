# RAG 챗봇 (기본 버전)

PDF 문서를 기반으로 질문에 답변하는 기본적인 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 기능

- **PDF 텍스트 추출**: PyPDF2를 사용한 기본 텍스트 추출
- **벡터 임베딩**: sentence-transformers의 고정 모델 (all-MiniLM-L6-v2) 사용
- **벡터 DB**: ChromaDB를 사용한 기본 벡터 저장소
- **로컬 LLM**: Ollama의 llama3.1:8b-instruct-q4_K_M 모델 사용

## 프로젝트 구조

```
test_chatbot/
├── data/                      # PDF 파일을 여기에 저장
├── vectordb/                  # 벡터 DB 저장 위치 (자동 생성)
│
├── pdf_to_vectordb.py         # PDF → 벡터 DB 변환 스크립트
├── rag_query.py               # RAG 질의 시스템
├── evaluation_metrics.py      # 평가 지표 모듈 (NEW!)
│
├── main.py                    # 메인 실행 파일 (대화형 인터페이스)
├── test_sample.py             # 샘플 테스트 스크립트 (빠른 확인)
├── benchmark.py               # 전체 벤치마크 실행 (30개 질문)
├── view_benchmark_results.py  # 벤치마크 결과 뷰어
│
├── qa.json                    # 벤치마크용 질문-답변 데이터 (30개)
├── requirements.txt           # Python 의존성
├── Dockerfile                 # Docker 이미지 설정
├── docker-compose.yml         # Docker Compose 설정
│
├── README.md                  # 프로젝트 전체 문서
├── QUICK_START.md             # 빠른 시작 가이드 (NEW!)
└── BENCHMARK_GUIDE.md         # 벤치마크 상세 가이드 (NEW!)
```

## 사전 요구사항

1. **Ollama 설치 및 실행**
   ```bash
   # Ollama 설치 후
   ollama pull llama3.1:8b-instruct-q4_K_M
   ollama serve
   ```

2. **PDF 파일 준비**
   - `data/` 폴더에 PDF 파일을 추가하세요.

## 실행 방법

### Docker 사용 (권장)

```bash
# 1. 이미지 빌드
docker-compose build

# 2. 컨테이너 실행
docker-compose run --rm chatbot
```

### 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 메인 프로그램 실행
python main.py
```

## 사용 방법

1. 프로그램을 실행하면 자동으로 `data/` 폴더의 PDF 파일을 처리합니다.
2. 처리가 완료되면 질문을 입력할 수 있습니다.
3. 질문을 입력하면 관련 문서를 검색하고 LLM이 답변을 생성합니다.
4. 종료하려면 `quit` 또는 `exit`를 입력하세요.

## 수동 실행

### 1. 벡터 DB 생성만 하기
```bash
python pdf_to_vectordb.py
```

### 2. RAG 질의만 하기
```bash
python rag_query.py
```

### 3. 벤치마크 실행
```bash
# qa.json의 질문으로 성능 평가
python benchmark.py

# 벤치마크 결과 확인
python view_benchmark_results.py
```

## 벤치마크

`qa.json` 파일에 포함된 30개의 질문-답변 쌍을 사용하여 RAG 시스템의 성능을 평가할 수 있습니다.

### 벤치마크 실행 방법

```bash
# 1. 샘플 테스트 (빠른 확인용)
python test_sample.py          # 3개 질문으로 테스트
python test_sample.py 5        # 5개 질문으로 테스트

# 2. 전체 벤치마크 실행 (결과는 benchmark_results.json에 저장됨)
python benchmark.py

# 3. 결과 확인
python view_benchmark_results.py
```

### 평가 지표

#### 1️⃣ 기본 Score (v5) - 실무 중심
- 키워드(35%) + 토큰 F1(25%) + 숫자(20%) + ROUGE-L(15%) + BLEU-2(5%)
- 실무에서 중요한 지표들의 가중 평균

#### 2️⃣ 도메인 특화 종합
- 숫자(50%) + 단위(30%) + 키워드(20%)
- 정수장 도메인 특화 평가

#### 3️⃣ RAG 핵심 지표
- **Faithfulness**: 답변이 검색된 문서에 충실한지 (환각 방지)
- **Answer Correctness**: 답변과 정답의 일치도
- **Context Precision**: 검색 품질 (관련 문서 비율)

#### 4️⃣ 학술 표준
- **Token F1**: 단어 수준 F1 점수
- **ROUGE-L**: 최장 공통 부분수열 기반 유사도
- **BLEU-2**: bi-gram 기반 유사도

#### 추가 지표
- **키워드 정확도**: 도메인 전문 용어 매칭
- **숫자 정확도**: 날짜, 버전, 수량 등
- **단위 정확도**: kWh, kg, % 등 단위 매칭
- **성능**: 성공률, 평균 응답 시간

## 주의사항

- 이 프로젝트는 비교용 기본 버전입니다.
- 프로덕션 환경에서는 더 고급 기능(청크 최적화, 하이브리드 검색 등)이 필요할 수 있습니다.
- Ollama가 실행 중이어야 하며, 지정된 모델이 설치되어 있어야 합니다.

## 기술 스택

- **Python 3.10**
- **PyPDF2**: PDF 텍스트 추출
- **sentence-transformers**: 텍스트 임베딩
- **ChromaDB**: 벡터 데이터베이스
- **Ollama**: 로컬 LLM 실행
- **Docker**: 컨테이너화

