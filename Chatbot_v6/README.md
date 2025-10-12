# Chatbot v6 - 정수처리 챗봇 (리빌딩 버전)

## 핵심 설계 원칙

### 1. One Source of Truth
- 모든 설정, 상수, 타입은 단일 소스에서 관리
- 중복 정의 없음
- `config/` 디렉토리에 모든 설정 집중

### 2. 선택값 Config 분리
- 성공/실패 상태 코드 (0/1 등) 설정 가능
- 환경별, 모드별 설정 분리
- `config/constants.py`에 모든 상수 정의

### 3. Error 처리
- Exception hierarchy 구조화
- Console.log 금지, 구조화된 로깅 사용
- 모든 에러는 명시적 처리 (눈가리고 아웅 금지)
- `core/exceptions.py`, `core/logger.py` 참조

### 4. 단일 책임 원칙
- 각 모듈/클래스는 하나의 책임만
- 명확한 인터페이스 정의
- 모듈 간 의존성 최소화

## 프로젝트 구조

```
Chatbot_v6/
├── config/                       # 모든 설정 (One Source of Truth)
│   ├── constants.py              # 상수 정의
│   ├── pipeline_config.py        # 파이프라인 설정
│   ├── model_config.py           # 모델 설정
│   ├── environment.py            # 환경별 설정
│   └── default.yaml              # 기본 설정 파일
├── modules/                      # 모든 비즈니스 로직 모듈
│   ├── core/                     # 핵심 기능
│   │   ├── exceptions.py         # 예외 계층 구조
│   │   ├── types.py              # 데이터 타입 정의
│   │   └── logger.py             # 구조화된 로깅
│   ├── preprocessing/            # 전처리 (단일 책임)
│   │   ├── pdf_extractor.py     # PDF 텍스트 추출
│   │   ├── text_cleaner.py      # 텍스트 정리
│   │   ├── ocr_corrector.py     # OCR 후처리
│   │   └── normalizer.py        # 정규화
│   ├── chunking/                 # 청킹 (단일 책임)
│   │   ├── base_chunker.py      # 기본 청커
│   │   ├── sliding_window_chunker.py  # 슬라이딩 윈도우
│   │   └── numeric_chunker.py   # 숫자 중심 청커
│   ├── embedding/                # 임베딩 (단일 책임)
│   │   ├── base_embedder.py
│   │   ├── sbert_embedder.py
│   │   └── factory.py           # 임베더 팩토리
│   ├── retrieval/                # 검색 (단일 책임)
│   │   ├── bm25_retriever.py
│   │   ├── vector_retriever.py
│   │   └── hybrid_retriever.py
│   ├── analysis/                 # 질문 분석
│   │   └── question_analyzer.py
│   ├── reranking/                # 리랭킹 (단일 책임)
│   │   └── reranker.py
│   ├── filtering/                # 필터링 (단일 책임)
│   │   ├── context_filter.py
│   │   ├── deduplicator.py
│   │   └── guardrail.py
│   ├── generation/               # 생성 (단일 책임)
│   │   ├── llm_client.py
│   │   ├── prompt_builder.py
│   │   └── answer_generator.py
│   └── pipeline/                 # 전체 파이프라인 조율
│       └── rag_pipeline.py
├── api/                          # API 엔드포인트
│   └── app.py
├── scripts/                      # 유틸리티 스크립트
│   ├── build_corpus.py           # Corpus 생성
│   ├── run_qa_benchmark.py       # QA 벤치마크
│   ├── test_complete_system.py   # 전체 시스템 테스트
│   └── show_test_proof.py        # 테스트 증거 출력
├── data/                         # 데이터 파일
│   ├── corpus.jsonl              # 생성된 corpus
│   ├── qa.json                   # QA 평가 데이터셋
│   ├── domain_dictionary.json    # 도메인 사전
│   └── *.pdf                     # PDF 문서들
├── out/                          # 출력 파일
│   ├── benchmarks/               # 벤치마크 결과
│   │   ├── qa_full_result.json
│   │   └── qa_full_result_summary.txt
│   └── tests/                    # 테스트 결과
│       ├── test_report.json
│       └── qa_sample.json
├── markdown/                     # 문서들
│   ├── README.md                 # 문서 가이드
│   ├── OPTIMIZATION_OPPORTUNITIES.md
│   └── ...
├── logs/                         # 로그 파일
├── vector_store/                 # 벡터 저장소 (자동 생성)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 주요 기능

### 1. 전처리
- PDF 텍스트 추출
- OCR 후처리 (도메인 사전 기반)
- 텍스트 정규화

### 2. 청킹
- 슬라이딩 윈도우 청킹
- 숫자 중심 청킹 (측정값 보존)
- 의미 기반 청킹

### 3. 임베딩
- Sentence-BERT 기반 임베딩
- 캐싱 및 메모리 최적화

### 4. 검색
- BM25 검색 (키워드 기반)
- Vector 검색 (의미 기반)
- Hybrid 검색 (가중치 조합)
- 질문 유형별 동적 가중치

### 5. 리랭킹
- Cross-encoder 리랭킹 (옵션)
- 컨텍스트 품질 필터링

### 6. 생성
- Ollama LLM 통합
- 도메인 특화 프롬프트
- 답변 품질 검증 및 재시도

## 설치

```bash
# Docker 환경 (권장)
docker-compose up -d

# 로컬 개발
pip install -r requirements.txt
```

## 사용법

```python
from pipeline.rag_pipeline import RAGPipeline
from config.pipeline_config import PipelineConfig

# 설정 로드
config = PipelineConfig.from_file("config/default.yaml")

# 파이프라인 초기화
pipeline = RAGPipeline(config)

# 질문 응답
result = pipeline.ask("고산 정수장 AI플랫폼 URL은?")
print(result.answer)
print(f"신뢰도: {result.confidence}")
```

## 설정

모든 설정은 `config/` 디렉토리에서 관리됩니다:

- `constants.py`: 상수 정의 (상태 코드, 기본값 등)
- `pipeline_config.py`: 파이프라인 설정
- `model_config.py`: 모델 설정
- `environment.py`: 환경별 설정

## 로깅

구조화된 JSON 로깅 사용:

```python
from core.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started", extra={"question": "...", "status": 0})
logger.error("LLM timeout", extra={"error_code": "E001"}, exc_info=True)
```

## 에러 처리

모든 예외는 `core/exceptions.py`에 정의된 계층 구조를 따릅니다:

```python
from core.exceptions import (
    ChatbotException,      # 베이스 예외
    ConfigurationError,    # 설정 오류
    EmbeddingError,        # 임베딩 오류
    RetrievalError,        # 검색 오류
    GenerationError,       # 생성 오류
)
```

## 빠른 시작

### 1. Corpus 생성 (처음 한 번만)

```bash
cd scripts
python build_corpus.py --pdf-dir ../data --output ../data/corpus.jsonl
```

### 2. QA 벤치마크 실행

```bash
# Ollama 서버가 localhost:11434에서 실행 중이어야 합니다
cd scripts
python run_qa_benchmark.py
```

### 3. 전체 시스템 테스트

```bash
cd scripts
python test_complete_system.py
python show_test_proof.py  # 테스트 결과 확인
```

### 4. API 서버 실행

```bash
# 로컬 실행
cd api
python app.py

# Docker 실행 (프로젝트 루트에서)
docker-compose up -d
```

## 개발

### 테스트
```bash
pytest tests/
```

### 코드 품질
```bash
# Linting
ruff check .

# Formatting
black .
```

## 라이선스

MIT License

