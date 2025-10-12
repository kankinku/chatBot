# Chatbot v6 구축 요약

## 완료된 작업

### 1. 핵심 설계 원칙 적용 ✅

#### 1.1 One Source of Truth
- `config/constants.py`: 모든 상수를 단일 소스에서 관리
- `config/pipeline_config.py`: 파이프라인 설정 통합 관리
- 중복 정의 제거, 단일 참조점 확보

#### 1.2 선택값 Config 분리
- `StatusCode` enum으로 성공/실패 표현 방식 선택 가능 (0/1, 1/0, HTTP 코드 등)
- `ErrorCode` enum으로 에러 코드 체계 관리
- 환경별 설정 분리 (development, staging, production)

#### 1.3 Error 처리
- `core/exceptions.py`: 계층적 예외 구조
  - `ChatbotException` (베이스)
  - 카테고리별 예외 (Configuration, Embedding, Retrieval, Generation, etc.)
  - 모든 예외에 error_code, details, cause 포함
- `core/logger.py`: 구조화된 JSON 로깅
  - Console.log 금지
  - 모든 로그에 컨텍스트 정보 자동 추가
  - 에러 추적 및 디버깅 용이

#### 1.4 단일 책임 원칙
- 각 모듈은 하나의 책임만 수행
- 명확한 인터페이스와 추상화
- 예시:
  - `TextCleaner`: 텍스트 정리만
  - `OCRCorrector`: OCR 후처리만
  - `SlidingWindowChunker`: 슬라이딩 윈도우 청킹만
  - `NumericChunker`: 숫자 중심 확장만

### 2. 구현된 모듈

#### 2.1 Config 시스템 (`config/`)
- ✅ `constants.py`: 모든 상수 정의
- ✅ `pipeline_config.py`: 파이프라인 설정 (Thresholds, RRF, Context, etc.)
- ✅ `model_config.py`: 모델 설정 (Embedding, LLM)
- ✅ `environment.py`: 환경별 설정

#### 2.2 Core 모듈 (`core/`)
- ✅ `exceptions.py`: 예외 계층 구조
- ✅ `logger.py`: 구조화된 로깅
- ✅ `types.py`: 데이터 타입 정의 (Chunk, RetrievedSpan, Answer, etc.)

#### 2.3 전처리 (`preprocessing/`)
- ✅ `text_cleaner.py`: 텍스트 정리
- ✅ `ocr_corrector.py`: OCR 후처리
- ✅ `normalizer.py`: 수치/단위/날짜 정규화
- ✅ `pdf_extractor.py`: PDF 텍스트 추출

#### 2.4 청킹 (`chunking/`)
- ✅ `base_chunker.py`: 청커 베이스 클래스
- ✅ `sliding_window_chunker.py`: 슬라이딩 윈도우 방식
- ✅ `numeric_chunker.py`: 숫자 중심 확장

#### 2.5 임베딩 (`embedding/`)
- ✅ `base_embedder.py`: 임베더 베이스 클래스
- ✅ `sbert_embedder.py`: Sentence-BERT 임베딩
- ✅ `factory.py`: 임베더 팩토리 (캐싱 포함)

#### 2.6 배포 설정
- ✅ `requirements.txt`: Python 패키지 의존성
- ✅ `Dockerfile`: Docker 이미지 빌드
- ✅ `docker-compose.yml`: 멀티 컨테이너 구성 (Ollama + Chatbot)
- ✅ `.gitignore`: Git 무시 파일
- ✅ `config/default.yaml`: 기본 설정
- ✅ `data/domain_dictionary.json`: 도메인 사전

## 진행 중인 작업

### 3. 남은 모듈 (현재 작업 중)

#### 3.1 검색 모듈 (`retrieval/`)
- [ ] BM25 검색
- [ ] Vector 검색
- [ ] Hybrid 검색
- [ ] 질문 분류기

#### 3.2 필터링/리랭킹 (`filtering/`, `reranking/`)
- [ ] Context 필터링
- [ ] Cross-encoder 리랭킹

#### 3.3 LLM 통합 (`generation/`)
- [ ] Ollama 클라이언트
- [ ] Prompt 빌더
- [ ] 답변 생성기

#### 3.4 파이프라인 (`pipeline/`)
- [ ] RAG 파이프라인 통합

#### 3.5 API 서버 (`api/`)
- [ ] FastAPI 애플리케이션
- [ ] 엔드포인트 정의
- [ ] 헬스체크

## 프로젝트 구조

```
Chatbot_v6/
├── config/                  ✅ 완료
│   ├── constants.py
│   ├── pipeline_config.py
│   ├── model_config.py
│   ├── environment.py
│   └── default.yaml
├── core/                    ✅ 완료
│   ├── exceptions.py
│   ├── logger.py
│   └── types.py
├── preprocessing/           ✅ 완료
│   ├── text_cleaner.py
│   ├── ocr_corrector.py
│   ├── normalizer.py
│   └── pdf_extractor.py
├── chunking/               ✅ 완료
│   ├── base_chunker.py
│   ├── sliding_window_chunker.py
│   └── numeric_chunker.py
├── embedding/              ✅ 완료
│   ├── base_embedder.py
│   ├── sbert_embedder.py
│   └── factory.py
├── retrieval/              🔄 작업 중
├── generation/             🔄 작업 중
├── pipeline/               🔄 작업 중
├── api/                    🔄 작업 중
├── data/                   ✅ 완료
│   └── domain_dictionary.json
├── requirements.txt        ✅ 완료
├── Dockerfile             ✅ 완료
├── docker-compose.yml     ✅ 완료
└── README.md              ✅ 완료
```

## 4가지 원칙 준수 현황

### ✅ 1. One Source of Truth
- 모든 상수는 `config/constants.py`에 정의
- 설정은 `config/` 디렉토리에서 통합 관리
- 중복 정의 없음

### ✅ 2. 선택값 Config 분리
- `StatusCode`, `ErrorCode` enum으로 유연한 표현
- 환경별 설정 분리
- YAML/JSON 파일로 런타임 설정 가능

### ✅ 3. Error 처리
- 계층적 예외 구조
- 구조화된 로깅 (JSON)
- 모든 에러에 추적 정보 포함
- Console.log 사용 금지

### ✅ 4. 단일 책임 원칙
- 각 클래스/모듈은 하나의 책임만
- 명확한 인터페이스
- 테스트 및 유지보수 용이

## 다음 단계

1. 검색 모듈 완성
2. LLM 통합 모듈 완성
3. RAG 파이프라인 구축
4. API 서버 구현
5. 통합 테스트
6. 문서화 완료

