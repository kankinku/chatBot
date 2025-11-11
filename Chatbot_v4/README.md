# WILL 챗봇 시스템 통합 문서

## 📋 목차
1. [시스템 개요](#시스템-개요)
2. [프로젝트 구조](#프로젝트-구조)
3. [기술 스택](#기술-스택)
4. [설치 및 실행](#설치-및-실행)
5. [사용 가이드](#사용-가이드)
6. [로깅 및 모니터링](#로깅-및-모니터링)
7. [기술 상세](#기술-상세)
8. [문제 해결](#문제-해결)

---

## 🎯 시스템 개요

본 시스템은 PDF 문서에서 데이터를 추출하고 LLM을 통해 질문에 대한 답변을 생성하는 RAG(Retrieval-Augmented Generation) 기반 챗봇입니다. 교통 데이터 도메인에 특화된 질문-답변 시스템으로, 정확도 1순위, 속도 2순위, 최적화 3순위를 지향합니다.

### 주요 특징
- **하이브리드 검색**: 벡터 검색 + BM25 키워드 검색 결합
- **도메인 특화**: 교통 데이터 전문 용어 및 질문 유형 분류
- **품질 보장**: 다층 가드레일 및 신뢰도 검증 시스템
- **실시간 모니터링**: 상세한 로깅 및 성능 메트릭 수집

---

## 📁 프로젝트 구조

```
Jungsu-ChatBot/
├── 📁 ollama-chatbot-api-ifro/          # 챗봇 AI 엔진
│   ├── 📁 data/                         # 데이터 저장소
│   │   ├── 📁 pdfs/                     # 원본 PDF 문서
│   │   ├── 📁 tests/                    # 테스트 데이터
│   │   ├── 📄 corpus_v1.jsonl           # 메인 코퍼스
│   │   └── 📄 domain_dictionary.json   # 도메인 사전
│   ├── 📁 scripts/                      # 실행 스크립트
│   │   ├── 🔧 build_corpus_from_pdfs.py
│   │   ├── 🔧 build_vector_index.py
│   │   ├── 🚀 autorun.py
│   │   └── 🚀 manual_cli.py
│   ├── 📁 src/unifiedpdf/               # 핵심 라이브러리
│   ├── 📁 vector_store/                 # 벡터 저장소
│   ├── 📁 out/                          # 결과 출력
│   ├── 📁 server/                       # 웹 서버
│   └── 📁 logs/                         # 로그 파일
├── 📁 django-react-backend-api-ifro/   # 백엔드 프록시 서버
│   └── 📁 src/
│       ├── 📁 dashboard/                # Django 설정
│       └── 📁 chatbot_proxy/            # 챗봇 프록시 API
└── 📁 django-react-frontend-ifro/      # 프론트엔드
    └── 📁 src/
        ├── 📁 features/chatbot/         # 챗봇 UI 컴포넌트
        └── 📁 shared/                   # 공통 유틸리티
```

---

## 🛠 기술 스택

### AI/ML 스택
- **LLM**: Llama 3 8B Instruct (Q4_K_M 양자화)
- **임베딩**: jhgan/ko-sroberta-multitask (768차원)
- **벡터 저장소**: FAISS, HNSW, TF-IDF
- **검색**: BM25 + 벡터 하이브리드 검색
- **리랭킹**: BAAI/bge-reranker-v2-m3

### 백엔드 스택
- **챗봇 서버**: FastAPI + Ollama
- **프록시 서버**: Django + Ninja
- **데이터베이스**: SQLite (프록시 서버용)

### 프론트엔드 스택
- **UI**: React + TypeScript
- **스타일링**: Tailwind CSS
- **HTTP 클라이언트**: Axios

---

## 🚀 설치 및 실행

### 1. 준비물
- Python 3.10 이상
- PowerShell 권장 (한글 콘솔: `chcp 65001` 또는 `set PYTHONUTF8=1`)
- (선택) Docker Desktop + GPU
- (선택) Ollama (로컬 LLM)

### 2. 설치 (권장 순서)

#### CPU 환경 (충돌 최소화 핀 고정)
```bash
# 1) pip 업그레이드
pip install --upgrade pip

# 2) PyTorch CPU 전용 설치
pip install --upgrade --force-reinstall "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cpu

# 3) 의존성 설치
pip install -r ollama-chatbot-api-ifro/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

#### GPU 환경 (선택)
PyTorch 공식 가이드에 따라 CUDA 빌드 지정 설치 후:
```bash
pip install -r ollama-chatbot-api-ifro/requirements.txt --no-deps
```

### 3. 폴더/파일 배치
- **PDF 문서**: `ollama-chatbot-api-ifro/data/pdfs/` 또는 `ollama-chatbot-api-ifro/data/tests/`
- **QA 테스트**: `ollama-chatbot-api-ifro/data/tests/qa.json`
- **출력 디렉터리**: 결과 리포트는 `ollama-chatbot-api-ifro/out/` 하위에 생성

---

## 📖 사용 가이드

### 1. 원클릭 실행 (Quickstart)

#### 단일 질문 테스트
```bash
python ollama-chatbot-api-ifro/scripts/autorun.py --pdf ollama-chatbot-api-ifro/data/tests --backend auto --question "교통사고가 발생했을 때 어떻게 해야 하나요?"
```

#### 인터랙티브 CLI
```bash
python ollama-chatbot-api-ifro/scripts/autorun.py --pdf ollama-chatbot-api-ifro/data/tests --backend auto --interactive
```

#### 벤치마크 실행
```bash
python ollama-chatbot-api-ifro/scripts/autorun.py --pdf ollama-chatbot-api-ifro/data/tests --qa ollama-chatbot-api-ifro/data/tests/qa.json --backend auto
```

#### API 서버 시작
```bash
python ollama-chatbot-api-ifro/scripts/autorun.py --pdf ollama-chatbot-api-ifro/data/tests --backend auto --server --host 0.0.0.0 --port 8000
```

### 2. 수동 절차

#### PDF → 코퍼스 생성
```bash
python ollama-chatbot-api-ifro/scripts/build_corpus_from_pdfs.py --pdf_dir ollama-chatbot-api-ifro/data/pdfs --out ollama-chatbot-api-ifro/data/corpus_v1.jsonl --pdf-extractor auto --ocr auto --ocr-lang kor+eng --ocr-dpi 200
```

#### 코퍼스 → 벡터 인덱스 생성
```bash
# FAISS
python ollama-chatbot-api-ifro/scripts/build_vector_index.py --corpus ollama-chatbot-api-ifro/data/corpus_v1.jsonl --backend faiss --outdir ollama-chatbot-api-ifro/vector_store

# HNSW
python ollama-chatbot-api-ifro/scripts/build_vector_index.py --corpus ollama-chatbot-api-ifro/data/corpus_v1.jsonl --backend hnsw --outdir ollama-chatbot-api-ifro/vector_store
```

#### 수동 CLI 테스트
```bash
python ollama-chatbot-api-ifro/scripts/manual_cli.py --corpus ollama-chatbot-api-ifro/data/corpus_v1.jsonl --mode accuracy --store-backend auto --question "질문 내용"
```

### 3. 전체 시스템 실행

#### Docker Compose (GPU 버전) - 권장
```bash
# GPU 지원 Docker Compose 실행
docker-compose -f docker-compose.gpu.yml up --build

# 백그라운드 실행
docker-compose -f docker-compose.gpu.yml up -d --build

# 로그 확인
docker-compose -f docker-compose.gpu.yml logs -f

# 서비스 중지
docker-compose -f docker-compose.gpu.yml down
```

#### 수동 실행 (개발용)

##### 백엔드 서버 시작
```bash
# 챗봇 AI 서버
cd ollama-chatbot-api-ifro
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# 프록시 서버
cd django-react-backend-api-ifro/src
python manage.py runserver 8001
```

##### 프론트엔드 시작
```bash
cd django-react-frontend-ifro
npm install
npm start
```

---

## 📊 로깅 및 모니터링

### 로그 파일 위치
```
ollama-chatbot-api-ifro/logs/
├── chatbot_conversations.log    # 간단한 요약 로그
├── qa_detailed.log             # 상세한 질문/답변 로그
├── conversations.jsonl         # JSON 형식의 구조화된 로그
├── failed_answers.jsonl        # 실패한 답변 로그
└── llm_errors.log             # LLM 오류 로그
```

### 실시간 로그 확인
```bash
# 간단한 요약 로그
tail -f ollama-chatbot-api-ifro/logs/chatbot_conversations.log

# 상세한 질문/답변 로그
tail -f ollama-chatbot-api-ifro/logs/qa_detailed.log

# JSON 형식 로그
tail -f ollama-chatbot-api-ifro/logs/conversations.jsonl
```

### 로그 검색
```bash
# 특정 키워드가 포함된 질문 찾기
grep "교통사고" ollama-chatbot-api-ifro/logs/qa_detailed.log

# 신뢰도가 낮은 답변 찾기
grep "신뢰도: 0\.[0-5]" ollama-chatbot-api-ifro/logs/qa_detailed.log

# 오류 로그 확인
grep "ERROR" ollama-chatbot-api-ifro/logs/chatbot_conversations.log
```

### Docker Desktop에서 로그 확인
1. Docker Desktop 실행
2. Containers 탭에서 `chatbot-gpu` 컨테이너 선택
3. Logs 탭 클릭
4. 실시간으로 질문과 답변이 표시됩니다

---

## 🔧 기술 상세

### 핵심 파이프라인
```
질문 입력 → 질문 분석 → 하이브리드 검색 → RRF 병합 → 중복 제거 → 
점수 교정/필터링 → 리랭킹 → 컨텍스트 구성 → 가드레일 검사 → 
LLM 생성 → 답변 후처리 → 메트릭 수집
```

### 주요 모듈 구성

#### 질문 분석기 (analyzer.py)
- **기능**: 질문 유형 분류 및 검색 전략 결정
- **분류 유형**: 
  - numeric: 수치/단위 질문
  - definition: 정의/개념 질문
  - procedural: 절차/방법 질문
  - comparative: 비교 질문
  - system_info: 시스템 정보 질문
  - technical_spec: 기술 사양 질문
  - operational: 운영 관련 질문
  - problem: 문제 해결 질문

#### 검색기 (retriever.py)
- **HybridRetriever**: 벡터 + BM25 하이브리드 검색
- **병렬 처리**: ThreadPoolExecutor를 통한 동시 검색
- **캐싱**: LRU 캐시로 검색 결과 재사용
- **타임아웃**: 검색 시간 제한 (SEARCH_TIMEOUT_S)

#### 리랭커 (reranker.py)
- **CrossEncoder**: BAAI/bge-reranker-v2-m3 모델
- **경량 리랭커**: 문자 n-gram 오버랩 기반 폴백
- **타임아웃 보호**: RERANK_TIMEOUT_S 제한

#### LLM 인터페이스 (llm.py)
- **Ollama API**: HTTP 기반 로컬 LLM 호출
- **프롬프트 엔지니어링**: 교통 데이터 도메인 특화 가이드라인
- **재시도 메커니즘**: 백오프를 통한 안정성 확보
- **타임아웃**: LLM_TIMEOUT_S 제한

### 슬라이딩 윈도우 기반 청킹 전략

#### 기본 슬라이딩 윈도우
- **윈도우 크기**: 800자 (기본), 900자 (교통 데이터 특화)
- **오버랩**: 200자 (기본), 225자 (교통 데이터 특화, 25% 비율)
- **스텝 크기**: 윈도우 크기 - 오버랩 = 600자 (기본), 675자 (교통 데이터)
- **경계 스냅**: 문장/공백 경계에 ±5% 윈도우 조정으로 의미 단위 보존

#### 숫자 입력 시 양옆 청크 포함 전략
- **수치 질문 감지**: 정규표현식 기반 숫자/단위 패턴 인식
- **이웃 청크 확장**: 
  - 문단 단위: 같은 문단 내 최대 1개 이웃 청크 추가
  - 페이지 단위: 인접 페이지(±1)에서 최대 1개 청크 추가
- **수치 정보 보존**: 
  - 청크 경계에서 수치 데이터 손실 방지
  - 측정값 연속성 보장을 위한 오버랩 활용
  - 단위 변환 및 동의어 매핑 지원

### 품질 보장 기술

#### 점수 교정 (filtering.py)
- **소스별 정규화**: 벡터/BM25 점수 min-max 정규화
- **z-score 캡핑**: 이상치 제거
- **임계값 조정**: 질문 유형별 동적 임계값

#### 가드레일 (guardrail.py)
- **오버랩 검사**: 질문-컨텍스트 유사도 (≥0.12)
- **핵심 토큰**: 도메인 키워드 매칭 (≥1개)
- **폴백 메커니즘**: 저신뢰도 재검색 → 단일 스팬 → 표준 부정응답

#### 수치 검증 (measurements.py)
- **단위 동의어**: mg/L ↔ ppm, ug/L ↔ ppb 등 교통 데이터 특화 단위 변환
- **허용오차**: ±5% 수치 정확도 검증
- **측정값 추출**: 정규표현식 기반 수치/단위 추출

---

## 🐛 문제 해결

### 자주 발생하는 문제

#### 콘솔 한글 깨짐
```bash
# PowerShell
chcp 65001

# 또는 환경변수 설정
set PYTHONUTF8=1
```

#### 벡터 인덱스 미사용
- `--backend auto` (TF‑IDF 폴백)로 실행 가능

#### PDF가 이미지 스캔본
- OCR 전처리 필요 (텍스트 추출 불가 시 코퍼스 비어 있음)
- 자동화됨: quickstart/build 스크립트가 `--ocr auto`일 때 Tesseract(PyMuPDF 렌더)→ocrmypdf 순으로 폴백 시도

#### Windows에서 자동 OCR 활용
1. Tesseract OCR 설치 (권장: UB Mannheim 배포본, kor 언어팩 포함)
2. `pip install pytesseract pillow pymupdf`
3. 재실행: `python ollama-chatbot-api-ifro/scripts/autorun.py --pdf ollama-chatbot-api-ifro/data/tests --backend auto --ocr auto`

#### Ollama 미기동
- LLM이 비어도 추출형 폴백으로 응답 (성능 저하 가능)

#### 코퍼스가 0 chunks로 생성됨
- 스캔 PDF 가능성 높음 → `--pdf-extractor fitz` 또는 OCR 후 재시도

### 로그가 보이지 않는 경우
1. 컨테이너가 실행 중인지 확인
2. 로그 볼륨이 올바르게 마운트되었는지 확인
3. 권한 문제가 없는지 확인

### 로그 파일이 너무 큰 경우
```bash
# 로그 파일 크기 확인
ls -lh ollama-chatbot-api-ifro/logs/

# 오래된 로그 파일 삭제
rm ollama-chatbot-api-ifro/logs/chatbot_conversations.log.old
```

---

## 📈 성능 최적화

### 병렬 처리
- **검색 병렬화**: 벡터/BM25 동시 실행
- **배치 임베딩**: 다중 텍스트 동시 처리
- **비동기 처리**: FastAPI 기반 비동기 API

### 캐싱 전략
- **검색 캐시**: LRU 기반 검색 결과 캐싱
- **임베딩 캐시**: 벡터 저장소 사전 구축
- **설정 해시**: config_hash 기반 캐시 무효화

### 메모리 관리
- **인덱스 압축**: FAISS/HNSW 압축 인덱스
- **배치 크기 조정**: 메모리 사용량 최적화
- **가비지 컬렉션**: 명시적 메모리 해제

---

## 🔒 보안 및 거버넌스

### 기술적 한계
- **RAG 한계**: 문맥 미회수 시 잔여 환각, 임베딩·벡터 검색·재랭킹 추가로 지연·병목, 도메인 특화 용어 검색 부정확 가능
- **로컬 LLM 운용 리스크**: 초기 GPU/스토리지/전력 비용, 운영 복잡성, 확장성·가용성 저하 위험
- **보안 이슈**: 데이터 분할·접근제어 필수, 프롬프트 인젝션 대비, 벡터DB 암호화·키 관리 필요

### 기술 보완 방안
- **검색 정확도**: BM25+벡터 하이브리드 고도화, 지식 그래프 보강 재랭킹, 정기 임베딩 재학습과 품질 점검
- **모델·배치 전략**: 3B~7B 경량 로컬 모델 우선, 고난도 쿼리는 클라우드 대형 모델 부분 위임(하이브리드)
- **보안·거버넌스**: 벡터DB 암호화, RBAC·접근로그 모니터링, 프롬프트 필터·정책 적용

---

## 📞 지원 및 문의

시스템 사용 중 문제가 발생하거나 추가 기능이 필요한 경우, 로그 파일을 확인하고 위의 문제 해결 가이드를 참조하시기 바랍니다.

### 서비스 접속 정보
- **프론트엔드**: http://localhost:3000
- **백엔드 프록시**: http://localhost:8001
- **챗봇 AI 서버**: http://localhost:8000
- **Ollama 서버**: http://localhost:11434

### 주요 엔드포인트
- **챗봇 API**: `POST /api/ask {question, mode, k}`
- **배치 처리**: `POST /api/qa/batch {items:[{id,question}], mode}`
- **상태 확인**: `GET /healthz`
- **메트릭**: `GET /metrics` (Prometheus 형식)

---

*이 문서는 WILL 챗봇 시스템의 통합 가이드입니다. 각 섹션을 참조하여 시스템을 효과적으로 활용하시기 바랍니다.*