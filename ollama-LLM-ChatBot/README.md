# 🤖 IFRO_SEJONG 범용 RAG 시스템 (GPU 지원)

## 📋 프로젝트 개요

IFRO_SEJONG 범용 RAG 시스템은 문서 검색과 데이터베이스 쿼리를 통합한 지능형 대화형 AI 시스템입니다. Dual Pipeline 아키텍처를 통해 기능 질문(메뉴얼, 설정, 운용 방법)과 법률 관련 질문은 PDF 파이프라인으로, 정량적 데이터 질문은 SQL 파이프라인으로 분류하여 하이브리드 답변을 생성합니다.

### 🚀 새로운 기능
- **CUDA GPU 지원**: NVIDIA GPU를 사용한 가속 처리
- **CPU 폴백**: GPU가 없거나 사용할 수 없는 경우 자동으로 CPU 사용
- **지연 로딩 제거**: 모든 모델을 즉시 로드하여 빠른 응답
- **메모리 최적화**: GPU/CPU 메모리 효율적 사용

## 🏗️ 시스템 아키텍처

### Dual Pipeline 분류 시스템

```
사용자 질문
    ↓
SBERT 기반 분류
    ↓
┌─────────────────┬─────────────────┐
│   PDF 파이프라인  │   SQL 파이프라인  │
│   (기능/법률)    │   (정량적 데이터) │
└─────────────────┴─────────────────┘
    ↓                    ↓
기능 질문 처리          데이터베이스 쿼리
- 메뉴얼 검색          - 통계 분석
- 설정 방법            - 집계 데이터
- 운용 방법            - 수치 조회
- 법률 관련            - 비교 분석
- 정책 지침            - 추세 분석
```
ollama-LLM-ChatBot/
├── 🚀 run_server.py              # 메인 서버 실행 파일
├── 🐳 Dockerfile                 # Docker 컨테이너 설정
├── 🔧 docker-entrypoint.sh       # Docker 초기화 스크립트
├── 📦 requirements.txt            # Python 의존성
├── 📚 README.md                  # 프로젝트 문서
├── 🔒 .dockerignore              # Docker 빌드 제외 파일
├── 🧠 core/                      # 핵심 처리 모듈
│   ├── __init__.py
│   ├── cache/                    # 캐싱 시스템
│   ├── database/                 # 데이터베이스 처리
│   ├── document/                 # 문서 처리
│   ├── llm/                      # LLM 통합 (KorT5-Small 기반)
│   ├── movement/                 # 데이터 이동 처리
│   └── query/                    # 질의 처리
├── 🌐 api/                       # API 엔드포인트
│   ├── __init__.py
│   ├── endpoints.py              # FastAPI 엔드포인트
│   ├── django_client.py          # Django 연동 클라이언트
│   └── typescript_client.ts      # TypeScript 클라이언트
├── 📊 data/                      # 데이터 및 벡터 저장소
│   ├── pdfs/                     # PDF 문서 저장소
│   ├── conversation_history.db    # 대화 기록 데이터베이스
│   └── intent_training_dataset.json # 의도 분석 데이터셋
├── 🗄️ vector_store/              # 벡터 저장소
│   ├── chroma/                   # ChromaDB 벡터 저장소
│   └── faiss/                    # FAISS 벡터 저장소
├── 🤖 models/                    # AI 모델 저장소
├── 📝 logs/                      # 시스템 로그
│   ├── chatbot_detailed.log      # 상세 로그
│   ├── chatbot_summary.log       # 요약 로그
│   ├── step_processing.log       # 단계별 처리 로그
│   ├── sql_queries.log          # SQL 쿼리 로그
│   └── pdf_queries.log          # PDF 질문 로그
└── 🧪 setup_sbert.py             # SBERT 모델 설정
```

## 🚀 빠른 시작

### 1. Docker를 통한 실행 (권장)

#### GPU 지원으로 실행
```bash
# 전체 시스템과 함께 실행 (GPU 지원)
cd ../
docker-compose up -d chatbot

# GPU 상태 확인
docker exec ifro-chatbot nvidia-smi

# GPU 지원 확인
docker exec ifro-chatbot python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### CPU로 실행 (GPU 없음)
```bash
# CPU 전용으로 실행
docker-compose up -d chatbot

# CPU 사용 확인
docker exec ifro-chatbot python -c "import torch; print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```

### 2. 로컬 개발 환경

#### GPU 지원 설치
```bash
# CUDA 지원 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 나머지 의존성 설치
pip install -r requirements.txt

# SBERT 모델 설정
python setup_sbert.py

# GPU 지원으로 실행
python run_server.py
```

#### CPU 전용 설치
```bash
# CPU 전용 PyTorch 설치
pip install torch torchvision torchaudio

# 나머지 의존성 설치
pip install -r requirements.txt

# SBERT 모델 설정
python setup_sbert.py

# CPU로 실행
python run_server.py
```

## 🔧 GPU 설정 가이드

GPU 사용을 위한 자세한 설정 가이드는 [GPU_SETUP.md](./GPU_SETUP.md)를 참조하세요.

### 주요 GPU 설정 단계:
1. **CUDA 드라이버 설치**
2. **NVIDIA Container Toolkit 설치** (Docker 사용 시)
3. **GPU 지원 PyTorch 설치**
4. **환경 변수 설정**

## 📊 성능 비교

| 설정 | 응답 시간 | 메모리 사용량 | 정확도 |
|------|-----------|---------------|--------|
| GPU (float16) | ~2초 | 8GB | 높음 |
| GPU (float32) | ~3초 | 16GB | 매우 높음 |
| CPU | ~10초 | 4GB | 높음 |

### 3. 서비스 접속

- **API 서버**: http://localhost:8008
- **API 문서**: http://localhost:8008/docs
- **헬스체크**: http://localhost:8008/health

## 🏛️ 핵심 기능

### 📄 PDF 문서 처리 (기능/법률 질문)
- **다양한 형식 지원**: PyPDF2, PyMuPDF, pdfplumber 통합
- **자동 텍스트 추출**: 구조화된 텍스트 및 메타데이터 추출
- **벡터 임베딩**: Sentence Transformers 기반 의미 분석
- **자동 업로드**: 서버 시작 시 data 폴더의 PDF 파일들이 자동으로 벡터 저장소에 업로드됩니다
- **기능 질문 처리**: 메뉴얼, 설정 방법, 운용 방법, 사용법 등
- **법률 관련 처리**: 법률, 규정, 정책, 지침, 가이드라인 등

### 🔍 지능형 검색
- **하이브리드 검색**: 키워드 + 의미 기반 검색
- **벡터 저장소**: ChromaDB와 FAISS 통합
- **실시간 인덱싱**: 문서 업로드 시 자동 벡터화

### 💬 대화형 인터페이스
- **컨텍스트 유지**: 이전 대화 기반 연속성
- **의도 분석**: 질문 유형 자동 분류
- **개인화**: 사용자별 대화 기록 관리
- **AI 답변 생성**: KorT5-Small 기반 자연스러운 텍스트 생성
- **LLM 기반 인사말**: 자연스러운 대화형 인사말 생성
- **회사 정보 설정**: 환경변수 기반 회사 정보 관리

### 🗄️ SQL 데이터 통합 (정량적 데이터 질문)
- **자동 SQL 생성**: 자연어를 SQL로 변환
- **스키마 인식**: 데이터베이스 구조 자동 파악
- **실시간 실행**: 생성된 SQL 즉시 실행
- **정량적 분석**: 통계, 집계, 분석, 수치 데이터 처리

## 🔧 API 엔드포인트

### 📋 기본 엔드포인트
- `GET /` - 서버 상태 확인
- `GET /docs` - Swagger UI API 문서
- `GET /health` - 헬스 체크

### 📄 PDF 관리
- `POST /upload-pdf` - PDF 파일 업로드
- `GET /pdfs` - 등록된 PDF 목록
- `DELETE /pdfs/{pdf_id}` - PDF 삭제

### 💬 질의응답
- `POST /ask` - 일반 질문
- `POST /ask-with-context` - 컨텍스트 기반 질문
- `GET /conversation-history` - 대화 기록



### 🗄️ 데이터베이스
- `POST /sql-query` - SQL 질의 실행
- `GET /database-schema` - 데이터베이스 스키마
- `POST /analyze-data` - 데이터 분석 요청

## ⚙️ 환경 변수

```bash
# 모델 설정
MODEL_TYPE=local                    # local/ollama/huggingface
MODEL_NAME=koelectra-small-v3      # 사용할 모델명
EMBEDDING_MODEL=ko-sroberta        # 임베딩 모델

# PDF 자동 업로드 설정
AUTO_UPLOAD_PDFS=true              # 서버 시작 시 PDF 자동 업로드 (항상 활성화)

# 데이터베이스 설정
MYSQL_DATABASE=traffic
MYSQL_USER=root
MYSQL_PASSWORD=1234
MYSQL_HOST=db
MYSQL_PORT=3306

# 시스템 설정
PYTHONPATH=/app
```

## 🚀 청크 초기화 및 재생성

새로운 모델로 변경했을 때 기존 청크를 초기화하고 새로운 모델에 맞는 청크를 재생성할 수 있습니다.

### 명령줄에서 청크 초기화

```bash
# 로컬 실행 시 청크 초기화
python main.py --reset-chunks
```

### Docker에서 청크 초기화

```bash
# 환경변수로 청크 초기화 활성화
docker-compose up -d --build

# 또는 환경변수 설정
export RESET_CHUNKS_ON_START=true
docker-compose up -d
```

### API를 통한 청크 초기화

```bash
# 청크 초기화 및 재생성
curl -X POST http://localhost:8008/reset-chunks

# 벡터 저장소만 초기화 (청크 재생성 없음)
curl -X POST http://localhost:8008/clear-vector-store
```

### 청크 초기화 과정

1. **기존 청크 초기화**: 모든 기존 청크 삭제
2. **PDF 파일 스캔**: `data/pdfs` 폴더의 PDF 파일들 검색
3. **청크 재생성**: 새로운 모델로 청크 생성 및 벡터 저장소에 추가

### 환경변수 설정

```bash
# docker-compose.yml
environment:
  - RESET_CHUNKS_ON_START=false  # true로 설정하면 시작 시 자동 초기화
  - AUTO_UPLOAD_PDFS=true        # PDF 자동 업로드
```

## 🧠 AI 모델

### 지원하는 모델 타입

1. **로컬 모델 (권장)**
   - **KorT5-Small**: 한국어 텍스트 생성 및 답변
   - **Ko-SRoBERTa**: 한국어 의미 임베딩
   - **SQLCoder**: SQL 생성 특화
   - **장점**: 빠른 응답, 오프라인 작동, 데이터 보안

2. **Hugging Face 모델**
   - **장점**: 최신 모델, 커스터마이징 가능
   - **단점**: 높은 리소스 요구사항

## 📊 성능 최적화

### 응답 시간
- **캐시된 질문**: 0.1-0.5초
- **일반 질문**: 1-3초
- **복잡한 분석**: 5-10초

### 메모리 사용량
- **기본 시스템**: 2-3GB RAM
- **모델 로딩**: 1-2GB RAM
- **벡터 저장소**: 0.5-1GB RAM

### 스토리지 요구사항
- **시스템**: 2-3GB
- **모델**: 3-5GB
- **데이터**: 사용량에 따라 증가

## 🛠️ 개발 가이드

### 모듈 구조

#### Core 모듈
- **`cache/`**: 인메모리 및 영구 캐싱
- **`database/`**: MySQL 연결 및 쿼리 실행
- **`document/`**: PDF 처리 및 텍스트 분석
- **`llm/`**: AI 모델 통합 및 관리
- **`query/`**: 질의 처리 및 라우팅

#### API 모듈
- **`endpoints.py`**: FastAPI 엔드포인트 정의
- **`django_client.py`**: Django 백엔드 연동
- **`typescript_client.ts`**: 프론트엔드 연동

### 확장 방법

1. **새로운 모델 추가**
   ```python
   # core/llm/에 새 모델 클래스 생성
   class NewModel(LLMInterface):
       def generate_response(self, prompt: str) -> str:
           # 구현
           pass
   ```

2. **새로운 검색 방법 추가**
   ```python
   # core/query/에 새 검색 클래스 생성
   class NewSearch(SearchInterface):
       def search(self, query: str) -> List[Document]:
           # 구현
           pass
   ```

## 🐛 문제 해결

### 일반적인 문제들

1. **모델 로딩 실패**
   ```bash
   # 모델 다운로드 확인
   python setup_sbert.py
   
   # 캐시 정리
   rm -rf ~/.cache/huggingface/
   ```

2. **메모리 부족**
   ```bash
   # Docker 메모리 제한 확인
   docker stats chatbot
   
   # 시스템 메모리 확인
   free -h
   ```

3. **벡터 저장소 오류**
   ```bash
   # ChromaDB 재설정
   rm -rf vector_store/chroma/
   python run_server.py
   ```

### 로그 확인

```bash
# 실시간 로그
docker-compose logs -f chatbot

# 특정 시간 로그
docker-compose logs --since="2024-01-01T00:00:00" chatbot

# 로그 파일 직접 확인
tail -f logs/chatbot_detailed.log

# 단계별 처리 로그 확인 (새로운 기능)
python check_logs.py summary          # 최근 세션 요약
python check_logs.py steps            # 단계별 상세 정보
python check_logs.py performance      # 성능 분석
python check_logs.py search "교통량"  # 키워드 검색

# 로그 뷰어 사용법
python check_logs.py --help
```

## 📊 로그 분석 및 모니터링

### 단계별 처리 로그 시스템

챗봇의 질문처리 과정을 단계별로 추적할 수 있는 새로운 로깅 시스템이 추가되었습니다.

#### 로그 파일 구조
- **`step_processing.log`**: 단계별 처리 로그 (새로운 기능)
- **`chatbot_detailed.log`**: 상세 JSON 로그
- **`chatbot_summary.log`**: 요약 로그
- **`sql_queries.log`**: SQL 쿼리 전용 로그
- **`pdf_queries.log`**: PDF 질문 전용 로그

#### 처리 단계
```
시작 → SBERT로 질문처리 → PDF파이프라인 시작 → 질문분석 → 벡터검색 → 답변생성 → 완료
시작 → SBERT로 질문처리 → SQL파이프라인 시작 → SQL생성 → 데이터베이스실행 → 답변생성 → 완료
시작 → SBERT로 질문처리 → 인사말처리 시작 → 완료
```

#### 로그 뷰어 사용법

```bash
# 기본 사용법
python check_logs.py [명령어] [옵션]

# 최근 세션 요약 (기본값)
python check_logs.py summary

# 단계별 상세 정보 포함
python check_logs.py steps

# 성능 분석
python check_logs.py performance

# 키워드 검색
python check_logs.py search "교통량"

# 옵션 사용
python check_logs.py summary --hours 48    # 48시간 범위
python check_logs.py steps --log-dir logs  # 다른 로그 디렉토리
```

#### 로그 출력 예시

```
📊 최근 24시간 세션 요약 (15개)
================================================================================
총 세션: 15
성공: 14 (93.3%)
실패: 1 (6.7%)
평균 처리 시간: 2.345초 (최소: 0.123초, 최대: 8.567초)

파이프라인별 분포:
  PDF: 10개 (66.7%)
  SQL: 4개 (26.7%)
  인사말: 1개 (6.7%)

📋 최근 세션들:
================================================================================
세션 ID: session_20241201_1430_0001
시작 시간: 2024-12-01 14:30:15
종료 시간: 2024-12-01 14:30:18
총 처리 시간: 3.245초
파이프라인: PDF
상태: ✅ 성공
질문: 교통량 분석 방법은?

단계별 처리:
   1. ✅ 시작: 0.000초
   2. ✅ SBERT로 질문처리: 0.234초
   3. ✅ PDF파이프라인 시작: 0.000초
   4. ✅ 질문분석: 0.456초
   5. ✅ 벡터검색: 0.789초
   6. ✅ 답변생성: 1.766초
   7. ✅ 완료: 3.245초
================================================================================
```

### 성능 분석

```bash
# 단계별 평균 처리 시간 분석
python check_logs.py performance

# 출력 예시:
📈 성능 분석 (최근 24시간)
================================================================================
단계별 평균 처리 시간:
  답변생성: 1.234초 (최소: 0.567초, 최대: 3.456초)
  벡터검색: 0.789초 (최소: 0.123초, 최대: 2.345초)
  SBERT로 질문처리: 0.234초 (최소: 0.123초, 최대: 0.567초)
  질문분석: 0.456초 (최소: 0.234초, 최대: 1.234초)
  SQL생성: 0.345초 (최소: 0.123초, 최대: 0.789초)

파이프라인별 평균 처리 시간:
  PDF: 2.345초 (10개 세션)
  SQL: 1.567초 (4개 세션)
  인사말: 0.123초 (1개 세션)
```

### 시스템 모니터링

#### 헬스체크
```bash
# 서비스 상태 확인
curl http://localhost:8008/health

# 상세 상태 확인
curl http://localhost:8008/system-status
```

#### 성능 메트릭
- **응답 시간**: 평균, 95th percentile
- **처리량**: 초당 요청 수
- **오류율**: 실패한 요청 비율
- **리소스 사용량**: CPU, 메모리, 디스크

## 🤖 AI 모델 정보

### 사용 중인 모델들
- **답변 생성**: `paust/pko-t5-small` (KorT5-Small)
  - 한국어 텍스트 생성에 최적화
  - 77M 파라미터로 경량화
  - 컨텍스트 이해 기반 자연스러운 답변 생성
  
- **임베딩/라우팅**: `jhgan/ko-sroberta-multitask` (SBERT)
  - 한국어 의미 분석에 특화
  - 질문 분류 및 라우팅에 사용
  
- **SQL 생성**: `defog/sqlcoder-7b` (SQLCoder)
  - SQL 전용 모델로 정확한 쿼리 생성
  - 데이터베이스 스키마 인식

### 모델 성능 특징
- **답변 품질**: 템플릿 기반 → AI 생성으로 향상
- **응답 속도**: 1-3초 (기존 0.1초에서 증가)
- **메모리 사용량**: 약 200-300MB (기존 100MB에서 증가)

## 🔒 보안 고려사항

### 데이터 보안
- **로컬 처리**: 민감한 데이터는 로컬에서만 처리
- **암호화**: 전송 및 저장 시 데이터 암호화
- **접근 제어**: API 키 기반 인증

### 모델 보안
- **신뢰할 수 있는 모델**: 검증된 오픈소스 모델만 사용
- **정기 업데이트**: 보안 패치 및 모델 업데이트
- **취약점 스캔**: 정기적인 보안 검사

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **문서**: `/docs` 폴더의 상세 문서
- **로그**: `logs/` 폴더의 시스템 로그

---

**개발팀**: IFRO_SEJONG Team  
**최종 업데이트**: 2024년 12월  
**버전**: 2.0.0