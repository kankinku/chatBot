# 교통 데이터 챗봇 시스템

교통 데이터 도메인에 특화된 PDF 기반 질문-답변 챗봇입니다. RAG(Retrieval-Augmented Generation) 기술을 활용하여 교통 관련 문서에서 정확한 정보를 검색하고 답변을 생성합니다.

## 🚀 빠른 시작

### 1분 원클릭 실행
```bash
# 자동 설치 및 실행
python scripts/autorun.py --backend auto

# 특정 질문 테스트
python scripts/autorun.py --backend auto --question "교통 데이터의 주요 지표는 무엇인가요?"
```

### 수동 설치 및 실행
```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. PDF에서 코퍼스 생성
python scripts/build_corpus_from_pdfs.py --pdf_dir data/pdfs --out data/corpus_v1.jsonl

# 3. 벡터 인덱스 구축
python scripts/build_vector_index.py --corpus data/corpus_v1.jsonl --backend faiss

# 4. 질문-답변 테스트
python scripts/manual_cli.py --corpus data/corpus_v1.jsonl --question "교통 데이터의 주요 지표는 무엇인가요?"
```

## 📋 목차

- [주요 특징](#주요-특징)
- [시스템 아키텍처](#시스템-아키텍처)
- [설치 가이드](#설치-가이드)
- [사용법](#사용법)
- [API 서버](#api-서버)
- [로깅 시스템](#로깅-시스템)
- [프로젝트 구조](#프로젝트-구조)
- [성능 벤치마크](#성능-벤치마크)
- [문제 해결](#문제-해결)
- [기술 문서](#기술-문서)

## 🎯 주요 특징

### 정확도 우선 설계
- **하이브리드 검색**: BM25 키워드 검색 + 벡터 의미 검색 결합
- **도메인 특화**: 교통 데이터 전문 용어 및 지표 이해
- **품질 보장**: 다층 가드레일 시스템으로 신뢰성 확보

### 한국어 최적화
- **한국어 임베딩**: ko-sroberta-multitask 모델 사용
- **한국어 LLM**: Llama 3 8B 한국어 특화 프롬프트
- **OCR 지원**: 한국어 문서 자동 인식 및 처리

### 실용적 기능
- **자동화**: 원클릭 설치 및 실행
- **다양한 인터페이스**: CLI, API, 웹 서버 지원
- **성능 모니터링**: 상세한 메트릭 및 리포트 제공

## 🏗️ 시스템 아키텍처

```
질문 입력 → 질문 분석 → 하이브리드 검색 → 점수 교정 → 
리랭킹 → 컨텍스트 구성 → 가드레일 검사 → LLM 생성 → 답변 출력
```

### 핵심 컴포넌트
- **질문 분석기**: 질문 유형 분류 및 검색 전략 결정
- **하이브리드 검색**: 벡터 + BM25 검색 결과 병합
- **리랭커**: CrossEncoder 기반 결과 재정렬
- **가드레일**: 답변 품질 검증 및 폴백 처리
- **LLM 인터페이스**: Ollama 기반 로컬 LLM 호출

## 💻 설치 가이드

### 시스템 요구사항
- Python 3.8+
- Windows 10/11 (권장)
- 최소 8GB RAM
- 2GB 이상 디스크 공간

### 단계별 설치

#### 1. 저장소 클론
```bash
git clone <repository-url>
cd 정수장\챗봇-vf
```

#### 2. Python 환경 설정
```bash
# pip 업데이트
pip install --upgrade pip

# PyTorch CPU 설치 (Windows 최적화)
pip install --upgrade --force-reinstall "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cpu

# 의존성 설치
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

#### 3. Ollama 설치 (LLM 서버)
```bash
# Ollama 다운로드 및 설치
# https://ollama.ai/download

# Llama 3 모델 다운로드
ollama pull llama3:8b-instruct-q4_K_M
```

#### 4. OCR 엔진 설치 (선택사항)
```bash
# Tesseract 설치 (한국어 OCR)
# https://github.com/UB-Mannheim/tesseract/wiki
```

## 🎮 사용법

### 자동 실행 (권장)
```bash
# 완전 자동 실행
python scripts/autorun.py --backend auto

# 특정 질문만 실행
python scripts/autorun.py --backend auto --question "교통 데이터의 주요 지표는 무엇인가요?"

# 자동 평가 실행
python scripts/autorun.py --auto
```

### 수동 실행
```bash
# 1. PDF 처리
python scripts/build_corpus_from_pdfs.py --pdf_dir data/pdfs --out data/corpus_v1.jsonl --pdf-extractor auto --ocr auto

# 2. 벡터 인덱스 구축
python scripts/build_vector_index.py --corpus data/corpus_v1.jsonl --backend faiss --outdir vector_store

# 3. CLI 인터페이스
python scripts/manual_cli.py --corpus data/corpus_v1.jsonl --mode accuracy --store-backend auto --question "질문을 입력하세요"

# 4. 벤치마크 실행
python scripts/run_qa_benchmark.py --input data/tests/qa.json --corpus data/corpus_v1.jsonl --mode accuracy --report out/report.json
```

### 주요 옵션
- `--backend`: 벡터 저장소 선택 (auto/faiss/hnsw)
- `--pdf-extractor`: PDF 추출기 (auto/plumber/fitz)
- `--ocr`: OCR 사용 여부 (auto/always/off)
- `--use-cross-reranker`: CrossEncoder 리랭킹 활성화
- `--thr-base`: 기본 임계값 (기본값: 0.25)

## 🌐 API 서버

### 서버 실행
```bash
# 로컬 서버 시작
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# 또는 quickstart 사용
python scripts/quickstart.py --pdf data/pdfs --backend auto --server --host 0.0.0.0 --port 8000
```

### API 엔드포인트

#### 단일 질문
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "교통 데이터의 주요 지표는 무엇인가요?",
    "mode": "accuracy",
    "k": 6
  }'
```

#### 배치 질문
```bash
curl -X POST "http://localhost:8000/api/qa/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"id": "1", "question": "교통 데이터의 주요 지표는 무엇인가요?"},
      {"id": "2", "question": "교통량 측정 방법은 무엇인가요?"}
    ],
    "mode": "accuracy"
  }'
```

#### 헬스체크
```bash
curl http://localhost:8000/healthz
```

## 📊 로깅 시스템

### Docker Desktop에서 로그 확인

챗봇 GPU 버전에서는 질문과 답변에 대한 상세한 로그를 Docker Desktop에서 실시간으로 확인할 수 있습니다.

#### 실시간 로그 확인 방법
1. **Docker Desktop 실행**
2. **Containers 탭**에서 `chatbot-gpu` 컨테이너 선택
3. **Logs 탭** 클릭하여 실시간 로그 확인

#### 로그 형식 예시
```
2024-01-15 14:30:25 [INFO] 📥 질문 수신 | 모드: accuracy | 길이: 25자
2024-01-15 14:30:25 [INFO] 📝 질문 내용: 교통사고가 발생했을 때 어떻게 해야 하나요?
2024-01-15 14:30:28 [INFO] 📤 답변 생성 완료 | 신뢰도: 0.85 | 소스: 3개 | Fallback: False
2024-01-15 14:30:28 [INFO] 📄 답변 내용: 교통사고 발생 시 다음과 같이 대응하세요...
2024-01-15 14:30:28 [INFO] 💬 Q&A 완료 | 질문: 교통사고가 발생했을 때 어떻게 해야 하나요? | 답변길이: 156 | 신뢰도: 0.85
```

#### 로그 파일 위치
```
ollama-chatbot-api-ifro/logs/
├── chatbot_conversations.log    # 간단한 요약 로그
├── qa_detailed.log             # 상세한 질문/답변 로그
├── conversations.jsonl         # JSON 형식의 구조화된 로그
├── failed_answers.jsonl        # 실패한 답변 로그
└── llm_errors.log             # LLM 오류 로그
```

#### 로깅 테스트
```bash
# 로깅 시스템 테스트
python test_logging.py

# 실시간 로그 확인
tail -f logs/chatbot_conversations.log
tail -f logs/qa_detailed.log
```

자세한 로깅 가이드는 [LOGGING_GUIDE.md](LOGGING_GUIDE.md)를 참조하세요.

## 📁 프로젝트 구조

```
교통 데이터 챗봇/
├── 📁 data/                    # 데이터 저장소
│   ├── 📁 pdfs/               # 원본 PDF 문서
│   ├── 📁 tests/              # 테스트 데이터
│   ├── 📄 corpus_v1.jsonl     # 메인 코퍼스
│   └── 📄 domain_dictionary.json  # 도메인 사전
├── 📁 scripts/                # 실행 스크립트
│   ├── 🚀 autorun.py          # 자동 실행
│   ├── 🔧 build_corpus_from_pdfs.py  # PDF 처리
│   ├── 🔧 build_vector_index.py     # 인덱스 구축
│   ├── 🚀 manual_cli.py       # CLI 인터페이스
│   └── 🚀 run_qa_benchmark.py # 벤치마크
├── 📁 src/unifiedpdf/         # 핵심 라이브러리
│   ├── 🏗️ facade.py          # 메인 파이프라인
│   ├── 🔍 retriever.py       # 검색 엔진
│   ├── 🔍 filtering.py       # 필터링
│   ├── 🔍 reranker.py        # 리랭킹
│   ├── 🤖 llm.py             # LLM 인터페이스
│   └── 🛡️ guardrail.py       # 가드레일
├── 📁 vector_store/           # 벡터 저장소
├── 📁 server/                 # 웹 서버
├── 📁 out/                    # 결과 출력
└── 📁 logs/                   # 로그 파일
```

## 📊 성능 벤치마크

### 벤치마크 실행
```bash
# 기본 벤치마크
python scripts/run_qa_benchmark.py --input data/tests/qa.json --corpus data/corpus_v1.jsonl --mode accuracy

# 상세 리포트 생성
python scripts/run_qa_benchmark.py --input data/tests/qa.json --corpus data/corpus_v1.jsonl --mode accuracy --report out/report.json --csv out/report.csv
```

### 성능 지표
- **정확도**: 답변의 정확성 (키워드/수치/단위 가중치)
- **신뢰도**: 컨텍스트 신뢰도 + 가드레일 오버랩
- **처리 시간**: 검색 + 생성 시간
- **수치 보존**: 컨텍스트-답변 간 수치 일치도

## 🔧 문제 해결

### 자주 발생하는 문제

#### 1. 한글 깨짐 현상
```bash
# PowerShell에서 실행
chcp 65001

# 또는 환경변수 설정
set PYTHONUTF8=1
```

#### 2. 코퍼스가 0건으로 나옴
```bash
# 스캔 PDF인 경우 OCR 활성화
python scripts/build_corpus_from_pdfs.py --pdf_dir data/pdfs --out data/corpus_v1.jsonl --pdf-extractor fitz --ocr always

# Tesseract 설치 확인
tesseract --version
```

#### 3. FAISS 설치 오류
```bash
# CPU 전용 설치
pip install faiss-cpu

# 또는 TF-IDF 폴백 사용
python scripts/autorun.py --backend auto
```

#### 4. Ollama 연결 오류
```bash
# Ollama 서버 상태 확인
curl http://127.0.0.1:11434/api/tags

# 모델 다운로드 확인
ollama list
```

### 로그 확인
```bash
# LLM 오류 로그
cat logs/llm_errors.log

# 상세 디버그 모드
python scripts/manual_cli.py --corpus data/corpus_v1.jsonl --question "질문" --debug
```

## 📚 기술 문서

- **[튜토리얼.md](튜토리얼.md)**: 단계별 사용 가이드
- **[정수장_챗봇_기술보고서.md](정수장_챗봇_기술보고서.md)**: 상세 기술 문서
- **[프로젝트_구조_통합문서.md](프로젝트_구조_통합문서.md)**: 프로젝트 구조 분석

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원

문제가 발생하거나 질문이 있으시면 다음을 통해 연락해 주세요:
- 이슈 트래커: [GitHub Issues](링크)
- 이메일: [이메일 주소]

---

**교통 데이터 챗봇 시스템** - 정확하고 신뢰할 수 있는 교통 데이터 정보 제공