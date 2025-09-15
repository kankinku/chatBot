# UnifiedPDFPipeline (Windows 친화 PDF RAG)

정확도 1순위·속도 2순위·최적화 3순위를 지향하는 단일 파이프라인입니다. 로컬 자동 테스트(하네스)·로컬 수동 CLI·서버(API)가 동일 Facade 경로를 사용합니다. Windows/한글/비‑POSIX 환경에서 안전하게 동작하도록 타임아웃·경로·인코딩을 고려했습니다.

## 목차
- 개요/특징
- 디렉터리 구조
- 설치(Windows/CPU 권장)
- 가장 쉬운 실행(autorun)
- 1분 원클릭 실행(quickstart)
- 수동 절차
- 오프라인 비교 러너
- API 서버
- Docker 실행
- 주요 구현 포인트
- FAQ/문제 해결
- 추가 문서

## 개요/특징
- 단일 실행 경로: Facade를 중심으로 하이브리드 검색 → RRF → 디듀프 → 점수 교정 → 동적 k/이웃 → [옵션] 리랭커 → LLM → 가드/폴백 → 메트릭.
- 하이브리드 검색: BM25(3–5 char n‑gram) + 벡터(TF‑IDF/FAISS/HNSW) → RRF(유형별 가중).
- 점수 교정/임계: 소스별 min–max → z‑score 캡핑 → [0,1], 임계 0.25(수치 0.16/장문 0.17 오버라이드).
- 동적 k/이웃: 질문 유형·길이에 따라 4–10, 문단/섹션/±1page 이웃 추가 후 최종 재디듀프.
- 리랭커: 경량/크로스엔코더(BAAI/bge‑reranker‑v2‑m3) + 타임아웃/예산 가드, OFF→ON A/B 가능.
- 가드/폴백: (overlap≥0.12 AND 핵심토큰≥1), 저신뢰 재탐색→단일 스팬→표준 부정응답.
- 수치/단위 정합: mg/L↔ppm 동의어, ±5% 허용오차 검증.
- 관측/재현: 질문별 메트릭과 `config_hash` 포함.

## 디렉터리 구조
- `src/unifiedpdf`: facade, retriever, merger, filtering, context, guardrail, analyzer, llm, config, utils, timeouts, vector_store, reranker, measurements
- `scripts/quickstart.py`: 원클릭(코퍼스/인덱스 → 질문/CLI/벤치마크/서버)
- `scripts/autorun.py`: 자동 설치 확인 + PDF/코퍼스 탐색 + quickstart 호출
- `scripts/run_qa_benchmark.py`: 하네스(정확도/지표 리포트 JSON/CSV)
- `scripts/manual_cli.py`: 수동 CLI(단일 질문/대화형)
- `scripts/build_corpus_from_pdfs.py`: PDF→코퍼스(JSONL) 생성(텍스트 추출+OCR 폴백)
- `scripts/build_vector_index.py`: 벡터 인덱스(FAISS/HNSW) 생성
- `server/app.py`: FastAPI 서버(`/api/ask`, `/api/qa/batch`, `/healthz`, `/metrics`)
- `docker-compose.yml`, `Dockerfile`: 컨테이너 실행 스캐폴딩(app+ollama)
- `data/`: `corpus.jsonl`, `tests/qa.json`, `pdfs/…`

## 설치(Windows/CPU 권장)
1) pip 업데이트
- `pip install --upgrade pip`
2) PyTorch CPU 핀 고정 설치
- `pip install --upgrade --force-reinstall "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cpu`
3) 의존성 설치
- `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`
- OCR 자동화를 쓰려면 Tesseract(엔진) 또는 ocrmypdf가 필요합니다. Tesseract 포터블 경로(TESSERACT_CMD/TESSDATA_PREFIX) 지정도 지원합니다.

## 가장 쉬운 실행(autorun)
- 완전 자동(설치 확인 → PDF 찾기 → OCR 폴백 → 벤치마크 실행):
  - `python scripts/autorun.py --backend auto`
  - 기본 PDF 탐색 경로: `data/pdfs` → `data/tests` → `./pdfs` → 없으면 `data/corpus_v1.jsonl` 사용
- 단일 질문만 실행:
  - `python scripts/autorun.py --backend auto --question "<질문>"`
- 경로를 고정하고 싶다면:
  - `python scripts/autorun.py --pdf data/pdfs --qa data/tests/qa.json --backend auto`
 - 자동 평가(기본 QA로 실행 후 리포트 저장):
   - `python scripts/autorun.py --auto`
   - 결과: `out/report_autorun_YYYYMMDD_HHMMSS.json,csv`

## 1분 원클릭 실행(quickstart)
- 단일 질문: `python scripts/quickstart.py --pdf data/pdfs --backend auto --question "<질문>"`
- 인터랙티브: `python scripts/quickstart.py --pdf data/pdfs --backend auto --interactive`
- 벤치마크: `python scripts/quickstart.py --pdf data/pdfs --qa data/tests/qa.json --backend auto`
- 서버 기동: `python scripts/quickstart.py --pdf data/pdfs --backend auto --server --host 0.0.0.0 --port 8000`
- 주요 옵션:
  - 백엔드: `--backend auto|faiss|hnsw`
  - PDF 추출기: `--pdf-extractor auto|plumber|fitz`
  - OCR: `--ocr auto|always|off`(기본 auto), `--ocr-lang kor+eng`, `--ocr-dpi 200`
  - 리랭커: `--use-cross-reranker --rerank-top-n 50`
  - 임계 튠: `--thr-base 0.25 --thr-numeric 0.16 --thr-long 0.17`

## 수동 절차
- PDF→코퍼스: `python scripts/build_corpus_from_pdfs.py --pdf_dir data/pdfs --out data/corpus_v1.jsonl --pdf-extractor auto --ocr auto --ocr-lang kor+eng --ocr-dpi 200`
- 벡터 인덱스: `python scripts/build_vector_index.py --corpus data/corpus_v1.jsonl --backend faiss --outdir vector_store`
- 수동 CLI: `python scripts/manual_cli.py --corpus data/corpus_v1.jsonl --mode accuracy --store-backend auto --question "<질문>"`
- 하네스: `python scripts/run_qa_benchmark.py --input data/tests/qa.json --corpus data/corpus_v1.jsonl --mode accuracy --store-backend auto --report out/report.json --csv out/report.csv`

 

## API 서버
- 로컬 실행: `uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload`
- 엔드포인트:
  - `POST /api/ask {question, mode, k}`
  - `POST /api/qa/batch {items:[{id,question}], mode}`
  - `GET /healthz`, `GET /metrics`(간단 프로메테우스 형식)

## Docker 실행(선택)
- `docker compose up --build`
- 서비스: app(FastAPI), ollama(LLM)
- 볼륨: `./data ./models ./vector_store ./logs`
- 헬스체크: app(`/healthz`), ollama(`/api/tags`)
- GPU 가능 시 자동 사용(`gpus: all`)

## 주요 구현 포인트
- 하이브리드 검색: BM25 + 벡터 → RRF(유형별 가중)
- 병합/디듀프: 안정 키 + 텍스트 해시 + 근사중복(Jaccard) + “최종 재디듀프”
- 필터 교정: 소스별 정규화 + z‑score 캡핑, 임계 0.25(+수치/장문 오버라이드), 다양성 제약
- 컨텍스트: 동적 k, NeighborSelector(문단/섹션/±1page), 최종 재디듀프
- 리랭커: 경량/크로스엔코더 + 타임아웃/예산 가드(On/Off & Top‑N 튠)
- 가드/폴백: 이중조건 + 저신뢰 재탐색/단일 스팬 → 표준 부정응답
- 수치/단위 정합: mg/L↔ppm, ±5% 허용오차 검증
- 관측/재현: 질문별 메트릭·`config_hash` 포함
- Windows 안전: 스레드 타임아웃, UTF‑8 경로/로그, 외부 의존 옵션화

## FAQ/문제 해결
- 콘솔 한글 깨짐: PowerShell `chcp 65001` 또는 `set PYTHONUTF8=1`
- 코퍼스가 0건으로 나옴: 스캔 PDF 가능성 → `--pdf-extractor fitz --ocr off` 재시도 또는 Tesseract/ocrmypdf 준비 후 `--ocr auto`
- Tesseract 설치 없이 가볍게: 포터블 바이너리만 두고 `TESSERACT_CMD`/`TESSDATA_PREFIX` 지정
- Windows에서 FAISS가 불필요: `--backend auto`로 TF‑IDF 폴백 사용

## 추가 문서
- `튜토리얼.md`: PDF 배치 위치, 명령 예시, OCR 옵션, 트러블슈팅을 단계별로 정리
