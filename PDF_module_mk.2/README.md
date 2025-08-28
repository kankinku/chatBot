# PDF QA 모듈 실행 가이드 (Python 스크립트 중심)

이 문서는 폴더 내부 구현 상세를 배제하고, 최상위 Python 스크립트들의 실행 방법만 빠르게 파악할 수 있도록 정리했습니다.

## 준비 사항

```powershell
# Windows PowerShell 기준
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
```

로컬 LLM은 기본적으로 Ollama 사용을 권장합니다. 설치 후 필요한 모델을 내려받으세요.

```powershell
ollama pull mistral:latest
# (옵션) 대체 모델
ollama pull llama2:7b
```

## 실행 스크립트 개요

- `main.py`: 전체 시스템 엔트리포인트 (대화형/서버/단일 처리)
- `korean_qa.py`: 간편 실행 스크립트 (설정/전처리/질문)
- `korean_qa_fast.py`: 경량 고속 질의응답
- `extract_pdf_content.py`: PDF 텍스트 추출 미리보기
- `simple_test.py`: 기본 동작 점검
- `test_qa_script.py`: 실제 질문 세트로 품질 측정
- `test_data.py`: 데이터베이스/벡터 저장소 점검
- `test_enhanced_keyword_recognition.py`: 키워드 인식률 향상 효과 측정(실전형)
- `test_keyword_enhancement.py`: 키워드 인식/검색/생성기 개선 포인트 점검
- `test_real_pdf_keywords.py`: 실제 PDF 기반 키워드 인식률/품질 테스트

아래 모든 커맨드는 프로젝트 루트에서 실행하며, PDF는 보통 `data/pdfs/` 하위에 둡니다. 경로 인자는 공백이 있을 수 있으니 필요 시 따옴표로 감싸세요.

## main.py (권장 진입점)

- 대화형 모드: PDF 추가/처리 + 질의응답

```powershell
python main.py --mode interactive --model-name mistral:latest

# 시작 시 PDF를 지정해 관리 폴더로 복사 및 처리
python main.py --mode interactive --pdf "C:\path\to\document.pdf" --model-name mistral:latest
```

- 서버 모드: FastAPI 구동 (문서: http://localhost:8000/docs)

```powershell
python main.py --mode server --model-name mistral:latest --host 0.0.0.0 --port 8000
```

- 단일 처리 모드: 1회성 PDF 처리 + 질문 1건

```powershell
python main.py --mode process --pdf "C:\path\to\document.pdf" --question "이 문서 핵심은?" --model-name mistral:latest
```

자주 쓰는 대화형 명령어(입력창에서 사용):

- `/clear` 대화 기록 초기화
- `/status` 시스템 상태 확인
- `/pdfs` 저장된 PDF 목록
- `/add <경로>` PDF 추가 및 카테고리 선택
- `/categories` 카테고리/저장소 요약
- `/exit` 종료

## korean_qa.py (간편 실행)

```powershell
# 초기 설정(전처리 포함)
python korean_qa.py --setup

# PDF 전처리만 수행 (data/pdfs 하위 전체)
python korean_qa.py --process

# 통계 출력
python korean_qa.py --stats

# 질문 1건 직접 실행
python korean_qa.py "여과공정이 뭐야?"

# 옵션 없이 실행 → 대화형 모드 진입
python korean_qa.py
```

## korean_qa_fast.py (고속 질의응답)

```powershell
# 대화형 고속 모드
python korean_qa_fast.py --interactive

# 단일 질문 1건
python korean_qa_fast.py "세종시 교통 문제 핵심은?"
```

## extract_pdf_content.py (PDF 내용 미리보기)

`./data` 폴더에서 PDF를 찾아 텍스트 일부와 메타데이터를 출력합니다.

```powershell
python extract_pdf_content.py
```

## 기본 점검 및 테스트 스크립트

- 단일 실행 진입점(권장):

```powershell
# simple/data/qa/keyword/keyword-adv/keyword-real 중 선택
python run_checks.py simple
python run_checks.py data
python run_checks.py qa --iterations 1 --save
python run_checks.py keyword
python run_checks.py keyword-adv
python run_checks.py keyword-real
```

- 개별 실행도 가능:

```powershell
python checks/test_qa_script.py --iterations 1
```

- 데이터 상태 점검(DB/벡터 인덱스/검색 샘플):

```powershell
python checks/test_data.py
```

## 키워드 인식/검색 개선 관련 테스트

- 실전형 개선 효과 측정(도메인별 키워드, 처리시간, 품질 비율 등):

```powershell
python test_enhanced_keyword_recognition.py
```

- 개선 포인트 빠른 점검(키워드 향상/질문분석/검색/생성기 개요):

```powershell
python test_keyword_enhancement.py
```

- 실제 PDF 기반 인식률/품질/성능 비교(샘플 키워드 추천 포함):

```powershell
python test_real_pdf_keywords.py
```

## 자주 묻는 질문

- 모델 로드가 느리거나 실패해요
  - Ollama 설치/백그라운드 실행 상태 확인
  - `ollama pull mistral:latest` 재시도

- PDF가 검색되지 않아요
  - `korean_qa.py --process` 또는 `main.py` 대화형에서 `/add <경로>`로 전처리 수행

- 벡터 저장소 초기화/재생성이 필요해요
  - `data/vector_store/`를 백업 후 삭제하고 재전처리

---

가장 빠른 시작: 대화형 실행 후 바로 질문하기

```powershell
python main.py --mode interactive --model-name mistral:latest
```

