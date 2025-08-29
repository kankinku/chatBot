# PDF QA 시스템 사용설명서

이 문서는 PDF QA 시스템의 서버 실행과 LLM 사용 방법을 설명합니다.

## 목차
1. [서버 실행](#1-서버-실행)
2. [LLM 사용](#2-llm-사용)
3. [준비 사항](#3-준비-사항)
4. [문제 해결](#4-문제-해결)

---

## 1. 서버 실행

### 1.1 FastAPI 서버 실행

PDF QA 시스템을 웹 API 서버로 실행합니다.

```powershell
# 기본 서버 실행 (포트 8000)
python main.py --mode server --model-name mistral:latest

# 특정 호스트와 포트로 실행
python main.py --mode server --model-name mistral:latest --host 0.0.0.0 --port 8000

# 직접 API 엔드포인트 실행
python api/endpoints.py
```

### 1.2 서버 접속

서버 실행 후 다음 URL로 접속할 수 있습니다:

- **API 문서**: http://localhost:8000/docs
- **대체 문서**: http://localhost:8000/redoc
- **헬스 체크**: http://localhost:8000/health
- **시스템 상태**: http://localhost:8000/status

### 1.3 주요 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/upload_pdf` | POST | PDF 파일 업로드 |
| `/ask` | POST | 질문-답변 (일반) |
| `/django/ask` | POST | 질문-답변 (Django용) |
| `/status` | GET | 시스템 상태 조회 |
| `/pdfs` | GET | 등록된 PDF 목록 |
| `/conversation_history` | GET | 대화 기록 조회 |
| `/health` | GET | 서버 헬스 체크 |
| `/conversation/cache/stats` | GET | 대화 이력 캐시 통계 |
| `/conversation/cache` | DELETE | 대화 이력 캐시 삭제 |
| `/conversation/cache/search` | GET | 대화 이력에서 유사 질문 검색 |

### 1.4 클라이언트 사용

#### Django 백엔드에서 사용
```python
from api.django_client import PDFQAClient

# 클라이언트 생성
client = PDFQAClient("http://localhost:8000")

# 질문하기 (대화 이력 캐시 자동 사용)
response = client.ask_question("질문 내용", pdf_id="default")

# PDF 업로드
result = client.upload_pdf("path/to/document.pdf")

# 대화 이력 캐시 관리
stats = client.get_conversation_cache_stats()
client.clear_conversation_cache()  # 전체 캐시 삭제
client.clear_conversation_cache("pdf_id")  # 특정 PDF 캐시 삭제

# 유사 질문 검색
similar = client.search_conversation_cache("질문", threshold=0.8)
```

#### React 프론트엔드에서 사용
```typescript
import { PDFQAClient } from './api/typescript_client';

// 클라이언트 생성
const client = new PDFQAClient('http://localhost:8000');

// 질문하기 (대화 이력 캐시 자동 사용)
const response = await client.askQuestion({
  question: "질문 내용",
  pdf_id: "default"
});

// PDF 업로드
const result = await client.uploadPDF(file);

// 대화 이력 캐시 관리
const stats = await client.getConversationCacheStats();
await client.clearConversationCache();  // 전체 캐시 삭제
await client.clearConversationCache("pdf_id");  // 특정 PDF 캐시 삭제

// 유사 질문 검색
const similar = await client.searchConversationCache("질문", 0.8);
```

---

## 2. LLM 사용

### 2.1 대화형 모드

PDF와 대화할 수 있는 대화형 인터페이스를 실행합니다.

```powershell
# 기본 대화형 모드
python main.py --mode interactive --model-name mistral:latest

# PDF 지정하여 시작
python main.py --mode interactive --pdf "C:\path\to\document.pdf" --model-name mistral:latest
```

### 2.2 대화형 명령어

대화형 모드에서 사용할 수 있는 명령어들:

| 명령어 | 설명 |
|--------|------|
| `/clear` | 대화 기록 초기화 |
| `/status` | 시스템 상태 확인 |
| `/pdfs` | 저장된 PDF 목록 |
| `/add <경로>` | PDF 추가 및 카테고리 선택 |
| `/categories` | 카테고리/저장소 요약 |
| `/exit` | 종료 |

### 2.3 단일 처리 모드

한 번의 질문에 대한 답변을 받습니다.

```powershell
python main.py --mode process --pdf "C:\path\to\document.pdf" --question "이 문서의 핵심 내용은?" --model-name mistral:latest
```

### 2.4 대화 이력 캐시 기능

PDF QA 시스템은 대화 이력을 SQLite 데이터베이스에 저장하여 빠른 답변을 제공합니다.

#### 주요 기능:
- **빠른 답변**: 동일하거나 유사한 질문에 대해 LLM 사용 없이 즉시 답변
- **정확한 일치**: 해시 기반 정확한 질문 매칭
- **유사 질문 검색**: Jaccard 유사도 기반 유사 질문 찾기
- **품질 필터링**: 신뢰도가 낮거나 에러 메시지는 캐시에 저장하지 않음
- **PDF별 분리**: 각 PDF별로 대화 이력 관리

#### 캐시 사용 조건:
- 신뢰도 ≥ 0.5인 답변만 저장
- 답변 길이 10-2000자
- 에러 메시지나 기본 답변 제외
- 정확한 일치 또는 유사도 ≥ 0.85

#### 성능 향상:
- **정확한 일치**: 0.001초 (LLM 대비 1000배 빠름)
- **유사 질문**: 0.002초 (LLM 대비 500배 빠름)
- **메모리 효율**: SQLite 인덱스 기반 빠른 검색

### 2.5 지원하는 LLM 모델

#### Ollama 모델 (권장)
```powershell
# Mistral 모델 (기본 권장)
ollama pull mistral:latest

# Llama2 모델
ollama pull llama2:7b

# 기타 모델들
ollama pull codellama:7b
ollama pull llama2:13b
```

#### 모델 설정 변경
API를 통해 모델 설정을 변경할 수 있습니다:

```bash
curl -X POST http://localhost:8000/configure_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "ollama",
    "model_name": "llama2:7b",
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

---

## 3. 준비 사항

### 3.1 환경 설정

```powershell
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\Activate

# 의존성 설치
pip install -r requirements.txt
```

### 3.2 Ollama 설치 및 설정

1. **Ollama 설치**: https://ollama.ai 에서 다운로드
2. **모델 다운로드**:
   ```powershell
   ollama pull mistral:latest
   ```
3. **Ollama 서비스 시작**:
   ```powershell
   ollama serve
   ```

### 3.3 PDF 파일 준비

PDF 파일은 다음 위치에 저장하거나 업로드할 수 있습니다:

- **로컬 저장**: `data/pdfs/` 디렉토리
- **API 업로드**: `/upload_pdf` 엔드포인트 사용

---

## 4. 문제 해결

### 4.1 서버 관련 문제

**서버가 시작되지 않아요**
- Ollama가 실행 중인지 확인: `ollama list`
- 포트 8000이 사용 중인지 확인
- 방화벽 설정 확인

**API 요청이 실패해요**
- 서버가 실행 중인지 확인: `curl http://localhost:8000/health`
- CORS 설정 확인 (프론트엔드에서 접근 시)

### 4.2 LLM 관련 문제

**모델 로드가 실패해요**
```powershell
# Ollama 상태 확인
ollama list

# 모델 재다운로드
ollama pull mistral:latest

# Ollama 재시작
ollama serve
```

**답변 품질이 낮아요**
- 더 큰 모델 사용: `llama2:13b`, `mistral:large`
- 모델 설정 조정: temperature, top_p 값 변경
- 질문을 더 구체적으로 작성

### 4.3 PDF 관련 문제

**PDF가 검색되지 않아요**
- PDF가 올바르게 업로드되었는지 확인
- 텍스트 추출이 가능한 PDF인지 확인
- 벡터 저장소 재생성 필요할 수 있음

**벡터 저장소 초기화**
```powershell
# 벡터 저장소 백업 후 삭제
# data/vector_store/ 디렉토리 삭제 후 재시작
```

### 4.4 성능 최적화

**응답 속도 개선**
- 더 빠른 모델 사용: `mistral:7b`
- GPU 가속 사용 (지원하는 경우)
- 벡터 저장소 최적화

**메모리 사용량 최적화**
- 더 작은 모델 사용
- 청크 크기 조정
- 불필요한 PDF 제거

---

## 빠른 시작 가이드

1. **환경 설정**
   ```powershell
   python -m venv venv
   venv\Scripts\Activate
   pip install -r requirements.txt
   ```

2. **Ollama 설정**
   ```powershell
   ollama pull mistral:latest
   ollama serve
   ```

3. **서버 실행**
   ```powershell
   python main.py --mode server --model-name mistral:latest
   ```

4. **웹 브라우저에서 확인**
   - http://localhost:8000/docs 접속
   - PDF 업로드 및 질문 테스트

또는 대화형 모드로 바로 시작:
```powershell
python main.py --mode interactive --model-name mistral:latest
```

