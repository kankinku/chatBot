# Chatbot Server 통합 가이드

## 📋 완료된 작업

### 1. ✅ FastAPI에 `/api` 프리픽스 추가

**변경 파일**: `api/app.py`

**변경 내용**:
- `/api/ask` 엔드포인트 추가 (프록시 서버 호환)
- `/api/healthz` 엔드포인트 추가
- `/api/status` 엔드포인트 추가
- 기존 엔드포인트는 하위 호환성을 위해 유지

**사용 가능한 엔드포인트**:
- `POST /ask` 또는 `POST /api/ask` - 질문 답변
- `GET /healthz` 또는 `GET /api/healthz` - 헬스 체크
- `GET /status` 또는 `GET /api/status` - 서비스 상태

### 2. ✅ Docker Compose 통합 설정

**변경 파일**: `docker-compose.yml`

**추가된 서비스**:
1. **chatbot-backend** (포트 8000)
   - FastAPI 챗봇 서버
   - RAG 파이프라인 실행

2. **mysql** (포트 3306)
   - 프록시 서버용 데이터베이스
   - 대화 기록, 메트릭 저장

3. **backend-proxy** (포트 8001)
   - Django 프록시 서버
   - 챗봇 서버로 요청 전달
   - 대화 기록 관리

4. **frontend** (포트 3000)
   - React 프론트엔드
   - Nginx로 서빙

5. **ollama** (포트 11434)
   - LLM 서버

### 3. ✅ Django 프록시 서버 설정 조정

**변경 파일**: `server/backend/chatbot_proxy/views.py`

**변경 내용**:
- `CHATBOT_URL` 환경 변수 지원
- 로깅 추가 (설정된 URL 확인 가능)

**환경 변수**:
- `CHATBOT_URL`: 챗봇 서버 URL (기본값: `http://localhost:8000`)
- Docker 환경에서는 `http://chatbot-backend:8000`로 자동 설정

### 4. ✅ 프론트엔드 API URL 수정

**변경 파일**:
- `server/frontend/Dockerfile` - 프로덕션 빌드로 변경
- `server/frontend/nginx.conf` - Nginx 설정 추가

**변경 내용**:
- 빌드 시 `REACT_APP_API_URL` 환경 변수 주입
- Nginx로 정적 파일 서빙
- React Router 지원

## 🚀 실행 방법

### 전체 시스템 실행

```bash
# 모든 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 특정 서비스 로그만 확인
docker-compose logs -f chatbot-backend
docker-compose logs -f backend-proxy
docker-compose logs -f frontend
```

### 개별 서비스 실행

```bash
# 챗봇 서버만 실행 (포트 8000)
docker-compose up chatbot-backend

# 프록시 서버만 실행 (포트 8001)
docker-compose up backend-proxy

# 프론트엔드만 실행 (포트 3000)
docker-compose up frontend
```

## 📡 API 엔드포인트

### 직접 챗봇 서버 호출 (포트 8000)

```bash
# 질문 답변
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "고산 정수장 URL은?", "top_k": 50}'

# 헬스 체크
curl http://localhost:8000/api/healthz

# 상태 확인
curl http://localhost:8000/api/status
```

### 프록시 서버를 통한 호출 (포트 8001)

```bash
# 질문 답변 (대화 기록 저장됨)
curl -X POST http://localhost:8001/api/chatbot/ask \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: test-session-123" \
  -d '{"question": "고산 정수장 URL은?", "mode": "accuracy", "k": "auto"}'

# 대화 기록 조회
curl http://localhost:8001/api/chatbot/conversations/test-session-123

# 메트릭 조회
curl http://localhost:8001/api/chatbot/metrics
```

### 프론트엔드 접속 (포트 3000)

브라우저에서 `http://localhost:3000` 접속

## 🔧 환경 변수 설정

### 챗봇 서버 (chatbot-backend)

```yaml
environment:
  - OLLAMA_HOST=ollama
  - OLLAMA_PORT=11434
```

### 프록시 서버 (backend-proxy)

```yaml
environment:
  - CHATBOT_URL=http://chatbot-backend:8000
  - MYSQL_HOST=mysql
  - MYSQL_DATABASE=chatbot_db
  - MYSQL_USER=chatbot_user
  - MYSQL_PASSWORD=1234
```

### 프론트엔드 (frontend)

```yaml
build:
  args:
    REACT_APP_API_URL: http://localhost:8001
```

## 📊 시스템 구조

```
사용자 (브라우저)
    ↓
Frontend (React) - 포트 3000
    ↓ HTTP 요청
Backend Proxy (Django) - 포트 8001
    ↓ 프록시 요청
Chatbot Backend (FastAPI) - 포트 8000
    ↓ RAG Pipeline
Modules (검색, 임베딩, 생성)
    ↓
Ollama (LLM) - 포트 11434
```

## 🗄️ 데이터베이스 초기화

```bash
# MySQL 데이터베이스 마이그레이션
docker-compose exec backend-proxy python manage.py migrate

# 관리자 계정 생성 (선택사항)
docker-compose exec backend-proxy python manage.py createsuperuser
```

## 🔍 문제 해결

### 챗봇 서버 연결 실패

```bash
# 챗봇 서버 로그 확인
docker-compose logs chatbot-backend

# 프록시 서버 로그 확인
docker-compose logs backend-proxy

# 환경 변수 확인
docker-compose exec backend-proxy env | grep CHATBOT_URL
```

### 프론트엔드 빌드 실패

```bash
# 프론트엔드 컨테이너 내부 접속
docker-compose exec frontend sh

# 수동 빌드 테스트
cd /app
npm run build
```

### MySQL 연결 실패

```bash
# MySQL 로그 확인
docker-compose logs mysql

# MySQL 컨테이너 접속
docker-compose exec mysql mysql -u chatbot_user -p1234 chatbot_db
```

## 📝 다음 단계

1. **실제 문서 로드**: 현재 더미 데이터 사용 중
2. **인증 추가**: JWT 토큰 기반 인증 구현
3. **모니터링**: Prometheus + Grafana 설정
4. **로깅**: ELK Stack 통합

## 🎯 주요 기능

- ✅ RAG 기반 질문 답변
- ✅ 대화 기록 저장 및 관리
- ✅ 성능 메트릭 수집
- ✅ 시스템 로그 관리
- ✅ React 기반 사용자 인터페이스
- ✅ Docker 기반 배포


