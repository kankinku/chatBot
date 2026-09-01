# 챗봇 백엔드 API

이 프로젝트는 챗봇 프론트엔드와 챗봇 서버 간의 프록시 역할을 하는 Django 백엔드 API입니다.

## 🚀 시작하기

### 1. 가상환경 활성화
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정
```bash
# env.example을 .env로 복사
copy env.example .env
```

### 4. 데이터베이스 마이그레이션
```bash
python manage.py migrate
```

### 5. 서버 실행
```bash
python manage.py runserver
```

## 📁 프로젝트 구조

```
chatbot_backend/
├── chatbot_backend/          # Django 프로젝트 설정
│   ├── __init__.py
│   ├── settings.py          # Django 설정
│   ├── urls.py             # URL 라우팅
│   ├── wsgi.py             # WSGI 설정
│   └── asgi.py             # ASGI 설정
├── chatbot_proxy/           # 챗봇 프록시 앱
│   ├── __init__.py
│   ├── apps.py
│   ├── views.py            # API 뷰
│   └── models.py           # 데이터 모델
├── manage.py               # Django 관리 스크립트
├── requirements.txt        # Python 의존성
├── env.example            # 환경변수 예시
└── README.md              # 프로젝트 문서
```

## 🔧 API 엔드포인트

### 챗봇 관련 API

- `POST /api/chatbot/chat` - 간단한 챗봇 메시지
- `POST /api/chatbot/ask` - AI 질문 답변
- `POST /api/chatbot/batch` - 배치 질문 답변
- `GET /api/chatbot/status` - 챗봇 서버 상태
- `GET /api/chatbot/health` - 헬스 체크
- `GET /api/chatbot/metrics` - 메트릭 조회
- `GET /api/chatbot/pdfs` - PDF 목록 조회
- `GET /api/chatbot/conversation_history` - 대화 기록 조회
- `DELETE /api/chatbot/conversation_history` - 대화 기록 초기화

## 🌐 환경변수 설정

### 필수 환경변수

```env
# Django 설정: ENVIRONMENT는 반드시 명시하고 실제 값은 secret manager에서 주입
ENVIRONMENT=development
# Inject a unique value from a secret manager; do not commit the real value.
SECRET_KEY=<SET_IN_SECRET_MANAGER>
DEBUG=True
CHATBOT_ALLOW_ANONYMOUS_LOCAL=False

# MySQL 설정
MYSQL_DATABASE=chatbot_db
MYSQL_USER=chatbot_user
MYSQL_PASSWORD=<SET_IN_SECRET_MANAGER>
MYSQL_ROOT_PASSWORD=<SET_IN_SECRET_MANAGER>
MYSQL_HOST=localhost
MYSQL_PORT=3306

# 챗봇 서버 설정
CHATBOT_URL=http://localhost:8000
```

### 선택적 환경변수

```env
# CORS 설정: wildcard origin은 사용하지 않음
CORS_ALLOW_ALL_ORIGINS=False
```

## 🔧 기술 스택

- **Django 4.2.7**: 웹 프레임워크
- **Django Ninja**: API 프레임워크
- **Django Ninja Extra**: 추가 기능
- **Django Ninja JWT**: JWT 인증
- **Django CORS Headers**: CORS 처리
- **httpx**: 비동기 HTTP 클라이언트
- **python-decouple**: 환경변수 관리

## 📱 사용법

1. 챗봇 서버가 실행 중인지 확인
2. 백엔드 API 서버 실행
3. 인증된 세션으로 프록시 API 호출

### 예시 요청

```bash
# 공개 liveness 확인
curl http://localhost:8001/api/chatbot/health

# 간단한 챗봇 메시지 (Django 인증 세션 필요)
curl -X POST http://localhost:8001/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -b "sessionid=<DJANGO_SESSION_COOKIE>" \
  -d '{"message": "안녕하세요"}'

# AI 질문 답변 (Django 인증 세션 필요)
curl -X POST http://localhost:8001/api/chatbot/ask \
  -H "Content-Type: application/json" \
  -b "sessionid=<DJANGO_SESSION_COOKIE>" \
  -d '{"question": "교통 상황은 어떤가요?", "mode": "accuracy", "k": "auto"}'

# 상세 상태 확인 (운영자 세션 필요)
curl http://localhost:8001/api/chatbot/status \
  -b "sessionid=<DJANGO_OPERATOR_SESSION_COOKIE>"
```

FastAPI의 8000 포트와 MySQL의 3306 포트는 Compose에서 호스트에 공개하지
않습니다. Ollama의 11434 포트도 `chatbot-backend` 내부 전용입니다.

## 🐛 문제 해결

### 챗봇 서버 연결 오류
1. 챗봇 서버가 실행 중인지 확인
2. `CHATBOT_URL` 환경변수가 올바른지 확인
3. 네트워크 연결 상태 확인

### CORS 오류
1. `CORS_ALLOW_ALL_ORIGINS=False`를 유지하는지 확인
2. 프론트엔드 URL이 `CORS_ALLOWED_ORIGINS`에 포함되어 있는지 확인

### 의존성 오류
1. 가상환경이 활성화되어 있는지 확인
2. `pip install -r requirements.txt` 재실행

## 📝 로그

로그는 `chatbot.log` 파일과 콘솔에 출력됩니다.

- **INFO**: 일반적인 정보
- **DEBUG**: 디버그 정보
- **ERROR**: 오류 정보
