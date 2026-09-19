# Server Integration Guide

canonical runtime은 React → Django gateway → FastAPI inference → RAG core 순서로 구성된다.

## Components

- `apps/web`: React UI, host port 3000
- `services/gateway`: Django public edge, loopback port 8001
- `services/inference`: FastAPI internal service, container port 8000
- `src/chatbot`: RAG core
- Ollama: internal model runtime
- MySQL: internal gateway persistence

FastAPI, MySQL, Ollama는 host에 직접 publish하지 않는다. 운영 API와 질문 API는 Django gateway의 인증/권한 경계를 통해 접근한다.

## Compose

구성 파일:

```text
deploy/compose/docker-compose.yml
```

전체 실행:

```bash
docker compose -f deploy/compose/docker-compose.yml up -d --build
```

로그:

```bash
docker compose -f deploy/compose/docker-compose.yml logs -f
docker compose -f deploy/compose/docker-compose.yml logs -f chatbot-backend
docker compose -f deploy/compose/docker-compose.yml logs -f backend-proxy
```

## Internal inference endpoints

FastAPI는 내부 네트워크에서 다음 endpoint를 제공한다.

- `POST /ask`, `POST /api/ask`
- `GET /healthz`, `GET /api/healthz`
- `GET /status`, `GET /api/status`

컨테이너 내부 liveness 확인:

```bash
docker compose -f deploy/compose/docker-compose.yml exec chatbot-backend \
  curl http://localhost:8000/healthz
```

## Public gateway

Django gateway 기본 host 경계는 `127.0.0.1:8001`이다.

인증된 질문:

```bash
curl -X POST http://localhost:8001/api/chatbot/ask \
  -H "Content-Type: application/json" \
  -b "sessionid=<DJANGO_SESSION_COOKIE>" \
  -H "X-Session-ID: test-session-123" \
  -d '{"question":"고산 정수장 URL은?","mode":"accuracy","k":"auto"}'
```

운영자 전용 endpoint:

- `/api/chatbot/status`
- `/api/chatbot/metrics`
- `/api/chatbot/upstream-metrics`
- PDF processing/log operational routes

## Required environment

Gateway/MySQL 실행에는 최소 다음 secret이 필요하다.

```bash
export MYSQL_ROOT_PASSWORD='<strong-root-password>'
export MYSQL_PASSWORD='<strong-app-password>'
export SECRET_KEY='<strong-django-secret>'
```

추가 설정은 `services/gateway/env.example`을 참고한다.

## Database administration

```bash
docker compose -f deploy/compose/docker-compose.yml exec backend-proxy python manage.py migrate
docker compose -f deploy/compose/docker-compose.yml exec backend-proxy python manage.py createsuperuser
```

## Source locations

- FastAPI app: `services/inference/api/app.py`
- Django settings: `services/gateway/chatbot_backend/settings.py`
- Django proxy: `services/gateway/chatbot_proxy/views.py`
- React source: `apps/web/src`
- RAG core: `src/chatbot`
- Compose: `deploy/compose/docker-compose.yml`
