# Canonical Repository Architecture

현재 제품 런타임은 하나의 canonical tree를 사용한다.

- `apps/web`: 사용자 UI
- `services/gateway`: 공개 Django edge, 인증/ownership/operator/persistence
- `services/inference`: 내부 FastAPI adapter
- `src/chatbot`: framework-independent RAG domain logic
- `config`: 런타임 구성
- `deploy`: container/compose wiring

## Dependency direction

```text
apps/web
   |
   v
services/gateway
   |
   v
services/inference
   |
   v
src/chatbot
```

`src/chatbot`은 Django/React에 의존하지 않는다. 서비스 계층은 core를 호출하는 adapter로 유지한다.

## Versioning

새 기능을 `Chatbot_v7` 같은 새 디렉터리로 복제하지 않는다. branch/PR에서 변경하고 tag/release로 배포 버전을 표현한다.
