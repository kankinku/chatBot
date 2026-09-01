# R2-B2 인증·소유권·운영 route 단계 보고서

- 단계: R2-B2 실제 proxy route 인증과 owner 격리
- 실행일: 2026-09-01
- 브랜치: `codex/r1-reproducibility`
- 기준 커밋: `27a51a4 feat: add secure proxy policy primitives`
- 범위: Django/Ninja proxy route의 actor 인증, operator route, 대화 소유권 필터, owner key additive migration, telemetry 실패 격리

## 변경

- `Server/backend/chatbot_proxy/views.py`
  - `_require_actor()`를 모든 민감 chat/ask/batch/conversation route에 연결
  - PDF 처리, upstream metrics, logs, DB metrics는 `operator=True`로 제한
  - 인증 실패는 HTTP 401, 권한/소유권 실패는 HTTP 403으로 변환
  - 비운영자 대화 목록/상세/삭제 조회에 `owner_key` 필터 적용
  - ownerless legacy 대화는 자동 귀속하지 않고 접근 거부
  - conversation/log/metrics 저장 helper가 owner key를 요구
  - 로그/metrics telemetry 저장 실패가 upstream 성공/실패 결과를 덮지 않도록 best-effort wrapper 적용
- `Server/backend/chatbot_proxy/models.py`
  - Conversation, ChatLog, ChatMetrics에 nullable/indexed `owner_key` 추가
- `Server/backend/chatbot_proxy/migrations/0002_add_owner_key.py`
  - 기존 행을 자동 claim하지 않는 additive migration 추가
- `Server/backend/chatbot_backend/urls.py`
  - root PDF 시작/상태 route에도 operator 인증 연결
- `tests/unit/test_proxy_auth_contracts.py`
  - route 인증, 401/403 경계, owner 필터, telemetry wrapper, migration 계약 검증

## TDD/검증

| 검사 | 결과 |
|---|---|
| R2-B2 인증/소유권 RED | 기존 route·model·migration 계약 실패 확인 |
| R2-B2 GREEN focused | `26 passed` (평가 기준) |
| 전체 unit | `68 passed` |
| 전체 pytest | `78 passed in 155.01s` |
| Python compileall | PASS |
| `git diff --check` | PASS |
| Django/MySQL runtime | BLOCKED: 현재 환경에는 Django/DB runtime 연결이 없음 |
| coverage | BLOCKED: coverage/pytest-cov 미설치 |

## 평가 결과

- 보안 평가 서브 에이전트: R2-B2 blocker 없음, LOW risk
- 품질 평가 서브 에이전트: telemetry blocker 수정 후 GOOD, blocker 없음
- 평가에서 확인된 deferred 항목
  - 상세 `/status`가 운영 정보를 공개하므로 R2-C에서 public health와 operator status를 분리
  - nullable legacy ownerless 대화/metrics의 backfill·재연결 정책 결정
  - Django request/migration runtime 테스트는 의존성·DB가 준비된 환경에서 추가

## 범위 경계

이 단계는 proxy route 인증과 owner 경계를 연결했다. FastAPI direct service endpoint의 별도 노출 정책과 health/status/error 공개 계약은 R2-C에서 처리한다. legacy 데이터는 안전을 위해 자동 claim하지 않았으므로, 기존 session ID의 사용자 재연결은 아직 제공하지 않는다.

## 롤백

`views.py`, `models.py`, `urls.py`, `0002_add_owner_key.py`, R2-B2 계약 테스트와 이 증거 리포트를 같은 단계 단위로 되돌린다. R1, R2-A, R2-B1 변경 및 기존 데이터 자체는 건드리지 않는다.
