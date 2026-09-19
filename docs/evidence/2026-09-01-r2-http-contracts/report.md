# R2-A HTTP 응답 계약 단계 보고서

- 단계: R2-A 응답 모델 충돌 제거
- 실행일: 2026-09-01
- 브랜치: `codex/r1-reproducibility`
- 기준 커밋: `08962bf test: establish R1 offline validation baseline`
- 범위: Django proxy 응답 schema 이름 충돌과 그 회귀 계약

## 문제

`Server/backend/chatbot_proxy/views.py`에서 단순 채팅 응답과 저장 메시지 응답이 모두 `ChatMessageResponse`라는 이름을 사용했다. 후반 클래스 정의가 전역 이름을 덮어써 `/chat` 함수의 실제 응답 생성자가 저장 메시지 필드를 요구하는 모델을 참조할 수 있었다.

## 변경

- 단순 채팅 응답을 `SimpleChatResponse`로 분리
- 저장 메시지 응답을 `StoredChatMessageResponse`로 분리
- `/chat` route decorator와 반환 생성자가 `SimpleChatResponse`를 사용하도록 연결
- 대화 상세 응답의 메시지 목록이 `StoredChatMessageResponse`를 사용하도록 연결
- Django/Ninja 런타임을 import하지 않고 AST로 schema 정의·route·생성자 연결을 검증하는 계약 테스트 추가

## 검증

| 검사 | 결과 |
|---|---|
| 응답 계약 RED 테스트 | 기존 코드에서 2건 실패 확인 |
| 응답 계약 GREEN 테스트 | `2 passed` |
| 전체 unit | `39 passed` |
| 전체 테스트 수집 | `49 collected` |
| Python AST | `88 files parsed` |
| `git diff --check` 및 staging 후 `git diff --cached --check` | PASS |

실제 Django/MySQL/FastAPI end-to-end는 현재 환경에서 Django·DB 의존성 및 운영 연결을 준비하지 않아 실행하지 않았다. 이 보고서는 응답 schema wiring이 수정됐다는 의미이며, 인증·소유권·upstream 실제 상태까지 해결했다는 의미가 아니다.

## 롤백

`views.py`의 두 응답 모델 이름 변경과 `tests/unit/test_proxy_contracts.py`를 같은 커밋 단위로 되돌린다. R1 변경과 기존 대화 기록 데이터는 건드리지 않는다.
