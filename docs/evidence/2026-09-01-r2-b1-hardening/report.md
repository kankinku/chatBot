# R2-B1 보안 설정 fail-closed 보완 보고서

- 구분: 독립 평가 FAIL 후 보완 추가 커밋
- 실행일: 2026-09-01
- 브랜치: `codex/r1-reproducibility`
- 보완 대상: `27a51a4 feat: add secure proxy policy primitives`
- 범위: 명시적 deployment environment, production secret/DB credential 검증, 실행 가능한 설정 검증 테스트

## 독립 평가 FAIL 원인

- `ENVIRONMENT` 미설정 시 development로 기본 처리되어 fail-open 가능
- production에서 compose/documentation placeholder secret과 짧은 secret을 허용할 수 있음
- production에서 `MYSQL_PASSWORD=1234`를 허용할 수 있음
- 설정 테스트가 source/AST 구조에 치우쳐 runtime 검증이 부족함

## 보완 내용

- `Server/backend/chatbot_backend/configuration_security.py` 추가
  - 허용 environment를 `development`/`production`으로 제한
  - production secret은 known placeholder를 거부하고 최소 50자 요구
  - production의 DEBUG, wildcard host, 전체 CORS, local anonymous, 빈 값/약한 DB password 거부
- `Server/backend/chatbot_backend/settings.py`
  - ENVIRONMENT 기본값을 빈 값으로 바꾸고 중앙 검증 함수에 위임
  - 설정 검증 실패를 Django `ImproperlyConfigured`로 변환
- `docker-compose.yml`
  - 현재 compose 실행 모드를 `ENVIRONMENT=development`로 명시
- 순수 설정 검증 테스트 추가 및 기존 source 계약 테스트를 중앙 검증 구조에 맞게 보강

## 검증

| 검사 | 결과 |
|---|---|
| 보완 설정 focused tests | `18 passed` |
| 전체 unit | `83 passed` |
| 전체 pytest | `93 passed in 138.92s` |
| Python compileall | PASS |
| `git diff --check` | PASS |
| Django runtime settings import | BLOCKED: 현재 환경에 Django/decouple이 없음 |
| coverage | BLOCKED: coverage/pytest-cov 미설치 |

## 재평가 상태

이 보고서는 보완 구현 전 증거다. 추가 커밋 생성 후, 원래 FAIL을 낸 평가자와 다른 새 컨텍스트의 독립 code-review 서브 에이전트가 추가 커밋을 재평가해야 한다. 재평가 PASS 전에는 R2-B2 및 merge를 진행하지 않는다.

## 범위 경계

이 보완은 R2-B1 설정 fail-closed 문제만 다룬다. proxy route 인증은 R2-B2 커밋에 있으며, health/status 공개 계약과 direct FastAPI service 노출은 R2-C 범위다.
