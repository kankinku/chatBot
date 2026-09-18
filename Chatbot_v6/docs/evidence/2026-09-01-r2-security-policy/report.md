# R2-B1 보안 정책·설정 기본값 단계 보고서

- 단계: R2-B1 인증 정책 primitive 및 안전한 설정 기본값
- 실행일: 2026-09-01
- 브랜치: `codex/r1-reproducibility`
- 기준 커밋: `ef1f15e fix: separate proxy response schemas`
- 범위: 인증 actor 판별, loopback 익명 접근 조건, 소유권 비교 primitive, production 설정 fail-closed 기본값

## 변경

- `Server/backend/chatbot_proxy/security.py` 추가
  - 인증 사용자에 대해 `user:<pk>` 기반의 안정적인 owner key 생성
  - 익명 접근은 `DEBUG=True` + 명시적 opt-in + IPv4/IPv6 loopback일 때만 허용
  - 익명 actor에는 운영자 권한을 부여하지 않음
  - 누락/불일치 owner를 거부하는 `require_owner()` 추가
  - 다음 route 연결 단계에서 HTTP 401/403으로 변환할 예외 타입 정의
- `Server/backend/chatbot_backend/settings.py` 보강
  - 허용 environment를 `development`/`production`으로 제한
  - production에서 기본/문서 placeholder `SECRET_KEY` 거부
  - production에서 `DEBUG=True`, wildcard `ALLOWED_HOSTS`, 전체 CORS, local anonymous access 거부
  - DB 비밀번호 fallback을 빈 값으로 바꾸고 production의 빈 값/`change-me` 거부
- `Server/backend/env.example`에 환경 변수와 안전한 개발 기본값을 명시
- Django 의존성 없이 실행되는 정책·설정 계약 테스트 추가

## TDD/검증

| 검사 | 결과 |
|---|---|
| 보안 정책 RED | `security.py` 부재로 import 실패 확인 |
| 보안 정책 GREEN | `19 passed` (정책·설정 focused tests) |
| 전체 unit | `58 passed` |
| Python compileall | PASS |
| `git diff --check` | PASS |
| Django/MySQL runtime | BLOCKED: 현재 Python 환경에 Django 및 DB 연결이 없음 |
| coverage | BLOCKED: coverage/pytest-cov 미설치 |

## 평가 결과

- 보안 평가 서브 에이전트: HIGH 없음, 조건부 MEDIUM 1건
- 품질 평가 서브 에이전트: B1 승인(GOOD)
- 남은 MEDIUM: reverse proxy 뒤에서 caller가 전달한 `REMOTE_ADDR`를 loopback으로 잘못 제공하지 않도록 B2에서 trusted peer/proxy 정책과 통합 테스트를 추가해야 함

## 범위 경계

이 단계는 정책 primitive와 설정 fail-closed를 추가한 단계다. 아직 proxy route가 `resolve_actor()`를 호출하지 않으므로, 실제 API 인증·운영자 권한·대화/로그/metrics 소유권 필터가 해결됐다고 판정하지 않는다. 해당 연결은 R2-B2에서 수행한다.

## 롤백

새 `security.py`, 보안 계약 테스트, 증거 리포트를 제거하고 `settings.py` 및 `env.example` 변경을 기준 커밋으로 되돌린다. 기존 R1/R2-A 변경과 데이터베이스 migration은 건드리지 않는다.
