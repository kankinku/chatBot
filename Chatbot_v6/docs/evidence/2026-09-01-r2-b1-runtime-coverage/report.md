# R2-B1 runtime settings 검증 보완 보고서

- 구분: 독립 평가 FAIL 후 추가 보완 커밋
- 실행일: 2026-09-01
- 브랜치: `codex/r1-reproducibility`
- 보완 대상: `c80d755 fix: fail closed on deployment configuration`
- 범위: 실제 `chatbot_backend.settings` subprocess import를 통한 fail-closed 검증

## 독립 평가 FAIL 원인

보완 커밋의 validator 로직은 통과했지만, 실제 Django settings module을 production 환경에서 import하는 테스트가 없다는 MEDIUM 지적을 받았다.

## 변경

- `tests/unit/test_backend_security_contracts.py`에 dependency-backed subprocess 테스트 추가
  - ENVIRONMENT 미설정 import 실패
  - production secret placeholder import 실패
  - production `MYSQL_PASSWORD=1234` import 실패
  - 명시적 보안 production 설정 import 성공
- 테스트는 `PYTHONPATH=Server/backend`로 실제 `chatbot_backend.settings`를 import하고 stderr/exit code를 검증한다.

## 검증

| 검사 | 결과 |
|---|---|
| 설정/runtime focused | `22 passed` |
| 전체 unit | `87 passed` |
| 전체 pytest | `97 passed in 140.09s` |
| Django `manage.py check` | PASS, 0 issues |
| `makemigrations --check --dry-run` | PASS, No changes detected |
| Python compileall | PASS |
| `git diff --check` | PASS |
| migration history DB 확인 | BLOCKED/WARN: local MySQL 미기동 |

## 재평가 게이트

이 추가 커밋은 runtime settings 검증을 포함한다. 독립 code-review 서브 에이전트의 PASS 전에는 merge하거나 다음 커밋 평가로 넘어가지 않는다.
