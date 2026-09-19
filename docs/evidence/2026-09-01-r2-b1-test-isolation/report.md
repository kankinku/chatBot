# R2-B1 runtime 테스트 격리 보완 보고서

- 구분: `b52ac50` 독립 평가 FAIL 후 추가 보완
- 실행일: 2026-09-01
- 브랜치: `codex/r1-reproducibility`
- 범위: Django settings subprocess의 환경 격리와 개별 누락 검증

## 독립 평가 FAIL 원인

`b52ac50` 재평가에서 subprocess가 host 환경과 repository root를 상속할 수 있고, secret/password 누락이 각각 독립 검증되지 않으며, secure fixture가 반복 문자라는 지적을 받았다.

## 변경

- subprocess 환경을 PATH/SystemRoot/TEMP/TMP와 테스트 입력 및 backend PYTHONPATH만 포함하도록 제한
- `.env` 오염을 방지하기 위해 임시 cwd에서 실제 `chatbot_backend.settings` import
- `SECRET_KEY` 미설정 production 실패 테스트 추가
- `MYSQL_PASSWORD` 미설정 production 실패 테스트 추가
- `secrets.token_urlsafe(48)` 기반 fixture 사용

## 검증

| 검사 | 결과 |
|---|---|
| 설정 runtime focused | `24 passed` |
| 전체 unit | `89 passed` |
| 전체 pytest | `99 passed in 140.05s` |
| Python compileall | PASS |
| `git diff --check` | PASS |

이 추가 변경은 테스트 코드와 증거 리포트만 포함하며, application runtime 동작은 변경하지 않는다. 독립 평가 PASS 전에는 merge하거나 다음 커밋 평가로 넘어가지 않는다.
