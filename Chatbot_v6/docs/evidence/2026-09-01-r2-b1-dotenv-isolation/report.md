# R2-B1 dotenv 탐색 격리 보완 보고서

- 구분: `bb01b9e` 독립 평가 FAIL 후 추가 보완
- 실행일: 2026-09-01
- 브랜치: `codex/r1-reproducibility`
- 범위: 실제 settings import subprocess가 host/repository `.env`에 영향을 받지 않도록 source tree 격리

## 독립 평가 FAIL 원인

`bb01b9e` 재평가에서 임시 cwd만으로는 `python-decouple`이 `settings.py` 경로의 상위 디렉터리 `.env`를 탐색할 수 있다는 MEDIUM 지적을 받았다.

## 변경

- subprocess 실행 시 PATH/SystemRoot/WINDIR/TEMP/TMP 및 테스트 입력만 유지
- backend의 `chatbot_backend` source package를 `.env` 상위 경로가 없는 임시 tree로 복사
- 임시 tree를 `PYTHONPATH`로 지정해 실제 `chatbot_backend.settings` 모듈을 격리된 경로에서 import
- 기존의 environment/secret/password 독립 runtime 테스트는 유지

## 검증

| 검사 | 결과 |
|---|---|
| settings contract | `11 passed` |
| 전체 unit | `89 passed` |
| 전체 pytest | `99 passed in 142.40s` |
| Python compileall | PASS |
| `git diff --check` | PASS |

이 추가 변경은 테스트 격리와 증거 리포트만 수정하며 application runtime은 변경하지 않는다. 독립 평가 PASS 전에는 merge하지 않는다.
