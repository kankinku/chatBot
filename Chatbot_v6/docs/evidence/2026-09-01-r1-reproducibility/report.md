# R1 실행·검증 재현성 단계 보고서

- 단계: R1 실행·검증 재현성
- 실행일: 2026-09-01
- 기준 브랜치: `codex/r1-reproducibility`
- 기준 커밋: `5d40acb chore: ignore local worktrees`
- 작업 원칙: 오프라인 unit/contract 중심, 모델 다운로드·외부 서비스·실제 데이터 변경 없음

## 목표

평가 및 interactive CLI가 프로젝트의 로컬 `scripts` 패키지를 확실히 사용하도록 하고, 기존 unit 실패 중 테스트 기대가 잘못된 항목과 질문 유형 우선순위 충돌을 회귀 테스트로 고정한다.

## RED

- `tests/unit/test_bm25_retriever.py::test_char_ngrams`: 3글자 입력에서 5-gram을 기대해 실패
- `tests/unit/test_question_analyzer.py`: `운영 현황은?`이 `procedural`로 분류되어 실패
- 새 CLI entrypoint/import 경계 테스트: 설치된 `scripts` 패키지가 로컬 모듈을 가릴 수 있는 조건을 검증 대상으로 고정

## 변경 내용

1. `scripts/__init__.py`를 추가해 프로젝트 `scripts`를 명시적 패키지로 만들었다.
2. 짧은 입력은 입력 길이를 넘는 n-gram을 만들지 않는다는 테스트와, 충분히 긴 입력에는 5-gram이 생성된다는 테스트로 기존 assertion을 교정했다.
3. `운영 현황/상태/정보`, `현재 상태/현황`, `실시간 상태/현황`을 구체적 운영 상태 질문으로 먼저 판정하고 `운영 방법`과 `현재 운영 방법`은 procedural로 유지했다.
4. 세 CLI의 직접 `--help` 실행 및 로컬 `scripts.unified_evaluation` 해석을 subprocess 테스트로 추가했다.

## 검증

| 검사 | 실행 명령 | 결과 | 비고 |
|---|---|---|---|
| Python 3.10.2 unit | `python -m pytest tests/unit -q -o addopts= -p no:cacheprovider` | PASS: 37 passed | 전체 unit 실행 |
| 전체 테스트 수집 | `python -m pytest tests --collect-only -q -o addopts= -p no:cacheprovider` | PASS: 47 collected | integration은 모델·저장소 의존성 때문에 수집만 수행 |
| 평가 CLI | `python scripts/evaluate_qa_unified.py --help` | PASS: exit 0, `usage:` 출력 | 모델·corpus 로드 없이 parser help만 실행 |
| interactive CLI | `python scripts/test_chatbot_interactive.py --help` | PASS: exit 0, `usage:` 출력 | 모델·corpus 로드 없이 parser help만 실행 |
| corpus CLI | `python scripts/build_corpus.py --help` | PASS: exit 0, `usage:` 출력 | 실제 PDF 처리는 수행하지 않음 |
| Python AST | `python -c "import ast; ..."` | PASS: 87 files parsed | 구문 검증이며 타입·실행 검증은 아님 |
| 커버리지 80% | `python -m pytest ... --cov=...` | BLOCKED | 현재 Python 환경에 `coverage`/`pytest-cov` 미설치. 임의의 수치로 대체하지 않음 |

재현에 사용한 핵심 명령의 원시 결과 요약:

```text
python -m pytest tests/unit -q -o addopts= -p no:cacheprovider
................................... [100%]
37 passed in 2.88s

python -m pytest tests --collect-only -q -o addopts= -p no:cacheprovider
47 tests collected in 0.04s

python scripts/evaluate_qa_unified.py --help
usage: evaluate_qa_unified.py ...

python scripts/test_chatbot_interactive.py --help
usage: test_chatbot_interactive.py ...

python scripts/build_corpus.py --help
usage: build_corpus.py ...

python -c "import ast; ..."
87 Python files parsed
```

## 범위 밖 검증

통합 RAG, 실제 LLM/Ollama, Chroma index, Docker build/up, Django/FastAPI 기동, 브라우저 E2E, load/stress/soak은 R1 범위에서 실행하지 않았다. 이 항목들은 설계서의 R2 이후 gate에서 별도 환경과 증거로 검증한다.

## 리뷰 판정

- 설계 적합성 리뷰: PASS
- 코드 품질·회귀 리뷰: PASS
- 남은 고우선 결함: 없음
- 커버리지 도구 부재: BLOCKED 상태 유지

## 롤백

- 기능 변경 rollback: `scripts/__init__.py`, analyzer 변경, 새 테스트 파일을 함께 되돌린다.
- 테스트 기대 변경 rollback: 기존 테스트를 복원하지 않고, n-gram 계약을 먼저 합의한 뒤 fixture를 수정한다.
- 작업 공간 rollback: 이 브랜치의 R1 커밋을 폐기해도 `main`의 기존 감사·설계 문서와 원본 데이터는 삭제되지 않는다.
