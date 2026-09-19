# Data

Git에는 작은 fixture, QA 데이터, 도메인 사전과 schema만 저장한다.

PDF 원문, corpus, vector store, model artifact와 대용량 benchmark 결과는 저장소 밖의 runtime/artifact storage에서 관리한다. 관련 패턴은 root `.gitignore`에서 차단한다.
