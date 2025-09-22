## 검색/벡터 로그 스키마

### 필드 정의
- session_id: 문자열
- route: "greeting" | "pdf_search" 등
- topK: 정수 (요청 상한)
- threshold: 실수 (유사도 임계값)
- store: "hybrid-faiss-first" | "chroma-only"
- results: 정수 (반환 청크 수)
- latency_ms: 실수
- error_code(optional): 문자열 (SEARCH_TIMEOUT 등)

### 기록 위치
- `utils/unified_logger.py`의 `JsonLogObserver`가 `logs/system.jsonl`에 기록 (SSOT)


