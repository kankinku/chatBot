## print 제거 및 세션ID 전파 가이드 (업데이트)

### print 제거
- 디버깅/정보 출력은 `unified_logger` 사용 (SSOT)
- 모듈/카테고리/메타데이터를 활용해 구조화 로깅

### 세션ID 전파
- 세션 생성 시: `unified_logger.set_session_id(session_id)`
- 요청 처리 중 동일 세션 유지
- 비동기 태스크에서도 세션ID 설정 필요 시 컨텍스트에서 재설정

### 예시 (SSOT)
```python
from utils.unified_logger import unified_logger, LogCategory

unified_logger.set_session_id(session_id)
unified_logger.info("vector_search", LogCategory.PERFORMANCE, "api.endpoints", metadata={"topK": 5})
```


