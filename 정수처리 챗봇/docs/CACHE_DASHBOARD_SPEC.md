## 캐시 대시보드 이벤트 스펙

### 이벤트 타입
- cache_hit, cache_miss, cache_expired, cache_put, cache_evict, cache_cleanup

### 공통 필드
- timestamp, key_prefix, cache_name(optional), ttl, size, session_id(optional)

### 메트릭
- hits, misses, hit_rate, evicted, expired, p95_latency(ms)

### 로깅
- 캐시 이벤트/메트릭은 `utils/unified_logger.py` 카테고리 PERFORMANCE 로 기록 (SSOT)


