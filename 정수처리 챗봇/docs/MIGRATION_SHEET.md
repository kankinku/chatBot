## 로깅 마이그레이션 시트 (업데이트)

### 대상
- `utils/chatbot_logger.py` → `utils/unified_logger.py`

### 단계별 체크리스트
1. 호출부 인식: `grep -R "chatbot_logger"`
2. 대체 매핑 적용(로그 레벨/카테고리 설정)
3. 세션ID 전파 코드 추가
4. 보안정보 마스킹 확인
5. 기능 플래그로 병행기간 운영(rollback 대비)
6. 구 코드 제거 후 재빌드/테스트

### 변경 요약
- 제거: `core/utils/optimized_logger.py`
- 수정: `run_server.py`의 `get_optimized_logger` 의존성 제거

### 검증 지표
- 5xx 비율, p95 응답, 로그 누락률, 에러 재현율


