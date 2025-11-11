## 로깅 일원화 계획 (chatbot_logger → unified_logger)

### 목표
- 관측성 SSOT 확보, 로그 포맷 표준화, 성능/보안/검색 로그 일관화

### 범위 및 일정 (업데이트)
- 1단계(완료): `utils/unified_logger.py`만 운영 로깅 소스로 사용 (SSOT)
- 2단계(완료): `core/utils/optimized_logger.py` 제거, `run_server.py` 의존성 제거
- 3단계(진행): `chatbot_logger`는 세션 단계 로그에 한정, 운영 로그는 `unified_logger`로 일원화

### 대체 매핑표 (예시)
- chatbot_logger.log_step → unified_logger.performance/info (category: PERFORMANCE)
- chatbot_logger.log_question → unified_logger.info (category: QUERY)
- chatbot_logger.log_error → unified_logger.error (category: SYSTEM/SECURITY)
- chatbot_logger.log_greeting → unified_logger.info (category: MODEL)

### 파일/경로
- 운영 로그: `logs/unified.log`, `logs/system.jsonl`

### 표준 필드
- 공통: timestamp, level, category, module, message, session_id, execution_time_ms, metadata
- 세션 전파: `unified_logger.set_session_id(session_id)` 호출로 일관 적용

### 운영 가이드
- 운영/시스템/보안/성능: unified_logger 사용
- 세션 단계(뷰어 대상): chatbot_logger 유지
- print() 출력 금지(아래 가이드 참조)
- 보안 민감정보(토큰/키/PII) 마스킹 의무

### 변경 요약
- 제거: `core/utils/optimized_logger.py`
- 수정: `run_server.py`에서 `get_optimized_logger` 제거


