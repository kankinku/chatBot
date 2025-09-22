## 로케일 운영 가이드

### 목적
- 플랫폼별 로케일 설정 실패 시 안전 기본값으로 진행하고 장애를 방지

### 권장 설정
- Linux: C.UTF-8
- Windows: Korean_Korea.UTF-8 (실패 시 en_US.UTF-8)
- 기타: en_US.UTF-8

### 실패 처리
- 예외는 잡고 경고 로그 남긴 후 기본값으로 진행
- 서비스 기동은 중단하지 않음

### 로깅
- 로케일 설정/폴백 로깅은 `utils/unified_logger.py` 사용 (SSOT)


