# 챗봇 로깅 가이드

## Docker Desktop에서 로그 확인 방법

### 1. 실시간 로그 확인
Docker Desktop에서 `chatbot-gpu` 컨테이너의 로그를 실시간으로 확인할 수 있습니다:

1. Docker Desktop 실행
2. Containers 탭에서 `chatbot-gpu` 컨테이너 선택
3. Logs 탭 클릭
4. 실시간으로 질문과 답변이 표시됩니다

### 2. 로그 파일 위치
호스트 시스템에서 로그 파일들을 직접 확인할 수 있습니다:

```
ollama-chatbot-api-ifro/logs/
├── chatbot_conversations.log    # 간단한 요약 로그
├── qa_detailed.log             # 상세한 질문/답변 로그
├── conversations.jsonl         # JSON 형식의 구조화된 로그
├── failed_answers.jsonl        # 실패한 답변 로그
└── llm_errors.log             # LLM 오류 로그
```

### 3. 로그 형식 설명

#### chatbot_conversations.log
```
2024-01-15 14:30:25 [INFO] 📥 질문 수신 | 모드: accuracy | 길이: 25자
2024-01-15 14:30:25 [INFO] 📝 질문 내용: 교통사고가 발생했을 때 어떻게 해야 하나요?
2024-01-15 14:30:28 [INFO] 📤 답변 생성 완료 | 신뢰도: 0.85 | 소스: 3개 | Fallback: False
2024-01-15 14:30:28 [INFO] 📄 답변 내용: 교통사고 발생 시 다음과 같이 대응하세요...
2024-01-15 14:30:28 [INFO] 💬 Q&A 완료 | 질문: 교통사고가 발생했을 때 어떻게 해야 하나요? | 답변길이: 156 | 신뢰도: 0.85
```

#### qa_detailed.log
```
2024-01-15 14:30:28 - ================================================================================
2024-01-15 14:30:28 - 🤖 질문: 교통사고가 발생했을 때 어떻게 해야 하나요?
2024-01-15 14:30:28 - ✅ 답변: 교통사고 발생 시 다음과 같이 대응하세요:
1. 즉시 차량을 안전한 곳으로 이동
2. 응급상황 시 119에 신고
3. 경찰서에 신고 (112)
4. 보험사에 연락
5. 현장 사진 촬영 및 증거 보전
2024-01-15 14:30:28 - 📊 신뢰도: 0.85 | 소스 수: 3 | Fallback: False
2024-01-15 14:30:28 - ================================================================================
```

### 4. 로그 모니터링 명령어

#### 실시간 로그 확인
```bash
# 간단한 요약 로그
tail -f ollama-chatbot-api-ifro/logs/chatbot_conversations.log

# 상세한 질문/답변 로그
tail -f ollama-chatbot-api-ifro/logs/qa_detailed.log

# JSON 형식 로그
tail -f ollama-chatbot-api-ifro/logs/conversations.jsonl
```

#### 로그 검색
```bash
# 특정 키워드가 포함된 질문 찾기
grep "교통사고" ollama-chatbot-api-ifro/logs/qa_detailed.log

# 신뢰도가 낮은 답변 찾기
grep "신뢰도: 0\.[0-5]" ollama-chatbot-api-ifro/logs/qa_detailed.log

# 오류 로그 확인
grep "ERROR" ollama-chatbot-api-ifro/logs/chatbot_conversations.log
```

### 5. 로그 로테이션
Docker Compose 설정에서 로그 파일 크기를 제한하여 디스크 공간을 절약합니다:
- 최대 파일 크기: 10MB
- 최대 파일 개수: 3개

### 6. 문제 해결

#### 로그가 보이지 않는 경우
1. 컨테이너가 실행 중인지 확인
2. 로그 볼륨이 올바르게 마운트되었는지 확인
3. 권한 문제가 없는지 확인

#### 로그 파일이 너무 큰 경우
```bash
# 로그 파일 크기 확인
ls -lh ollama-chatbot-api-ifro/logs/

# 오래된 로그 파일 삭제
rm ollama-chatbot-api-ifro/logs/chatbot_conversations.log.old
```

### 7. 로그 분석 도구
JSON 형식의 로그를 분석하려면 다음 도구들을 사용할 수 있습니다:

```bash
# jq를 사용한 JSON 로그 분석
cat ollama-chatbot-api-ifro/logs/conversations.jsonl | jq '.question'

# 신뢰도별 통계
cat ollama-chatbot-api-ifro/logs/conversations.jsonl | jq '.confidence' | sort -n
```
