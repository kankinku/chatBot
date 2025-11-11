# new-chatbot Docker 환경

이 프로젝트는 GPU를 활용한 챗봇 시스템을 Docker로 구성한 것입니다.

## 🏗️ 프로젝트 구조

```
new-chatbot/
├── new-chatbot-backend-api/     # Django 백엔드 API (프록시 서버)
├── new-chatbot-front/           # React 프론트엔드
├── ollama-chatbot-api-ifro/     # FastAPI 챗봇 서버 (Ollama 연동)
├── docker-compose.yml           # Docker Compose 설정 (모든 스크립트 포함)
└── env.example                  # 환경 변수 예시
```

## 🚀 빠른 시작

### 1. 사전 요구사항

- Docker 및 Docker Compose 설치
- NVIDIA GPU 및 NVIDIA Container Toolkit 설치
- 최소 16GB RAM 권장

### 2. 환경 설정 (선택사항)

```bash
# 환경 변수 파일 생성 (기본값으로도 동작)
cp env.example .env
```

### 3. 서비스 시작

```bash
# 모든 서비스 시작 (권장)
docker-compose up --build -d

# 로그와 함께 시작
docker-compose up --build

# 백그라운드에서 시작
docker-compose up -d
```

## 🌐 서비스 접속 정보

- **프론트엔드**: http://localhost:3001
- **백엔드 API**: http://localhost:8001
- **챗봇 API**: http://localhost:8009
- **Ollama**: http://localhost:11435
- **MySQL**: localhost:3308

## 🔧 서비스 구성

### 1. 프론트엔드 (React)
- **포트**: 3001
- **기술**: React, TypeScript, Tailwind CSS
- **기능**: 챗봇 UI, 대화 인터페이스

### 2. 백엔드 API (Django)
- **포트**: 8001
- **기술**: Django, Django Ninja
- **기능**: 프론트엔드와 챗봇 서버 간 프록시

### 3. 챗봇 서버 (FastAPI)
- **포트**: 8009
- **기술**: FastAPI, Ollama, Sentence Transformers
- **기능**: AI 질문 답변, 문서 검색

### 4. Ollama 서버
- **포트**: 11435
- **기능**: LLM 모델 실행 (GPU 가속)
- **GPU 최적화**: 모든 레이어 GPU 실행

### 5. MySQL 데이터베이스
- **포트**: 3308
- **기능**: 챗봇 대화 기록, 로그, 메트릭 저장


## 📋 주요 명령어

### 서비스 관리
```bash
# 서비스 시작
docker-compose up -d

# 서비스 중지
docker-compose down

# 서비스 재시작
docker-compose restart

# 서비스 상태 확인
docker-compose ps
```

### 로그 확인
```bash
# 모든 서비스 로그
docker-compose logs -f

# 특정 서비스 로그
docker-compose logs -f chatbot
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f ollama
```

### 컨테이너 접속
```bash
# 챗봇 서버 접속
docker-compose exec chatbot bash

# 백엔드 서버 접속
docker-compose exec backend bash
```

## 🔧 GPU 최적화 설정

### Ollama GPU 설정
- `OLLAMA_GPU_LAYERS=-1`: 모든 레이어를 GPU에서 실행
- `OLLAMA_NUM_PARALLEL=4`: 병렬 처리 수 증가
- `OLLAMA_MAX_LOADED_MODELS=2`: 동시 로드 모델 수
- `OLLAMA_FLASH_ATTENTION=1`: Flash Attention 활성화

### 메모리 설정
- Ollama: 최대 16GB 메모리 할당
- 챗봇 서버: 최대 8GB 메모리 할당
- 프론트엔드: 최대 4GB 메모리 할당

## 🐛 문제 해결

### GPU 인식 문제
```bash
# NVIDIA Container Toolkit 설치 확인
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 메모리 부족
- Docker Desktop에서 메모리 할당량 증가
- `docker-compose.yml`에서 메모리 제한 조정

### 포트 충돌
- `docker-compose.yml`에서 포트 번호 변경
- 기존 실행 중인 서비스 확인

### 모델 다운로드 실패
```bash
# Ollama 컨테이너에서 직접 모델 다운로드
docker-compose exec ollama ollama pull llama2
```

## 📊 모니터링

### 서비스 상태 확인
```bash
# 챗봇 서버 상태
curl http://localhost:8009/status

# Ollama 모델 상태
curl http://localhost:11435/api/tags
```

### 리소스 사용량 확인
```bash
# 컨테이너 리소스 사용량
docker stats
```

## 🔄 업데이트

### 코드 변경 후 재빌드
```bash
docker-compose up --build -d
```

### 특정 서비스만 재빌드
```bash
docker-compose up --build -d chatbot
```

## 📊 대화 기록 및 로그 관리

### 데이터베이스 저장 정보
- **대화 세션**: 사용자별 대화 세션 관리
- **메시지 기록**: 질문/답변 내용, 신뢰도, 처리시간
- **시스템 로그**: 모든 챗봇 활동 로그
- **성능 메트릭**: 응답시간, 성공률, 평균 신뢰도

### API 엔드포인트

#### 대화 기록 관리
```bash
# 모든 대화 세션 목록 조회
GET http://localhost:8001/api/chatbot/conversations

# 특정 대화 세션 상세 조회
GET http://localhost:8001/api/chatbot/conversations/{session_id}

# 대화 세션 삭제
DELETE http://localhost:8001/api/chatbot/conversations/{session_id}
```

#### 로그 및 메트릭 조회
```bash
# 시스템 로그 조회
GET http://localhost:8001/api/chatbot/logs?level=INFO&limit=100

# 성능 메트릭 조회
GET http://localhost:8001/api/chatbot/metrics
```

### 로그 파일
- 챗봇 대화 로그: `ollama-chatbot-api-ifro/logs/conversations.jsonl`
- 상세 질문/답변 로그: `ollama-chatbot-api-ifro/logs/qa_detailed.log`
- Django 로그: `new-chatbot-backend-api/chatbot.log`
- 데이터베이스 로그: MySQL에 저장된 구조화된 로그

## 🆘 지원

문제가 발생하면 다음을 확인하세요:

1. Docker 및 NVIDIA Container Toolkit 설치 상태
2. GPU 메모리 사용량
3. 서비스 로그 (`docker-compose logs -f`)
4. 네트워크 연결 상태
