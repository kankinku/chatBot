FROM python:3.10-slim

# GPU 지원을 위한 CUDA 런타임 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Python 경로 설정
ENV PYTHONPATH=/app/src:/app

# 포트 노출
EXPOSE 8000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# 애플리케이션 실행
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]