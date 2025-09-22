# GPU 지원을 위한 Docker 설정
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 3.9를 기본으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치 (GPU 지원 포함)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# GPU 지원을 위한 추가 패키지
RUN pip3 install --no-cache-dir psutil torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 애플리케이션 코드 복사
COPY . .

# GPU 및 메모리 최적화 환경 변수 설정
ENV MAX_MODEL_MEMORY_GB=8.0
ENV MEMORY_WARNING_THRESHOLD=75.0
ENV MEMORY_CRITICAL_THRESHOLD=85.0
ENV PDF_MAX_MEMORY_GB=2.0

# GPU 설정
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 메모리 제한 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 메모리 최적화를 위한 JVM 설정 (필요시)
ENV JAVA_OPTS="-Xmx2g -Xms1g"

# 포트 노출
EXPOSE 8000

# GPU 지원 엔트리포인트 실행
ENTRYPOINT ["./docker-entrypoint.sh"]
