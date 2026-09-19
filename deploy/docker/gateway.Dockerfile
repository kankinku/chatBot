FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-mysql-client \
    curl \
    pkg-config \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY services/gateway/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY services/gateway/ ./

ENV LANG=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8
ENV LC_CTYPE=ko_KR.UTF-8
ENV PYTHONUTF8=1

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
