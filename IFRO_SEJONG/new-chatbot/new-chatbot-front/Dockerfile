FROM node:18-alpine

WORKDIR /app

# 패키지 파일 복사 및 의존성 설치
COPY package*.json ./
RUN npm install

# 애플리케이션 코드 복사
COPY . .

# 한글 지원 설정
ENV LANG=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8
ENV LC_CTYPE=ko_KR.UTF-8

# 포트 노출
EXPOSE 3000

# 개발 서버 실행
CMD ["npm", "start"]
