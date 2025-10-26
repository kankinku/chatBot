# Build stage
FROM node:18-alpine as build

WORKDIR /app

# 의존성 파일 복사
COPY package*.json ./

# 의존성 설치
RUN npm ci

# 소스 코드 복사
COPY . .

# 환경 변수 설정 (빌드 타임)
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=$REACT_APP_API_URL

# 프로덕션 빌드
RUN npm run build

# Production stage
FROM nginx:alpine

# Nginx 설정 복사
COPY --from=build /app/build /usr/share/nginx/html

# Nginx 기본 설정 제거 및 커스텀 설정 추가
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

