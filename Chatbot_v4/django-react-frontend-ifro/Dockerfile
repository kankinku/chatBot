FROM node:18-alpine

WORKDIR /app

# package.json과 package-lock.json 복사
COPY package*.json ./

# 의존성 설치
RUN npm install

# 소스 코드 복사
COPY . .

# App.optimized.tsx를 제거하고 간단한 App.tsx만 사용
RUN rm -f src/App.optimized.tsx src/App.original.tsx

# 포트 노출
EXPOSE 3000

# 개발 서버 실행
CMD ["npm", "start"]