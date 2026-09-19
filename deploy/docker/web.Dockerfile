FROM node:18-alpine AS builder

WORKDIR /app

COPY apps/web/package*.json ./
RUN npm install

COPY apps/web/ ./

ARG REACT_APP_API_URL=http://localhost:8001
ENV REACT_APP_API_URL=${REACT_APP_API_URL}

RUN npm run build

FROM nginx:alpine

COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
