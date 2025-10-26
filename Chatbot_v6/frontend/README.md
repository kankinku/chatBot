# 챗봇 v6 프론트엔드

정수처리 챗봇 v6의 React/TypeScript 프론트엔드입니다.

## 기능

- 🤖 AI 기반 대화형 인터페이스
- 💬 실시간 채팅
- 📊 답변 신뢰도 표시
- ⚡ 캐시 기능 (빠른 응답)
- 📱 반응형 디자인

## 설치

```bash
cd frontend
npm install
```

## 환경 설정

`.env` 파일을 생성하고 백엔드 API URL을 설정하세요:

```env
REACT_APP_API_URL=http://localhost:8000
```

## 실행

### 개발 모드

```bash
npm start
```

브라우저에서 [http://localhost:3000](http://localhost:3000)으로 접속합니다.

### 프로덕션 빌드

```bash
npm run build
```

빌드된 파일은 `build/` 폴더에 생성됩니다.

## 디렉토리 구조

```
src/
├── features/
│   └── chatbot/
│       └── components/
│           ├── ChatBotButton.tsx    # 챗봇 버튼
│           └── ChatBotPanel.tsx     # 챗봇 패널
├── shared/
│   ├── services/
│   │   └── chat.ts                  # API 통신
│   └── utils/
│       ├── chatCache.ts             # 캐시 관리
│       └── debugUtils.ts            # 디버그 유틸
├── App.tsx
├── App.css
├── index.tsx
└── index.css
```

## 기술 스택

- React 18
- TypeScript
- Tailwind CSS
- Axios
- Lucide React (아이콘)

## API 엔드포인트

- `POST /ask` - 질문 답변
- `GET /status` - AI 서비스 상태
- `GET /healthz` - 헬스 체크

