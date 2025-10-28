# AI 챗봇 프론트엔드

이 프로젝트는 기존 IFRO_SEJONG 프로젝트의 챗봇 디자인을 복사한 독립적인 챗봇 애플리케이션입니다.

## 🚀 시작하기

### 설치
```bash
npm install
```

### 실행
```bash
npm start
```

## 📁 프로젝트 구조

```
src/
├── features/
│   └── chatbot/
│       └── components/
│           ├── ChatBotButton.tsx    # 플로팅 챗봇 버튼
│           └── ChatBotPanel.tsx     # 챗봇 패널
├── shared/
│   ├── components/
│   │   └── ui/
│   │       ├── FeedbackMessage.tsx  # 피드백 메시지 컴포넌트
│   │       └── FloatingLabelInput.tsx # 플로팅 라벨 입력 필드
│   ├── services/
│   │   └── chat.ts                 # 챗봇 API 서비스
│   └── utils/
│       ├── chatCache.ts            # 챗봇 캐시 유틸리티
│       └── debugUtils.ts           # 디버그 유틸리티
├── lib/
│   └── utils.ts                    # 클래스 병합 유틸리티
├── App.tsx                         # 메인 앱 컴포넌트
├── App.css                         # 앱 스타일
├── index.tsx                       # 앱 진입점
└── index.css                       # 글로벌 스타일
```

## 🎨 디자인 특징

- **플로팅 UI**: 우하단 고정 위치의 원형 버튼과 패널
- **반응형 디자인**: 모바일부터 데스크톱까지 모든 화면 크기 지원
- **Tailwind CSS**: 유틸리티 기반 스타일링
- **부드러운 애니메이션**: 호버 효과, 페이드 인/아웃, 로딩 애니메이션
- **상태 표시**: AI 상태, 캐시 정보, 신뢰도 등 실시간 피드백

## 🔧 기술 스택

- React 18
- TypeScript
- Tailwind CSS
- Lucide React (아이콘)
- Axios (HTTP 클라이언트)

## 🌐 API 설정

챗봇 서버 URL을 설정하려면 환경변수를 사용하세요:

```env
REACT_APP_API_URL=http://localhost:8000
```

## 📱 사용법

1. 애플리케이션을 실행합니다
2. 우하단의 파란색 챗봇 버튼을 클릭합니다
3. AI와 대화를 시작합니다
4. 닫기 버튼(X)을 클릭하여 챗봇을 닫습니다
