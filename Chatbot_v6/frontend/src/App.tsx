import React, { useState } from 'react';
import './App.css';
import { ChatBotButton } from './features/chatbot/components/ChatBotButton';
import { ChatBotPanel } from './features/chatbot/components/ChatBotPanel';

function App() {
  const [isChatOpen, setIsChatOpen] = useState(false);

  return (
    <div className="App min-h-screen bg-gray-50">
      <header className="bg-blue-600 text-white p-4">
        <div className="container mx-auto">
          <h1 className="text-2xl font-bold">정수처리 챗봇 v6</h1>
          <p className="text-sm mt-1">AI 기반 정수장 관리 어시스턴트</p>
        </div>
      </header>

      <main className="container mx-auto p-4">
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">챗봇 v6 정보</h2>
          <div className="space-y-2 text-gray-700">
            <p>✅ RAG (Retrieval-Augmented Generation) 기반</p>
            <p>✅ 성능 최적화 (비동기 처리, 배치 임베딩)</p>
            <p>✅ 실시간 모니터링 (Prometheus 메트릭)</p>
            <p>✅ 답변 품질 자동 검증</p>
            <p>✅ 적응형 컨텍스트 선택</p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">사용 방법</h2>
          <ol className="list-decimal list-inside space-y-2 text-gray-700">
            <li>오른쪽 하단의 챗봇 아이콘을 클릭하세요</li>
            <li>정수처리 관련 질문을 입력하세요</li>
            <li>AI가 문서 기반으로 정확한 답변을 제공합니다</li>
          </ol>
        </div>
      </main>

      <ChatBotButton onClick={() => setIsChatOpen(true)} />
      <ChatBotPanel isOpen={isChatOpen} onClose={() => setIsChatOpen(false)} />
    </div>
  );
}

export default App;

