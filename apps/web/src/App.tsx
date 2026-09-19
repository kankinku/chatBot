import React from 'react';
import './App.css';
import { ChatBotButton } from './features/chatbot/components/ChatBotButton';
import { ProcessPDFButton } from './features/chatbot/components/ProcessPDFButton';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>챗봇 테스트</h1>
        <p>우하단의 챗봇 버튼을 클릭하여 AI와 대화하세요!</p>
      </header>
      
      {/* PDF 처리 버튼 */}
      <ProcessPDFButton />
      
      {/* 챗봇 플로팅 버튼 */}
      <ChatBotButton />
    </div>
  );
}

export default App;
