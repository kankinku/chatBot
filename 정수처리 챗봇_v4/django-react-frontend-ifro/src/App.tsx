import React from "react";
import { ChatBotPanel } from "./features/chatbot/components/ChatBotPanel";

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <ChatBotPanel isOpen={true} onClose={() => {}} />
    </div>
  );
}

export default App;
