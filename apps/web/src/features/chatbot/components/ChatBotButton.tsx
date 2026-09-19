import React, { useState } from "react";
import { MessageCircle, X } from "lucide-react";
import { ChatBotPanel } from "./ChatBotPanel";

export const ChatBotButton: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleChatBot = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* 플로팅 챗봇 버튼 */}
      <div className="fixed bottom-32 right-6 z-[1000] md:bottom-28 md:right-8">
        <button
          onClick={toggleChatBot}
          className={`
            w-14 h-14 rounded-full shadow-lg transition-all duration-300 ease-in-out
            flex items-center justify-center text-white
            hover:scale-110 hover:shadow-xl
            ${
              isOpen
                ? "bg-red-500 hover:bg-red-600"
                : "bg-blue-500 hover:bg-blue-600"
            }
          `}
          title={isOpen ? "Close AI Chat" : "Open AI Chat"}
        >
          {isOpen ? <X size={24} /> : <MessageCircle size={24} />}
        </button>
      </div>

      {/* 챗봇 패널 */}
      {isOpen && (
        <ChatBotPanel isOpen={isOpen} onClose={() => setIsOpen(false)} />
      )}
    </>
  );
};
