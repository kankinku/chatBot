import React from 'react';
import { MessageCircle } from 'lucide-react';

interface ChatBotButtonProps {
  onClick: () => void;
}

export const ChatBotButton: React.FC<ChatBotButtonProps> = ({ onClick }) => {
  return (
    <button
      onClick={onClick}
      className="
        fixed bottom-6 right-6 w-16 h-16 bg-blue-600 hover:bg-blue-700
        text-white rounded-full shadow-lg hover:shadow-xl
        transition-all duration-200 ease-in-out
        flex items-center justify-center z-[998]
        transform hover:scale-110
      "
      title="챗봇 열기"
    >
      <MessageCircle size={28} />
    </button>
  );
};

