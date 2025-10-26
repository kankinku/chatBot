import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Bot,
  User,
  Minimize2,
  Info,
  AlertCircle,
  Database,
  Zap,
} from "lucide-react";
import {
  sendAIChatMessage,
  checkAIServiceStatus,
  getChatCacheInfo,
  clearChatCache,
} from "../../../shared/services/chat";

interface Message {
  id: string;
  content: string;
  sender: "user" | "bot";
  timestamp: Date;
  metadata?: {
    confidence_score?: number;
    generation_time?: number;
    from_cache?: boolean;
  };
}

interface ChatBotPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ChatBotPanel: React.FC<ChatBotPanelProps> = ({
  isOpen,
  onClose,
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [aiStatus, setAiStatus] = useState<{
    ai_available: boolean;
    model_loaded: boolean;
    total_pdfs: number;
    total_chunks: number;
  }>({
    ai_available: false,
    model_loaded: false,
    total_pdfs: 0,
    total_chunks: 0,
  });
  const [showInfo, setShowInfo] = useState(false);
  const [cacheInfo, setCacheInfo] = useState<{
    totalEntries: number;
    sizeInKB: number;
    hitRate: number;
  }>({
    totalEntries: 0,
    sizeInKB: 0,
    hitRate: 0,
  });
  const [useCache, setUseCache] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // 메시지 스크롤
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // AI 상태 확인
  const checkAIStatus = async () => {
    try {
      const status = await checkAIServiceStatus();
      setAiStatus(status);
    } catch (error) {
      console.error('AI 상태 확인 중 오류:', error);
    }
  };

  // 패널이 열릴 때 초기화
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      if (inputRef.current) {
        setTimeout(() => {
          inputRef.current?.focus();
        }, 100);
      }

      const initializeChatbot = async () => {
        try {
          await checkAIStatus();
          setCacheInfo(getChatCacheInfo());

          const welcomeMessage: Message = {
            id: "1",
            content: `안녕하세요! 정수처리 챗봇 v6입니다! 🤖\n\n저는 PDF 문서 기반 지능형 AI로, 정수처리 관련 질문에 답변해드립니다.\n\n💾 캐시 기능이 활성화되어 빠른 답변을 제공합니다.\n📊 신뢰도 점수로 답변 품질을 확인할 수 있습니다.\n\n무엇이든 물어보세요!`,
            sender: "bot",
            timestamp: new Date(),
          };
          setMessages([welcomeMessage]);
        } catch (error) {
          console.error("챗봇 초기화 중 오류:", error);
        }
      };

      initializeChatbot();
    }
  }, [isOpen]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage.trim(),
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await sendAIChatMessage(
        userMessage.content,
        useCache
      );

      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: response.answer,
        sender: "bot",
        timestamp: new Date(),
        metadata: {
          confidence_score: response.confidence,
          generation_time: response.processing_time,
          from_cache: false,
        },
      };

      setMessages((prev) => [...prev, botResponse]);
      setCacheInfo(getChatCacheInfo());
    } catch (error) {
      console.error("AI 응답 생성 중 오류:", error);
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        content:
          error instanceof Error
            ? error.message
            : "죄송합니다. 일시적인 오류가 발생했습니다.",
        sender: "bot",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearCache = () => {
    const removedCount = clearChatCache();
    setCacheInfo(getChatCacheInfo());

    const notificationMessage: Message = {
      id: (Date.now() + 1).toString(),
      content: `캐시가 정리되었습니다. ${removedCount}개의 항목이 제거되었습니다.`,
      sender: "bot",
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, notificationMessage]);
  };

  const getStatusColor = () => {
    if (aiStatus.ai_available && aiStatus.model_loaded) return "text-green-600";
    if (aiStatus.ai_available) return "text-yellow-600";
    return "text-red-600";
  };

  const getStatusText = () => {
    if (aiStatus.ai_available && aiStatus.model_loaded) return "AI 모델 준비됨";
    if (aiStatus.ai_available) return "AI 서비스 연결됨";
    return "AI 서비스 연결 안됨";
  };

  const getStatusDotColor = () => {
    if (aiStatus.ai_available && aiStatus.model_loaded) return "bg-green-500";
    if (aiStatus.ai_available) return "bg-yellow-500 animate-pulse";
    return "bg-red-500";
  };

  if (!isOpen) return null;

  return (
    <div
      className={`
      fixed bottom-32 right-6 w-[480px] h-[600px] bg-white rounded-lg shadow-2xl
      border border-gray-200 z-[999] flex flex-col
      transform transition-all duration-300 ease-in-out
      md:bottom-28 md:right-8 md:w-[480px]
      max-w-[calc(100vw-3rem)] max-h-[calc(100vh-8rem)]
      ${isOpen ? "scale-100 opacity-100" : "scale-95 opacity-0"}
    `}
    >
      {/* 헤더 */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-blue-50 rounded-t-lg">
        <div className="flex items-center space-x-2">
          <Bot className="w-6 h-6 text-blue-600" />
          <div>
            <h3 className="font-semibold text-gray-800">AI 어시스턴트</h3>
            <div className="flex items-center space-x-1 text-xs">
              <div
                className={`w-2 h-2 rounded-full ${getStatusDotColor()}`}
              ></div>
              <span className={getStatusColor()}>{getStatusText()}</span>
              {useCache && (
                <>
                  <span className="text-gray-400">•</span>
                  <Zap size={10} className="text-yellow-500" />
                  <span className="text-yellow-600">캐시 활성화</span>
                </>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-1">
          <button
            onClick={() => setShowInfo(!showInfo)}
            className="p-1 hover:bg-gray-200 rounded transition-colors"
            title="AI 상태 정보"
          >
            <Info size={16} className="text-gray-600" />
          </button>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-200 rounded transition-colors"
            title="채팅 최소화"
          >
            <Minimize2 size={18} className="text-gray-600" />
          </button>
        </div>
      </div>

      {/* AI 상태 정보 패널 */}
      {showInfo && (
        <div className="p-3 bg-gray-50 border-b border-gray-200 text-xs">
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>AI 모델:</span>
              <span
                className={
                  aiStatus.model_loaded ? "text-green-600" : "text-red-600"
                }
              >
                {aiStatus.model_loaded ? "로드됨" : "로드 안됨"}
              </span>
            </div>
            <div className="flex justify-between">
              <span>PDF 문서:</span>
              <span>{aiStatus.total_pdfs}개</span>
            </div>
            <div className="flex justify-between">
              <span>문서 청크:</span>
              <span>{aiStatus.total_chunks}개</span>
            </div>

            {/* 캐시 정보 */}
            <div className="border-t pt-2 mt-2">
              <div className="flex items-center space-x-1 mb-1">
                <Database size={10} className="text-blue-500" />
                <span className="font-medium">캐시 정보</span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span>캐시된 답변:</span>
                  <span>{cacheInfo.totalEntries}개</span>
                </div>
                <div className="flex justify-between">
                  <span>캐시 크기:</span>
                  <span>{cacheInfo.sizeInKB}KB</span>
                </div>
                <div className="flex justify-between">
                  <span>캐시 히트율:</span>
                  <span>{cacheInfo.hitRate}%</span>
                </div>
              </div>
            </div>

            {/* 캐시 제어 */}
            <div className="border-t pt-2 mt-2">
              <div className="flex items-center justify-between">
                <label className="flex items-center space-x-1">
                  <input
                    type="checkbox"
                    checked={useCache}
                    onChange={(e) => setUseCache(e.target.checked)}
                    className="w-3 h-3"
                  />
                  <span>캐시 사용</span>
                </label>
                <button
                  onClick={handleClearCache}
                  className="text-xs text-red-600 hover:text-red-800"
                  title="캐시 정리"
                >
                  캐시 정리
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 메시지 영역 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`flex items-start space-x-2 max-w-[80%] ${
                message.sender === "user"
                  ? "flex-row-reverse space-x-reverse"
                  : ""
              }`}
            >
              <div
                className={`
                w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0
                ${message.sender === "user" ? "bg-blue-500" : "bg-gray-300"}
              `}
              >
                {message.sender === "user" ? (
                  <User size={16} className="text-white" />
                ) : (
                  <Bot size={16} className="text-gray-600" />
                )}
              </div>
              <div className="flex flex-col">
                <div
                  className={`
                  p-3 rounded-lg whitespace-pre-wrap
                  ${
                    message.sender === "user"
                      ? "bg-blue-500 text-white rounded-br-none"
                      : "bg-gray-100 text-gray-800 rounded-bl-none"
                  }
                `}
                >
                  {message.content}
                </div>
                {/* AI 메타데이터 표시 */}
                {message.sender === "bot" && message.metadata && (
                  <div className="mt-1 text-xs text-gray-500 space-y-1">
                    {message.metadata.confidence_score !== undefined && (
                      <div className="flex items-center space-x-2">
                        <span>신뢰도:</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-1">
                          <div
                            className="bg-green-500 h-1 rounded-full"
                            style={{
                              width: `${
                                message.metadata.confidence_score * 100
                              }%`,
                            }}
                          ></div>
                        </div>
                        <span>
                          {Math.round(message.metadata.confidence_score * 100)}%
                        </span>
                      </div>
                    )}
                    <div className="flex items-center space-x-4">
                      {message.metadata.generation_time !== undefined && (
                        <span>
                          처리시간:{" "}
                          {message.metadata.generation_time.toFixed(1)}초
                        </span>
                      )}
                      {message.metadata.from_cache && (
                        <div className="flex items-center space-x-1 text-yellow-600">
                          <Zap size={10} />
                          <span>캐시됨</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">
                <Bot size={16} className="text-gray-600" />
              </div>
              <div className="bg-gray-100 p-3 rounded-lg rounded-bl-none">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0.1s" }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0.2s" }}
                    ></div>
                  </div>
                  <span className="text-sm text-gray-600">
                    AI가 생각 중...
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 입력 영역 */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="AI에게 질문하세요..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className={`
              px-4 py-2 rounded-lg transition-colors
              ${
                inputMessage.trim() && !isLoading
                  ? "bg-blue-500 hover:bg-blue-600 text-white"
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }
            `}
          >
            <Send size={18} />
          </button>
        </div>
        <div className="flex items-center justify-between mt-2">
          <p className="text-xs text-gray-500">
            Enter를 눌러 전송, Shift+Enter로 줄바꿈
          </p>
          {useCache && (
            <div className="flex items-center space-x-1 text-xs text-yellow-600">
              <Zap size={12} />
              <span>캐시 활성화</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

