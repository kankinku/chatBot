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
    question_type?: string;
    generation_time?: number;
    model_name?: string;
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
    oldestEntry: string;
    newestEntry: string;
    hitRate: number;
  }>({
    totalEntries: 0,
    sizeInKB: 0,
    oldestEntry: "없음",
    newestEntry: "없음",
    hitRate: 0,
  });
  const [useCache, setUseCache] = useState(true);
  const [isCheckingStatus, setIsCheckingStatus] = useState(false);
  const [lastStatusChange, setLastStatusChange] = useState<Date | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const statusCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // 메시지 스크롤
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // AI 상태 확인 함수
  const checkAIStatus = async () => {
    if (isCheckingStatus) return; // 이미 확인 중이면 스킵
    
    setIsCheckingStatus(true);
    try {
      const status = await checkAIServiceStatus();
      setAiStatus(prevStatus => {
        // 상태가 변경되었는지 확인
        const hasChanged = 
          prevStatus.ai_available !== status.ai_available ||
          prevStatus.model_loaded !== status.model_loaded ||
          prevStatus.total_pdfs !== status.total_pdfs ||
          prevStatus.total_chunks !== status.total_chunks;
        
        if (hasChanged) {
          console.log('AI 상태 변경 감지:', status);
          setLastStatusChange(new Date());
          
          // 상태 변경 알림 메시지 추가
          const statusMessage: Message = {
            id: `status-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            content: getStatusChangeMessage(prevStatus, status),
            sender: "bot",
            timestamp: new Date(),
          };
          setMessages(prev => [...prev, statusMessage]);
        }
        
        return status;
      });
    } catch (error) {
      console.error('AI 상태 확인 중 오류:', error);
      // 오류 시에도 상태를 업데이트하여 연결 안됨 상태로 표시
      setAiStatus(prevStatus => ({
        ...prevStatus,
        ai_available: false,
        model_loaded: false,
      }));
    } finally {
      setIsCheckingStatus(false);
    }
  };

  // 주기적 상태 확인 설정
  useEffect(() => {
    if (isOpen) {
      // 즉시 한 번 확인
      checkAIStatus();
      
      // 5초마다 상태 확인
      statusCheckIntervalRef.current = setInterval(checkAIStatus, 5000);
      
      return () => {
        if (statusCheckIntervalRef.current) {
          clearInterval(statusCheckIntervalRef.current);
          statusCheckIntervalRef.current = null;
        }
      };
    }
  }, [isOpen]);

  // 상태 변경 메시지 생성 함수
  const getStatusChangeMessage = (prevStatus: any, newStatus: any): string => {
    const now = new Date().toLocaleTimeString();
    
    if (!prevStatus.ai_available && newStatus.ai_available) {
      if (newStatus.model_loaded) {
        return `🟢 AI 서비스가 연결되었습니다! (${now})\n모델이 준비되어 정확한 답변을 제공할 수 있습니다.`;
      } else {
        return `🟡 AI 서비스가 연결되었습니다! (${now})\n모델 로딩 중입니다. 잠시 후 더 정확한 답변을 받으실 수 있습니다.`;
      }
    }
    
    if (prevStatus.ai_available && !newStatus.ai_available) {
      return `🔴 AI 서비스 연결이 끊어졌습니다. (${now})\n기본 모드로 전환됩니다.`;
    }
    
    if (prevStatus.ai_available && newStatus.ai_available && 
        !prevStatus.model_loaded && newStatus.model_loaded) {
      return `🟢 AI 모델이 로드되었습니다! (${now})\n이제 더 정확한 답변을 제공할 수 있습니다.`;
    }
    
    if (prevStatus.ai_available && newStatus.ai_available && 
        prevStatus.model_loaded && !newStatus.model_loaded) {
      return `🟡 AI 모델이 언로드되었습니다. (${now})\n기본 모드로 전환됩니다.`;
    }
    
    return `🔄 AI 서비스 상태가 업데이트되었습니다. (${now})`;
  };

  // 패널이 열릴 때 초기화 (최적화된 버전)
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      // 입력 필드 포커스
      if (inputRef.current) {
        setTimeout(() => {
          inputRef.current?.focus();
        }, 100);
      }

      // 초기화 함수 (상태 확인은 별도 useEffect에서 처리)
      const initializeChatbot = async () => {
        try {

          // 캐시 정보 업데이트
          setCacheInfo(getChatCacheInfo());

          // 환영 메시지 설정 (상태는 실시간으로 업데이트됨)
          const currentHour = new Date().getHours();
          let timeGreeting = "안녕하세요!";
          
          if (currentHour >= 5 && currentHour < 12) {
            timeGreeting = "좋은 아침입니다! 🌅";
          } else if (currentHour >= 12 && currentHour < 18) {
            timeGreeting = "안녕하세요! 😊";
          } else if (currentHour >= 18 && currentHour < 22) {
            timeGreeting = "좋은 저녁입니다! 🌆";
          } else {
            timeGreeting = "좋은 밤 되세요! 🌙";
          }
          
          const welcomeMessage: Message = {
            id: "1",
            content: `${timeGreeting} 정수장 운영 AI 어시스턴트입니다! 🤖\n\n저는 PDF 문서를 기반으로 한 지능형 AI로, 정수장 운영, 설비 관리 및 시스템 사용법에 대해 도움을 드릴 수 있습니다.\n\n💾 캐시 기능이 활성화되어 있어 동일한 질문에 대해 빠른 답변을 제공합니다.\n\n💧 정수처리 공정, 📊 설비 모니터링, 🔧 기술 진단, 📈 탄소중립 관리 등 무엇이든 물어보세요!\n\n🔄 AI 서비스 상태는 실시간으로 모니터링됩니다.`,
            sender: "bot",
            timestamp: new Date(),
          };
          setMessages([welcomeMessage]);
        } catch (error) {
          console.error("챗봇 초기화 중 오류:", error);
          // 오류 시에도 기본 환영 메시지 표시
          const fallbackMessage: Message = {
            id: "1",
            content: "안녕하세요! 정수장 운영 AI 어시스턴트입니다! 💧\n\n서비스에 연결 중입니다. 잠시만 기다려주세요.",
            sender: "bot",
            timestamp: new Date(),
          };
          setMessages([fallbackMessage]);
        }
      };

      initializeChatbot();
    }
  }, [isOpen]); // messages.length 의존성 제거

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

    // 프론트엔드 질문 로그
    console.log("[FRONTEND-QUESTION] Sending question:", {
      question: userMessage.content,
      timestamp: new Date().toISOString(),
      useCache,
    });

    try {
      // AI 기반 챗봇 API 호출 (캐시 사용)
      const response = await sendAIChatMessage(
        userMessage.content,
        "default_pdf",
        true, // 대화 컨텍스트 사용
        useCache // 캐시 사용 여부
      );

      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: response.answer,
        sender: "bot",
        timestamp: new Date(),
        metadata: {
          confidence_score: response.confidence_score,
          question_type: response.question_type,
          generation_time: response.generation_time,
          model_name: response.model_name,
          from_cache: response.from_cache,
        },
      };

      setMessages((prev) => [...prev, botResponse]);

      // 프론트엔드 답변 로그
      console.log("[FRONTEND-ANSWER] Received answer:", {
        question: userMessage.content,
        answer: response.answer.substring(0, 200) + (response.answer.length > 200 ? "..." : ""),
        confidence_score: response.confidence_score,
        generation_time: response.generation_time,
        from_cache: response.from_cache,
        timestamp: new Date().toISOString(),
      });

      // 캐시 정보 업데이트
      setCacheInfo(getChatCacheInfo());
    } catch (error) {
      console.error("[FRONTEND-ERROR] AI 응답 생성 중 오류:", error);
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        content:
          error instanceof Error
            ? error.message
            : "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
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
    const removedCount = clearChatCache("default_pdf");
    setCacheInfo(getChatCacheInfo());

    // 사용자에게 알림
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
    if (isCheckingStatus) return "상태 확인 중...";
    if (aiStatus.ai_available && aiStatus.model_loaded) return "AI 모델 준비됨";
    if (aiStatus.ai_available) return "AI 서비스 연결됨";
    return "AI 서비스 연결 안됨";
  };

  const getStatusDotColor = () => {
    if (isCheckingStatus) return "bg-blue-500 animate-pulse";
    if (aiStatus.ai_available && aiStatus.model_loaded) return "bg-green-500";
    if (aiStatus.ai_available) return "bg-yellow-500 animate-pulse";
    return "bg-red-500";
  };

  if (!isOpen) return null;

  return (
    <div
      className={`
      fixed bottom-32 right-20 w-[480px] h-[600px] bg-white rounded-lg shadow-2xl
      border border-gray-200 z-[999] flex flex-col
      transform transition-all duration-300 ease-in-out
      md:bottom-28 md:right-[88px] md:w-[480px]
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
            {lastStatusChange && (
              <div className="flex justify-between">
                <span>마지막 업데이트:</span>
                <span className="text-xs text-gray-500">
                  {lastStatusChange.toLocaleTimeString()}
                </span>
              </div>
            )}
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
                <div className="flex justify-between">
                  <span>최신 캐시:</span>
                  <span className="text-xs">{cacheInfo.newestEntry}</span>
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
                  title="현재 PDF 캐시 정리"
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
                      {message.metadata.model_name && (
                        <span>모델: {message.metadata.model_name}</span>
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
                    {useCache
                      ? "AI가 생각 중... (캐시 확인 중)"
                      : "AI가 생각 중..."}
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
          <div className="flex items-center space-x-2">
            {!aiStatus.model_loaded && (
              <div className="flex items-center space-x-1 text-xs text-yellow-600">
                <AlertCircle size={12} />
                <span>기본 모드</span>
              </div>
            )}
            {useCache && (
              <div className="flex items-center space-x-1 text-xs text-yellow-600">
                <Zap size={12} />
                <span>캐시 활성화</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
