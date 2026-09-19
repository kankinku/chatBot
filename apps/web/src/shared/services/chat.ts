import axios from "axios";
import { debugLog } from "../utils/debugUtils";
import { chatCache } from "../utils/chatCache";

// 환경변수에서 백엔드 API URL 가져오기 (기본값: localhost:8000)
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// 챗봇 API는 이제 백엔드를 통해 프록시됨
const CHATBOT_PROXY_URL = `${API_BASE_URL}/api/chatbot`;

// 디버깅을 위한 로그
debugLog("Backend API URL:", API_BASE_URL);
debugLog("ChatBot Proxy URL:", CHATBOT_PROXY_URL);
debugLog("Environment variable API_URL:", process.env.REACT_APP_API_URL);

// 챗봇 API 인스턴스 생성 (이제 백엔드 프록시를 통해 통신)
const chatApi = axios.create({
  baseURL: CHATBOT_PROXY_URL,
  timeout: 120000, // 120초 타임아웃 (백엔드 프록시 + AI 처리 시간 고려)
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: false, // CORS는 백엔드에서 처리
});

// 요청 인터셉터 - 토큰 추가
chatApi.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("access");
    if (token && config.headers) {
      config.headers["Authorization"] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 응답 인터셉터 - 에러 처리
chatApi.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error("Chat API Error:", error);
    return Promise.reject(error);
  }
);

// AI 기반 고급 챗봇 메시지 전송 함수 (캐시 통합)
export const sendAIChatMessage = async (
  message: string,
  pdfId?: string,
  useContext: boolean = true,
  useCache: boolean = true
): Promise<{
  answer: string;
  confidence_score: number;
  question_type: string;
  generation_time: number;
  model_name: string;
  from_cache: boolean;
}> => {
  const targetPdfId = pdfId || "default_pdf";

  // 인사말인지 확인
  const isGreeting = chatCache.isGreetingQuestion(message);
  
  // 인사말이거나 캐시 사용이 비활성화된 경우 캐시를 사용하지 않음
  const shouldUseCache = useCache && !isGreeting;

  // 캐시 사용이 활성화된 경우 캐시에서 먼저 검색
  if (shouldUseCache) {
    const cachedResponse = chatCache.findCachedAnswer(message, targetPdfId);
    if (cachedResponse) {
      debugLog(`캐시에서 답변 찾음: "${message}"`);
      return {
        answer: cachedResponse.answer,
        confidence_score: cachedResponse.confidence_score,
        question_type: cachedResponse.question_type,
        generation_time: cachedResponse.generation_time,
        model_name: cachedResponse.model_name,
        from_cache: true,
      };
    }
  }

  try {
    // AI 기반 질문 답변 API 호출
    const response = await chatApi.post("/ask", {
      question: message,
      top_k: 50,
    });

    if (response.data) {
      const data = response.data as any;
      
      // 백엔드 응답 형식에 맞게 매핑
      const metrics = data.metrics || {};
      const processingTime = metrics.total_time_ms ? metrics.total_time_ms / 1000 : 0;
      
      const result = {
        answer: data.answer || '답변을 생성할 수 없습니다.',
        confidence_score: data.confidence || 0.0,
        question_type: 'ai_generated',
        generation_time: processingTime,
        model_name: 'qwen2.5:3b-instruct-q4_K_M',
        from_cache: false,
      };

      // 성공적인 응답을 캐시에 저장 (인사말은 제외)
      if (shouldUseCache) {
        chatCache.cacheAnswer(
          message,
          result.answer,
          result.confidence_score,
          result.question_type,
          result.generation_time,
          result.model_name,
          targetPdfId
        );
      }

      return result;
    } else {
      throw new Error("AI 응답이 올바르지 않습니다.");
    }
  } catch (error: any) {
    console.error("AI Chat message error:", error);

    // AI 서비스 실패 시 폴백으로 간단한 챗봇 사용
    if (error.response?.status === 404) {
      debugLog("AI 서비스 실패, 간단한 챗봇으로 폴백");
      return await sendSimpleChatMessage(message);
    } else if (error.response?.status === 500) {
      throw new Error(
        "AI 서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
      );
    } else if (error.code === "ECONNABORTED") {
      throw new Error(
        "요청 시간이 초과되었습니다. 네트워크 상태를 확인해주세요."
      );
    } else if (error.message.includes("Network Error")) {
      throw new Error(
        "AI 서버에 연결할 수 없습니다. 서버 상태를 확인해주세요."
      );
    }

    throw new Error(
      `AI 응답 오류: ${error.response?.data?.detail || error.message}`
    );
  }
};

// 기존 간단한 챗봇 메시지 전송 함수 (폴백용)
export const sendSimpleChatMessage = async (
  message: string
): Promise<{
  answer: string;
  confidence_score: number;
  question_type: string;
  generation_time: number;
  model_name: string;
  from_cache: boolean;
}> => {
  try {
    const response = await chatApi.post("/chat", {
      message: message,
    });

    const data = response.data as any;
    if (data && data.success && data.response) {
      return {
        answer: data.response,
        confidence_score: 0.8, // 간단한 챗봇은 기본 신뢰도
        question_type: "simple_chat",
        generation_time: 0.1,
        model_name: "keyword_based",
        from_cache: false,
      };
    } else {
      throw new Error("챗봇 응답이 올바르지 않습니다.");
    }
  } catch (error: any) {
    console.error("Simple chat message error:", error);
    throw new Error(
      `챗봇 응답 오류: ${error.response?.data?.message || error.message}`
    );
  }
};

// 기존 함수명 유지 (하위 호환성)
export const sendChatMessage = async (message: string): Promise<string> => {
  try {
    const result = await sendAIChatMessage(message);
    return result.answer;
  } catch (error: any) {
    throw new Error(error.message);
  }
};

// 캐시 관련 유틸리티 함수들
export const clearChatCache = (pdfId?: string): number => {
  if (pdfId) {
    return chatCache.clearPdfCache(pdfId);
  } else {
    chatCache.clearAllCache();
    return 0;
  }
};

export const getChatCacheStats = () => {
  return chatCache.getCacheStats();
};

export const getChatCacheInfo = () => {
  return chatCache.getCacheInfo();
};

export const findSimilarCachedQuestions = (
  question: string,
  limit: number = 5
) => {
  return chatCache.findSimilarQuestions(question, limit);
};

// 챗봇 연결 상태 확인 함수 (백엔드 프록시를 통해)
export const checkChatServerStatus = async (): Promise<boolean> => {
  try {
    await chatApi.get("/health", { timeout: 10000 });
    return true;
  } catch (error) {
    return false;
  }
};

// AI 서비스 상태 확인 함수
export const checkAIServiceStatus = async (): Promise<{
  ai_available: boolean;
  model_loaded: boolean;
  total_pdfs: number;
  total_chunks: number;
}> => {
  try {
    const response = await chatApi.get("/status");
    const data = response.data as any;
    return {
      ai_available: data.ai_available,
      model_loaded: data.model_loaded,
      total_pdfs: data.total_pdfs,
      total_chunks: data.total_chunks,
    };
  } catch (error) {
    console.log("챗봇 서버 상태 확인 실패:", error);
    return {
      ai_available: false,
      model_loaded: false,
      total_pdfs: 0,
      total_chunks: 0,
    };
  }
};

// 프록시 API 인스턴스 (타임아웃 없음)
const proxyApi = axios.create({
  baseURL: API_BASE_URL + '/api',
  timeout: 0, // 타임아웃 제거
  headers: {
    "Content-Type": "application/json",
  },
});

// PDF 처리 시작 함수
export const startProcessPDFs = async (): Promise<void> => {
  try {
    await proxyApi.post("/process-pdfs");
  } catch (error: any) {
    console.error("PDF 처리 시작 오류:", error);
    throw new Error(
      error.response?.data?.detail || error.message || "PDF 처리 시작 중 오류가 발생했습니다."
    );
  }
};

// PDF 처리 상태 확인 함수
export const getProcessPDFsStatus = async (): Promise<{
  status: string;
  progress: number;
  message: string;
  processed_files: string[];
  skipped_files: string[];
  total_chunks: number;
  processing_time_seconds: number;
}> => {
  try {
    const response = await proxyApi.get("/process-pdfs/status");
    return response.data as {
      status: string;
      progress: number;
      message: string;
      processed_files: string[];
      skipped_files: string[];
      total_chunks: number;
      processing_time_seconds: number;
    };
  } catch (error: any) {
    console.error("PDF 처리 상태 확인 오류:", error);
    throw new Error(
      error.response?.data?.detail || error.message || "PDF 처리 상태 확인 중 오류가 발생했습니다."
    );
  }
};

// PDF 처리 함수 (비동기 + 폴링)
export const processPDFs = async (
  onProgress?: (progress: number, message: string) => void
): Promise<{
  success: boolean;
  message: string;
  processed_files: string[];
  skipped_files: string[];
  total_chunks: number;
  processing_time_seconds: number;
}> => {
  // 처리 시작
  await startProcessPDFs();
  
  // 상태 확인 폴링 (3초마다)
  return new Promise((resolve, reject) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await getProcessPDFsStatus();
        
        // 진행률 콜백 호출
        if (onProgress) {
          onProgress(status.progress, status.message);
        }
        
        if (status.status === "completed") {
          clearInterval(pollInterval);
          resolve({
            success: true,
            message: status.message,
            processed_files: status.processed_files || [],
            skipped_files: status.skipped_files || [],
            total_chunks: status.total_chunks,
            processing_time_seconds: status.processing_time_seconds,
          });
        } else if (status.status === "error") {
          clearInterval(pollInterval);
          reject(new Error(status.message || "PDF 처리 중 오류가 발생했습니다."));
        }
      } catch (error: any) {
        clearInterval(pollInterval);
        reject(error);
      }
    }, 3000); // 3초마다 상태 확인
    
    // 최대 대기 시간 (1시간)
    setTimeout(() => {
      clearInterval(pollInterval);
      reject(new Error("PDF 처리 시간이 초과되었습니다."));
    }, 3600000);
  });
};


