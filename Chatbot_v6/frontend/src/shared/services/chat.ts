import axios from "axios";

// 환경변수에서 백엔드 API URL 가져오기
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

console.log("Backend API URL:", API_BASE_URL);

// API 인스턴스 생성
const chatApi = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 120초 타임아웃
  headers: {
    "Content-Type": "application/json",
  },
});

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

// AI 기반 챗봇 메시지 전송 함수
export const sendAIChatMessage = async (
  message: string,
  useCache: boolean = true
): Promise<{
  answer: string;
  confidence: number;
  processing_time: number;
}> => {
  try {
    const response = await chatApi.post("/ask", {
      question: message,
      top_k: 50,
    });

    if (response.data) {
      const data = response.data as any;
      return {
        answer: data.answer,
        confidence: data.confidence,
        processing_time: data.metrics?.processing_time || 0,
      };
    } else {
      throw new Error("AI 응답이 올바르지 않습니다.");
    }
  } catch (error: any) {
    console.error("AI Chat message error:", error);

    if (error.response?.status === 500) {
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

// 캐시 관련 유틸리티 함수들 (로컬 스토리지 기반)
const CACHE_KEY = "chatbot_cache";

export const clearChatCache = (): number => {
  try {
    const cache = localStorage.getItem(CACHE_KEY);
    if (cache) {
      const parsed = JSON.parse(cache);
      const count = Object.keys(parsed).length;
      localStorage.removeItem(CACHE_KEY);
      return count;
    }
  } catch (error) {
    console.error("Cache clear error:", error);
  }
  return 0;
};

export const getChatCacheInfo = () => {
  try {
    const cache = localStorage.getItem(CACHE_KEY);
    if (cache) {
      const parsed = JSON.parse(cache);
      const entries = Object.keys(parsed).length;
      const sizeInKB = Math.round(new Blob([cache]).size / 1024);

      return {
        totalEntries: entries,
        sizeInKB: sizeInKB,
        hitRate: 0, // 실제 구현 시 계산 필요
      };
    }
  } catch (error) {
    console.error("Cache info error:", error);
  }

  return {
    totalEntries: 0,
    sizeInKB: 0,
    hitRate: 0,
  };
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

// 헬스 체크
export const checkChatServerStatus = async (): Promise<boolean> => {
  try {
    await chatApi.get("/healthz", { timeout: 10000 });
    return true;
  } catch (error) {
    return false;
  }
};

