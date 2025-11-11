import axios from "axios";
import { debugLog } from "../shared/utils/debugUtils";

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

// AI 기반 고급 챗봇 메시지 전송 함수
export const sendAIChatMessage = async (
  message: string,
  pdfId?: string,
  useContext: boolean = true
): Promise<{
  answer: string;
  confidence_score: number;
  question_type: string;
  generation_time: number;
  model_name: string;
}> => {
  const targetPdfId = pdfId || "default_pdf";

  try {
    // AI 기반 질문 답변 API 호출
    const response = await chatApi.post("/ask", {
      question: message,
      pdf_id: targetPdfId,
      use_conversation_context: useContext,
      max_chunks: 5,
      use_dual_pipeline: true,
    });

    if (response.data) {
      return {
        answer: response.data.answer,
        confidence_score: response.data.confidence_score,
        question_type: response.data.question_type,
        generation_time: response.data.generation_time,
        model_name: response.data.model_name,
      };
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
}> => {
  try {
    const response = await chatApi.post("/chat", {
      message: message,
    });

    if (response.data && response.data.success && response.data.response) {
      return {
        answer: response.data.response,
        confidence_score: 0.8, // 간단한 챗봇은 기본 신뢰도
        question_type: "simple_chat",
        generation_time: 0.1,
        model_name: "keyword_based",
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
    return {
      ai_available: response.data.ai_available,
      model_loaded: response.data.model_loaded,
      total_pdfs: response.data.total_pdfs,
      total_chunks: response.data.total_chunks,
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

// 사용 가능한 PDF 목록 조회
export const getAvailablePDFs = async (): Promise<
  Array<{
    pdf_id: string;
    filename: string;
    upload_time: string;
    total_pages: number;
    total_chunks: number;
  }>
> => {
  try {
    const response = await chatApi.get("/pdfs");
    return response.data.pdfs || [];
  } catch (error) {
    console.error("PDF 목록 조회 실패:", error);
    return [];
  }
};

// 간단한 테스트 함수 (백엔드 프록시를 통해)
export const testChatConnection = async (): Promise<void> => {
  try {
    debugLog("Testing chat connection through backend proxy...");
    debugLog("Base URL:", chatApi.defaults.baseURL);

    const response = await chatApi.get("/health");
    debugLog("Health check response:", response.data);

    // AI 서비스 상태도 확인
    const aiStatus = await checkAIServiceStatus();
    debugLog("AI Service status:", aiStatus);
  } catch (error) {
    console.error("Connection test failed:", error);
  }
};
