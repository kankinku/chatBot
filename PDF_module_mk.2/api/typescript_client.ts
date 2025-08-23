/**
 * TypeScript 클라이언트 라이브러리
 * 
 * React/Vue/Angular 등 TypeScript 기반 프론트엔드에서
 * PDF QA API를 쉽게 사용할 수 있도록 하는 클라이언트 라이브러리입니다.
 */

// API 타입 정의
export interface QuestionRequest {
  question: string;
  pdf_id: string;
  use_conversation_context?: boolean;
  max_chunks?: number;
}

export interface QuestionResponse {
  answer: string;
  confidence_score: number;
  used_chunks: string[];
  generation_time: number;
  question_type: string;
  model_name: string;
}

export interface PDFUploadResponse {
  pdf_id: string;
  filename: string;
  total_pages: number;
  total_chunks: number;
  processing_time: number;
}

export interface ConversationHistoryItem {
  question: string;
  answer: string;
  timestamp: string;
  confidence_score: number;
}

export interface SystemStatus {
  status: string;
  model_loaded: boolean;
  total_pdfs: number;
  total_chunks: number;
  memory_usage: {
    rss_mb: number;
    vms_mb: number;
    cpu_percent: number;
  };
}

export interface PDFInfo {
  pdf_id: string;
  filename: string;
  upload_time: string;
  total_pages: number;
  total_chunks: number;
}

export interface EvaluationRequest {
  questions: string[];
  generated_answers: string[];
  reference_answers: string[];
}

export interface ModelConfig {
  model_type: 'ollama' | 'huggingface' | 'llama_cpp';
  model_name: string;
  max_length?: number;
  temperature?: number;
  top_p?: number;
}

// API 클라이언트 설정
export interface PDFQAClientConfig {
  baseURL: string;
  timeout?: number;
  headers?: Record<string, string>;
}

// 에러 타입
export class PDFQAError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: any
  ) {
    super(message);
    this.name = 'PDFQAError';
  }
}

/**
 * PDF QA API 클라이언트
 * 
 * 사용 예시:
 * ```typescript
 * const client = new PDFQAClient({ baseURL: 'http://localhost:8000' });
 * 
 * // PDF 업로드
 * const uploadResult = await client.uploadPDF(file);
 * 
 * // 질문하기
 * const answer = await client.askQuestion({
 *   question: '이 문서의 주요 내용은 무엇인가요?',
 *   pdf_id: uploadResult.pdf_id
 * });
 * ```
 */
export class PDFQAClient {
  private baseURL: string;
  private timeout: number;
  private headers: Record<string, string>;

  constructor(config: PDFQAClientConfig) {
    this.baseURL = config.baseURL.replace(/\/$/, ''); // 마지막 슬래시 제거
    this.timeout = config.timeout || 30000; // 30초 기본값
    this.headers = {
      'Content-Type': 'application/json',
      ...config.headers
    };
  }

  /**
   * HTTP 요청 헬퍼 메서드
   */
  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      timeout: this.timeout,
      headers: this.headers,
      ...options
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new PDFQAError(
          errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          errorData
        );
      }

      return await response.json();
    } catch (error) {
      if (error instanceof PDFQAError) {
        throw error;
      }
      
      throw new PDFQAError(`네트워크 오류: ${error.message}`);
    }
  }

  /**
   * 시스템 상태 조회
   */
  async getSystemStatus(): Promise<SystemStatus> {
    return await this.request<SystemStatus>('/status');
  }

  /**
   * 헬스 체크
   */
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return await this.request('/health');
  }

  /**
   * PDF 파일 업로드
   * 
   * @param file PDF 파일 (File 객체)
   * @returns 업로드 결과
   */
  async uploadPDF(file: File): Promise<PDFUploadResponse> {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      throw new PDFQAError('PDF 파일만 업로드할 수 있습니다.');
    }

    const formData = new FormData();
    formData.append('file', file);

    // FormData 사용 시 Content-Type 헤더 제거 (브라우저가 자동 설정)
    const headers = { ...this.headers };
    delete headers['Content-Type'];

    return await this.request<PDFUploadResponse>('/upload_pdf', {
      method: 'POST',
      headers,
      body: formData
    });
  }

  /**
   * 질문하기
   * 
   * @param request 질문 요청
   * @returns 답변 결과
   */
  async askQuestion(request: QuestionRequest): Promise<QuestionResponse> {
    return await this.request<QuestionResponse>('/ask', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }

  /**
   * 대화 기록 조회
   * 
   * @param pdfId PDF ID
   * @param maxItems 최대 항목 수
   * @returns 대화 기록
   */
  async getConversationHistory(
    pdfId: string, 
    maxItems: number = 10
  ): Promise<{ conversation_history: ConversationHistoryItem[] }> {
    const params = new URLSearchParams({
      pdf_id: pdfId,
      max_items: maxItems.toString()
    });

    return await this.request(`/conversation_history?${params}`);
  }

  /**
   * 대화 기록 초기화
   */
  async clearConversationHistory(): Promise<{ message: string }> {
    return await this.request('/conversation_history', {
      method: 'DELETE'
    });
  }

  /**
   * 등록된 PDF 목록 조회
   */
  async listPDFs(): Promise<{ pdfs: PDFInfo[] }> {
    return await this.request('/pdfs');
  }

  /**
   * PDF 삭제
   * 
   * @param pdfId PDF ID
   */
  async deletePDF(pdfId: string): Promise<{ message: string }> {
    return await this.request(`/pdfs/${pdfId}`, {
      method: 'DELETE'
    });
  }

  /**
   * 모델 설정 변경
   * 
   * @param config 모델 설정
   */
  async configureModel(config: ModelConfig): Promise<{ message: string }> {
    return await this.request('/configure_model', {
      method: 'POST',
      body: JSON.stringify(config)
    });
  }

  /**
   * 시스템 성능 평가
   * 
   * @param request 평가 요청
   */
  async evaluateSystem(request: EvaluationRequest): Promise<any> {
    return await this.request('/evaluate', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }
}

/**
 * React Hook: PDF QA 클라이언트 사용
 * 
 * React 애플리케이션에서 PDF QA 기능을 쉽게 사용할 수 있는 커스텀 훅입니다.
 * 
 * 사용 예시:
 * ```typescript
 * function MyComponent() {
 *   const { client, uploadPDF, askQuestion, loading, error } = usePDFQA({
 *     baseURL: 'http://localhost:8000'
 *   });
 * 
 *   const handleFileUpload = async (file: File) => {
 *     const result = await uploadPDF(file);
 *     console.log('업로드 완료:', result);
 *   };
 * 
 *   const handleAskQuestion = async (question: string, pdfId: string) => {
 *     const answer = await askQuestion({ question, pdf_id: pdfId });
 *     console.log('답변:', answer);
 *   };
 * 
 *   return (
 *     <div>
 *       {loading && <div>처리 중...</div>}
 *       {error && <div>오류: {error.message}</div>}
 *       {/* UI 컴포넌트들 */}
 *     </div>
 *   );
 * }
 * ```
 */
export function usePDFQA(config: PDFQAClientConfig) {
  // React hooks는 실제 React 환경에서만 사용 가능
  // 여기서는 타입 정의만 제공
  
  const client = new PDFQAClient(config);
  
  return {
    client,
    
    // 래퍼 함수들
    uploadPDF: async (file: File) => {
      try {
        return await client.uploadPDF(file);
      } catch (error) {
        console.error('PDF 업로드 실패:', error);
        throw error;
      }
    },
    
    askQuestion: async (request: QuestionRequest) => {
      try {
        return await client.askQuestion(request);
      } catch (error) {
        console.error('질문 처리 실패:', error);
        throw error;
      }
    },
    
    getSystemStatus: async () => {
      try {
        return await client.getSystemStatus();
      } catch (error) {
        console.error('시스템 상태 조회 실패:', error);
        throw error;
      }
    }
  };
}

/**
 * Vue Composition API: PDF QA 클라이언트 사용
 * 
 * Vue 3 Composition API에서 사용할 수 있는 컴포저블입니다.
 * 
 * 사용 예시:
 * ```typescript
 * import { usePDFQAComposable } from './typescript_client';
 * 
 * export default defineComponent({
 *   setup() {
 *     const { client, uploadPDF, askQuestion, loading, error } = usePDFQAComposable({
 *       baseURL: 'http://localhost:8000'
 *     });
 * 
 *     return {
 *       uploadPDF,
 *       askQuestion,
 *       loading,
 *       error
 *     };
 *   }
 * });
 * ```
 */
export function usePDFQAComposable(config: PDFQAClientConfig) {
  const client = new PDFQAClient(config);
  
  // Vue의 ref, reactive 등은 실제 Vue 환경에서만 사용 가능
  // 여기서는 기본 구조만 제공
  
  return {
    client,
    
    uploadPDF: async (file: File) => {
      return await client.uploadPDF(file);
    },
    
    askQuestion: async (request: QuestionRequest) => {
      return await client.askQuestion(request);
    },
    
    loading: false, // 실제로는 ref(false)
    error: null     // 실제로는 ref(null)
  };
}

// 유틸리티 함수들

/**
 * 파일 크기를 사람이 읽기 쉬운 형태로 변환
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * 신뢰도 점수를 색상으로 변환
 */
export function getConfidenceColor(score: number): string {
  if (score >= 0.8) return '#4CAF50'; // 녹색
  if (score >= 0.6) return '#FF9800'; // 주황색
  if (score >= 0.4) return '#FFC107'; // 노란색
  return '#F44336'; // 빨간색
}

/**
 * 질문 유형을 한국어로 변환
 */
export function translateQuestionType(type: string): string {
  const translations: Record<string, string> = {
    'factual': '사실형',
    'conceptual': '개념형',
    'comparative': '비교형',
    'procedural': '절차형',
    'analytical': '분석형',
    'follow_up': '후속질문',
    'clarification': '명확화'
  };
  
  return translations[type] || type;
}

/**
 * 시간을 사람이 읽기 쉬운 형태로 변환
 */
export function formatDuration(seconds: number): string {
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}초`;
  
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  
  return `${minutes}분 ${remainingSeconds}초`;
}

// 기본 클라이언트 인스턴스 (개발용)
export const defaultClient = new PDFQAClient({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000'
});

// 타입 가드 함수들
export function isPDFQAError(error: any): error is PDFQAError {
  return error instanceof PDFQAError;
}

export function isValidPDFFile(file: File): boolean {
  return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
}

// 캐시 관리 (선택적 기능)
export class PDFQACache {
  private cache = new Map<string, any>();
  private ttl = 5 * 60 * 1000; // 5분

  set(key: string, value: any): void {
    this.cache.set(key, {
      value,
      timestamp: Date.now()
    });
  }

  get(key: string): any | null {
    const item = this.cache.get(key);
    if (!item) return null;

    if (Date.now() - item.timestamp > this.ttl) {
      this.cache.delete(key);
      return null;
    }

    return item.value;
  }

  clear(): void {
    this.cache.clear();
  }
}

export default PDFQAClient;
