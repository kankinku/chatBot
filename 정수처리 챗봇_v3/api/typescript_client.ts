/**
 * React 프론트엔드에서 PDF QA API를 호출하기 위한 TypeScript 클라이언트
 * 
 * 이 모듈은 React 애플리케이션에서 PDF QA 시스템의 API를 쉽게 호출할 수 있도록
 * 하는 클라이언트 클래스를 제공합니다.
 */

// API 응답 타입 정의
export interface QuestionResponse {
  success: boolean;
  answer: string;
  confidence_score: number;
  question_type: string;
  intent?: string;
  keywords?: string[];
  used_chunks?: string[];
  processing_time: number;
  model_name: string;
  error?: string;
}

export interface PDFUploadResponse {
  pdf_id: string;
  filename: string;
  total_pages: number;
  total_chunks: number;
  processing_time: number;
}

export interface SystemStatusResponse {
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

export interface ConversationHistoryItem {
  question: string;
  answer: string;
  timestamp: string;
  confidence_score: number;
}

export interface PDFInfo {
  pdf_id: string;
  filename: string;
  upload_time: string;
  total_pages: number;
  total_chunks: number;
}

export interface ConversationHistoryResponse {
  conversation_history: ConversationHistoryItem[];
}

export interface PDFListResponse {
  pdfs: PDFInfo[];
}

export interface HealthCheckResponse {
  status: string;
  timestamp: string;
}

export interface ErrorResponse {
  success: false;
  error: string;
  detail?: string;
}

// API 요청 타입 정의
export interface AskQuestionRequest {
  question: string;
  pdf_id?: string;
  conversation_history?: ConversationHistoryItem[];
}

export interface UploadPDFRequest {
  file: File;
}

/**
 * PDF QA API 클라이언트
 * 
 * React 프론트엔드에서 PDF QA 시스템과 통신하기 위한 클라이언트입니다.
 */
export class PDFQAClient {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = "http://localhost:8000", timeout: number = 30000) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeout = timeout;
  }

  /**
   * HTTP 요청 수행
   */
  private async makeRequest<T>(
    method: string,
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...options.headers,
      },
      timeout: this.timeout,
      ...options,
    };

    try {
      const response = await fetch(url, defaultOptions);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API 요청 실패: ${method} ${url}`, error);
      throw error;
    }
  }

  /**
   * 질문에 대한 답변 요청
   */
  async askQuestion(request: AskQuestionRequest): Promise<QuestionResponse> {
    const data = {
      question: request.question,
      pdf_id: request.pdf_id || "default",
      conversation_history: request.conversation_history || []
    };

    return this.makeRequest<QuestionResponse>("POST", "/django/ask", {
      body: JSON.stringify(data)
    });
  }

  /**
   * PDF 파일 업로드
   */
  async uploadPDF(file: File): Promise<PDFUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.makeRequest<PDFUploadResponse>("POST", "/upload_pdf", {
      headers: {}, // FormData를 사용하므로 Content-Type 헤더 제거
      body: formData
    });
  }

  /**
   * 시스템 상태 조회
   */
  async getSystemStatus(): Promise<SystemStatusResponse> {
    return this.makeRequest<SystemStatusResponse>("GET", "/status");
  }

  /**
   * 대화 기록 조회
   */
  async getConversationHistory(pdfId: string, maxItems: number = 10): Promise<ConversationHistoryResponse> {
    const params = new URLSearchParams({
      pdf_id: pdfId,
      max_items: maxItems.toString()
    });

    return this.makeRequest<ConversationHistoryResponse>(`GET", "/conversation_history?${params}`);
  }

  /**
   * 대화 기록 초기화
   */
  async clearConversationHistory(): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>("DELETE", "/conversation_history");
  }

  /**
   * 등록된 PDF 목록 조회
   */
  async listPDFs(): Promise<PDFListResponse> {
    return this.makeRequest<PDFListResponse>("GET", "/pdfs");
  }

  /**
   * PDF 삭제
   */
  async deletePDF(pdfId: string): Promise<{ message: string }> {
    return this.makeRequest<{ message: string }>("DELETE", `/pdfs/${pdfId}`);
  }

  /**
   * 서버 헬스 체크
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    return this.makeRequest<HealthCheckResponse>("GET", "/health");
  }


}

// React Hook을 위한 커스텀 훅들

import { useState, useCallback } from 'react';

/**
 * PDF QA 질문-답변을 위한 React Hook
 */
export function usePDFQA(baseUrl?: string) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<QuestionResponse | null>(null);

  const client = new PDFQAClient(baseUrl);

  const askQuestion = useCallback(async (question: string, pdfId?: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await client.askQuestion({ question, pdf_id: pdfId });
      setLastResponse(response);
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '알 수 없는 오류가 발생했습니다.';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [client]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    askQuestion,
    loading,
    error,
    lastResponse,
    clearError
  };
}

/**
 * PDF 업로드를 위한 React Hook
 */
export function usePDFUpload(baseUrl?: string) {
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [lastUpload, setLastUpload] = useState<PDFUploadResponse | null>(null);

  const client = new PDFQAClient(baseUrl);

  const uploadPDF = useCallback(async (file: File) => {
    setUploading(true);
    setUploadError(null);

    try {
      const response = await client.uploadPDF(file);
      setLastUpload(response);
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '파일 업로드 중 오류가 발생했습니다.';
      setUploadError(errorMessage);
      throw err;
    } finally {
      setUploading(false);
    }
  }, [client]);

  const clearUploadError = useCallback(() => {
    setUploadError(null);
  }, []);

  return {
    uploadPDF,
    uploading,
    uploadError,
    lastUpload,
    clearUploadError
  };
}

/**
 * 시스템 상태를 위한 React Hook
 */
export function useSystemStatus(baseUrl?: string) {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const client = new PDFQAClient(baseUrl);

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await client.getSystemStatus();
      setStatus(response);
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '시스템 상태 조회 중 오류가 발생했습니다.';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [client]);

  return {
    status,
    loading,
    error,
    fetchStatus
  };
}

// React 컴포넌트 예시

/**
 * PDF QA 채팅 컴포넌트 예시
 */
export const PDFQAChat: React.FC<{ baseUrl?: string }> = ({ baseUrl }) => {
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState<Array<{ question: string; answer: string }>>([]);
  const { askQuestion, loading, error, lastResponse } = usePDFQA(baseUrl);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    try {
      const response = await askQuestion(question);
      setConversation(prev => [...prev, { question, answer: response.answer }]);
      setQuestion('');
    } catch (err) {
      console.error('질문 처리 실패:', err);
    }
  };

  return (
    <div className="pdf-qa-chat">
      <div className="conversation">
        {conversation.map((item, index) => (
          <div key={index} className="message">
            <div className="question">
              <strong>질문:</strong> {item.question}
            </div>
            <div className="answer">
              <strong>답변:</strong> {item.answer}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="question-form">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="질문을 입력하세요..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? '처리 중...' : '질문하기'}
        </button>
      </form>

      {error && (
        <div className="error">
          오류: {error}
        </div>
      )}
    </div>
  );
};

/**
 * PDF 업로드 컴포넌트 예시
 */
export const PDFUpload: React.FC<{ baseUrl?: string; onUpload?: (response: PDFUploadResponse) => void }> = ({ 
  baseUrl, 
  onUpload 
}) => {
  const { uploadPDF, uploading, uploadError } = usePDFUpload(baseUrl);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const response = await uploadPDF(file);
      onUpload?.(response);
    } catch (err) {
      console.error('PDF 업로드 실패:', err);
    }
  };

  return (
    <div className="pdf-upload">
      <input
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        disabled={uploading}
      />
      {uploading && <div>업로드 중...</div>}
      {uploadError && <div className="error">오류: {uploadError}</div>}
    </div>
  );
};

// 사용 예시:
/*
import React from 'react';
import { PDFQAClient, usePDFQA, PDFQAChat, PDFUpload } from './api/typescript_client';

// 기본 사용법
const client = new PDFQAClient('http://localhost:8000');

// React 컴포넌트에서 사용
function App() {
  return (
    <div>
      <h1>PDF QA 시스템</h1>
      <PDFUpload baseUrl="http://localhost:8000" />
      <PDFQAChat baseUrl="http://localhost:8000" />
    </div>
  );
}

// 커스텀 훅 사용
function CustomChat() {
  const { askQuestion, loading, error } = usePDFQA('http://localhost:8000');
  
  const handleAsk = async () => {
    try {
      const response = await askQuestion('질문 내용');
      console.log('답변:', response.answer);
    } catch (err) {
      console.error('오류:', err);
    }
  };

  return (
    <div>
      <button onClick={handleAsk} disabled={loading}>
        질문하기
      </button>
      {error && <div>오류: {error}</div>}
    </div>
  );
}
*/
