// 기본 타입들
export type Coordinates = {
  lat: number;
  lng: number;
};

// 챗봇 관련 타입들
export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export interface ChatSession {
  id: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

// API 응답 타입들
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// 사용자 관련 타입들
export interface User {
  id: number;
  username: string;
  email: string;
  createdAt: string;
}

// 설정 관련 타입들
export interface AppSettings {
  theme: 'light' | 'dark';
  language: 'ko' | 'en';
  notifications: boolean;
}