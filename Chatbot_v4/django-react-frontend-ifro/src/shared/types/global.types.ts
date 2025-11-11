// 챗봇 관련 타입들
export type Message = {
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
};