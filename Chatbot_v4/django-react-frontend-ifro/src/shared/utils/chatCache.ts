import { debugLog } from "./debugUtils";

/**
 * 챗봇 질문-답변 캐시 시스템
 *
 * 이 모듈은 사용자의 질문과 AI 답변을 로컬 스토리지에 캐싱하여
 * 동일한 질문에 대해 빠르고 일관된 답변을 제공합니다.
 */

export interface CachedResponse {
  question: string;
  answer: string;
  confidence_score: number;
  question_type: string;
  generation_time: number;
  model_name: string;
  timestamp: number;
  pdf_id: string;
  question_hash: string;
}

export interface CacheStats {
  total_cached: number;
  cache_hits: number;
  hit_rate: number;
}

class ChatCache {
  private readonly CACHE_KEY = "ifro_chat_cache";
  private readonly STATS_KEY = "ifro_chat_cache_stats";
  private readonly MAX_CACHE_SIZE = 100; // 최대 캐시 항목 수
  private readonly CACHE_EXPIRY_DAYS = 7; // 캐시 만료 기간 (일)

  /**
   * 질문을 해시로 변환
   */
  private hashQuestion(question: string): string {
    // 간단한 해시 함수 (실제로는 더 정교한 해시 사용 가능)
    let hash = 0;
    const normalizedQuestion = question.toLowerCase().trim();

    for (let i = 0; i < normalizedQuestion.length; i++) {
      const char = normalizedQuestion.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // 32비트 정수로 변환
    }

    return Math.abs(hash).toString(36);
  }

  /**
   * 유사한 질문인지 확인 (유사도 기반)
   */
  private isSimilarQuestion(question1: string, question2: string): boolean {
    const normalized1 = question1.toLowerCase().trim();
    const normalized2 = question2.toLowerCase().trim();

    // 정확히 일치하는 경우
    if (normalized1 === normalized2) return true;

    // 단어 기반 유사도 계산
    const words1 = new Set(normalized1.split(/\s+/));
    const words2 = new Set(normalized2.split(/\s+/));

    const intersection = new Set([...words1].filter((x) => words2.has(x)));
    const union = new Set([...words1, ...words2]);

    const similarity = intersection.size / union.size;
    return similarity > 0.7; // 70% 이상 유사한 경우
  }

  /**
   * 캐시된 답변 찾기
   */
  findCachedAnswer(
    question: string,
    pdfId: string = "default_pdf"
  ): CachedResponse | null {
    try {
      const cache = this.getCache();
      const questionHash = this.hashQuestion(question);

      // 정확한 해시 매칭 먼저 시도
      const exactMatch = cache.find(
        (entry) =>
          entry.question_hash === questionHash && entry.pdf_id === pdfId
      );

      if (exactMatch) {
        this.updateCacheStats(true);
        return exactMatch;
      }

      // 유사한 질문 검색
      const similarMatch = cache.find(
        (entry) =>
          this.isSimilarQuestion(entry.question, question) &&
          entry.pdf_id === pdfId
      );

      if (similarMatch) {
        this.updateCacheStats(true);
        return similarMatch;
      }

      this.updateCacheStats(false);
      return null;
    } catch (error) {
      console.error("캐시 검색 오류:", error);
      return null;
    }
  }

  /**
   * 답변을 캐시에 저장
   */
  cacheAnswer(
    question: string,
    answer: string,
    confidence_score: number,
    question_type: string,
    generation_time: number,
    model_name: string,
    pdfId: string = "default_pdf"
  ): void {
    try {
      const cache = this.getCache();
      const questionHash = this.hashQuestion(question);

      const cachedResponse: CachedResponse = {
        question,
        answer,
        confidence_score,
        question_type,
        generation_time,
        model_name,
        timestamp: Date.now(),
        pdf_id: pdfId,
        question_hash: questionHash,
      };

      // 기존 동일한 질문이 있으면 업데이트
      const existingIndex = cache.findIndex(
        (entry) =>
          entry.question_hash === questionHash && entry.pdf_id === pdfId
      );

      if (existingIndex !== -1) {
        cache[existingIndex] = cachedResponse;
      } else {
        // 새 항목 추가
        cache.push(cachedResponse);

        // 캐시 크기 제한 확인
        if (cache.length > this.MAX_CACHE_SIZE) {
          // 가장 오래된 항목 제거
          cache.sort((a, b) => a.timestamp - b.timestamp);
          cache.shift();
        }
      }

      // 만료된 항목 제거
      this.cleanExpiredEntries(cache);

      // 캐시 저장
      localStorage.setItem(this.CACHE_KEY, JSON.stringify(cache));

      debugLog(`캐시에 저장됨: "${question}" (총 ${cache.length}개 항목)`);
    } catch (error) {
      console.error("캐시 저장 오류:", error);
    }
  }

  /**
   * 만료된 캐시 항목 제거
   */
  private cleanExpiredEntries(cache: CachedResponse[]): void {
    const expiryTime =
      Date.now() - this.CACHE_EXPIRY_DAYS * 24 * 60 * 60 * 1000;

    const validEntries = cache.filter((entry) => entry.timestamp > expiryTime);

    if (validEntries.length !== cache.length) {
      const removedCount = cache.length - validEntries.length;
      debugLog(`${removedCount}개의 만료된 캐시 항목이 제거되었습니다.`);
      cache.splice(0, cache.length, ...validEntries);
    }
  }

  /**
   * 캐시 통계 업데이트
   */
  private updateCacheStats(hit: boolean): void {
    try {
      const stats = this.getCacheStats();
      stats.total_cached += 1;

      if (hit) {
        stats.cache_hits += 1;
      }

      stats.hit_rate = stats.cache_hits / stats.total_cached;

      localStorage.setItem(this.STATS_KEY, JSON.stringify(stats));
    } catch (error) {
      console.error("캐시 통계 업데이트 오류:", error);
    }
  }

  /**
   * 캐시 통계 가져오기
   */
  getCacheStats(): CacheStats {
    try {
      const stats = localStorage.getItem(this.STATS_KEY);
      if (stats) {
        return JSON.parse(stats);
      }
    } catch (error) {
      console.error("캐시 통계 로드 오류:", error);
    }

    return {
      total_cached: 0,
      cache_hits: 0,
      hit_rate: 0,
    };
  }

  /**
   * 캐시 데이터 가져오기
   */
  private getCache(): CachedResponse[] {
    try {
      const cache = localStorage.getItem(this.CACHE_KEY);
      if (cache) {
        return JSON.parse(cache);
      }
    } catch (error) {
      console.error("캐시 로드 오류:", error);
    }

    return [];
  }

  /**
   * 캐시 상태 정보 가져오기
   */
  getCacheInfo(): {
    totalEntries: number;
    sizeInKB: number;
    oldestEntry: string;
    newestEntry: string;
    hitRate: number;
  } {
    try {
      const cache = this.getCache();
      const stats = this.getCacheStats();

      const cacheSize = new Blob([JSON.stringify(cache)]).size / 1024;

      const oldestEntry =
        cache.length > 0
          ? new Date(
              Math.min(...cache.map((entry) => entry.timestamp))
            ).toLocaleString()
          : "없음";

      const newestEntry =
        cache.length > 0
          ? new Date(
              Math.max(...cache.map((entry) => entry.timestamp))
            ).toLocaleString()
          : "없음";

      return {
        totalEntries: cache.length,
        sizeInKB: Math.round(cacheSize * 100) / 100,
        oldestEntry,
        newestEntry,
        hitRate: Math.round(stats.hit_rate * 100),
      };
    } catch (error) {
      console.error("캐시 정보 조회 오류:", error);
      return {
        totalEntries: 0,
        sizeInKB: 0,
        oldestEntry: "오류",
        newestEntry: "오류",
        hitRate: 0,
      };
    }
  }

  /**
   * 모든 캐시 삭제
   */
  clearAllCache(): void {
    try {
      localStorage.removeItem(this.CACHE_KEY);
      localStorage.removeItem(this.STATS_KEY);
      debugLog("모든 캐시가 삭제되었습니다.");
    } catch (error) {
      console.error("캐시 삭제 오류:", error);
    }
  }

  /**
   * 특정 PDF의 캐시 삭제
   */
  clearPdfCache(pdfId: string): number {
    try {
      const cache = this.getCache();
      const originalLength = cache.length;
      
      const filteredCache = cache.filter(entry => entry.pdf_id !== pdfId);
      
      localStorage.setItem(this.CACHE_KEY, JSON.stringify(filteredCache));
      const removedCount = originalLength - filteredCache.length;
      
      debugLog(`PDF "${pdfId}"의 캐시가 삭제되었습니다. (${removedCount}개 항목 제거)`);
      return removedCount;
    } catch (error) {
      console.error("PDF 캐시 삭제 오류:", error);
      return 0;
    }
  }

  /**
   * 특정 질문의 캐시 삭제
   */
  clearQuestionCache(question: string, pdfId: string = "default_pdf"): void {
    try {
      const cache = this.getCache();
      const questionHash = this.hashQuestion(question);
      
      const filteredCache = cache.filter(
        (entry) => !(entry.question_hash === questionHash && entry.pdf_id === pdfId)
      );
      
      localStorage.setItem(this.CACHE_KEY, JSON.stringify(filteredCache));
      debugLog(`질문 "${question}"의 캐시가 삭제되었습니다.`);
    } catch (error) {
      console.error("특정 질문 캐시 삭제 오류:", error);
    }
  }

  /**
   * 특정 질문과 유사한 캐시된 질문들 찾기
   */
  findSimilarQuestions(question: string, limit: number = 5): CachedResponse[] {
    try {
      const cache = this.getCache();
      const similarQuestions: Array<{
        entry: CachedResponse;
        similarity: number;
      }> = [];

      for (const entry of cache) {
        const normalized1 = question.toLowerCase().trim();
        const normalized2 = entry.question.toLowerCase().trim();

        const words1 = new Set(normalized1.split(/\s+/));
        const words2 = new Set(normalized2.split(/\s+/));

        const intersection = new Set([...words1].filter((x) => words2.has(x)));
        const union = new Set([...words1, ...words2]);

        const similarity = intersection.size / union.size;

        if (similarity > 0.3) {
          // 30% 이상 유사한 질문만
          similarQuestions.push({ entry, similarity });
        }
      }

      // 유사도 순으로 정렬하고 상위 결과 반환
      return similarQuestions
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, limit)
        .map((item) => item.entry);
    } catch (error) {
      console.error("유사 질문 검색 오류:", error);
      return [];
    }
  }

  /**
   * 인사말인지 확인
   */
  isGreetingQuestion(question: string): boolean {
    const greetingPatterns = [
      '안녕', '반갑', '하이', 'hi', 'hello', '안녕하세요', '안녕하십니까',
      '반갑습니다', '만나서 반가워', '만나서 반가워요', '반가워', '반가워요',
      '좋은 아침', '좋은 오후', '좋은 저녁', '좋은 밤', '좋은 하루',
      '환영', '환영합니다', '오신 것을 환영', '환영해요',
      '뭐야', '뭔가', '뭔가요', '뭐예요', '무엇', '무엇을', '무엇인가', '무엇인가요',
      '시스템', '시스템이', '시스템이에요', '시스템입니다', '시스템이 뭐야',
      '기능', '기능이', '기능이에요', '기능입니다', '기능이 뭐야',
      '할 수', '할 수 있어', '할 수 있어요', '할 수 있나', '할 수 있나요',
      '궁금', '궁금해', '궁금해요', '궁금합니다', '궁금하네', '궁금하네요'
    ];
    
    const questionLower = question.toLowerCase();
    return greetingPatterns.some(pattern => questionLower.includes(pattern));
  }
}

// 싱글톤 인스턴스 생성
export const chatCache = new ChatCache();
