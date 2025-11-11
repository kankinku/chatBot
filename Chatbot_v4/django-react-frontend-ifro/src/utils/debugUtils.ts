// 개발 환경에서만 로그를 출력하는 유틸리티 함수들
export const debugLog = (...args: any[]) => {
  if (process.env.NODE_ENV === "development") {
    console.log(...args);
  }
};

export const debugUtils = {
  log: (...args: any[]) => {
    if (process.env.NODE_ENV === "development") {
      console.log(...args);
    }
  },

  warn: (...args: any[]) => {
    if (process.env.NODE_ENV === "development") {
      console.warn(...args);
    }
  },

  error: (...args: any[]) => {
    // 에러는 프로덕션에서도 출력
    console.error(...args);
  },

  info: (...args: any[]) => {
    if (process.env.NODE_ENV === "development") {
      console.info(...args);
    }
  },
};

// 성능 측정을 위한 유틸리티
export const performanceLog = {
  start: (label: string) => {
    if (process.env.NODE_ENV === "development") {
      console.time(label);
    }
  },

  end: (label: string) => {
    if (process.env.NODE_ENV === "development") {
      console.timeEnd(label);
    }
  },
};
