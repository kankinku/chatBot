// 개발 환경에서만 로그를 출력하는 유틸리티 함수들
export const debugLog = (...args: any[]) => {
  if (process.env.NODE_ENV === "development") {
    if (process.env.REACT_APP_ENABLE_LOGS !== "false") {
      console.log(...args);
    }
  }
};

export const errorLog = (...args: any[]) => {
  console.error(...args);
};

export const warnLog = (...args: any[]) => {
  console.warn(...args);
};

export const infoLog = (...args: any[]) => {
  if (process.env.NODE_ENV === "development") {
    console.info(...args);
  }
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
