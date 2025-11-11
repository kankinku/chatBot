import axios, {
  InternalAxiosRequestConfig,
  AxiosResponse,
  AxiosError,
} from "axios";
import { debugLog } from "../shared/utils/debugUtils";

const api = axios.create({
  baseURL: "http://localhost:8000/api/", // Django backend URL (api 루트로 변경)
  timeout: 60000, // 60초로 증가

  headers: {
    "Content-Type": "application/json",
  },
});

// 요청 인터셉터
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem("access");

    if (token) {
      config.headers["Authorization"] = `Bearer ${token}`;
    }

    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

// 응답 인터셉터
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // 응답 데이터를 가공
    debugLog("API Response Data:", response.data);
    return response;
  },
  (error: AxiosError) => {
    // 오류 응답을 처리
    console.error("API Error:", error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export default api;
