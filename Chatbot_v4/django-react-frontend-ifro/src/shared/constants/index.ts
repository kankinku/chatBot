// App constants
export const GOOGLE_MAPS_API_KEY = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;
export const LIMA_CENTER = { lat: -12.0464, lng: -77.0428 };
export const DEFAULT_ZOOM = 12;

// API endpoints
export const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

// App routes
export const ROUTES = {
  HOME: "/",
  DASHBOARD: "/dashboard",
  LOGIN: "/login",
  REGISTER: "/register",
  PROPOSALS: "/proposals",
  TRAFFIC_ANALYSIS: "/traffic-analysis",
} as const;

// App settings
export const APP_CONFIG = {
  DEFAULT_LANGUAGE: "es",
  SUPPORTED_LANGUAGES: ["es", "en"],
  CHART_COLORS: {
    PRIMARY: "#3B82F6",
    SECONDARY: "#10B981",
    ACCENT: "#F59E0B",
    DANGER: "#EF4444",
  },
} as const;
