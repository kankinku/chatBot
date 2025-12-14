"""
Market Indices Module
주요 시장 지수 (Nasdaq, S&P 500, Gold, Bitcoin) 데이터 제공자
"""
import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from config.settings import settings
from src.core.logger import logger

# 캐시 파일 경로
CACHE_FILE = settings.CACHE_DIR / "indices_cache.json"


class MarketIndices:
    """
    주요 시장 지수 (Nasdaq, S&P 500, Gold, Bitcoin) 데이터 제공자
    - 서버 시작 시 initialize()로 데이터 미리 수집
    - 캐싱으로 API 호출 최소화
    """
    
    # Ticker Mapping
    INDICES = {
        "NASDAQ": {"ticker": "QQQ", "name": "Nasdaq 100", "color": "#0091ea"},
        "SNP500": {"ticker": "SPY", "name": "S&P 500", "color": "#ff3b30"},
        "GOLD": {"ticker": "GC=F", "name": "Gold", "color": "#f5a623"},
        "BTC": {"ticker": "BTC-USD", "name": "Bitcoin", "color": "#f7931a"}
    }

    def __init__(self):
        self.cache_path = CACHE_FILE
        self._ensure_cache_dir()
        self._cache: Dict = {}
        self._load_cache()

    def _ensure_cache_dir(self) -> None:
        """캐시 디렉토리 생성"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> None:
        """캐시 파일 로드"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except Exception as e:
                logger.warning(f"[Indices] Cache load failed: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """캐시 파일 저장"""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Indices] Cache save failed: {e}")

    def initialize(self, force: bool = False) -> None:
        """
        서버 시작 시 모든 지수 데이터 미리 수집
        
        Args:
            force: True면 캐시 무시하고 강제 갱신
        """
        logger.info("[Indices] Initializing market indices data...")
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        updated_count = 0
        
        for key in self.INDICES.keys():
            cache_key = f"{key}_1y"
            
            # 캐시 유효성 검사 (오늘 날짜와 동일하면 스킵)
            if not force and cache_key in self._cache:
                entry = self._cache[cache_key]
                if entry.get("date") == today_str:
                    logger.debug(f"[Indices] {key} already up-to-date")
                    continue
            
            # 데이터 수집
            result = self._fetch_index_data(key)
            if result and "error" not in result:
                self._cache[cache_key] = {
                    "date": today_str,
                    "data": result
                }
                updated_count += 1
                logger.info(f"[Indices] {key} updated: ${result['current_price']}")
            else:
                logger.warning(f"[Indices] {key} fetch failed: {result.get('error', 'Unknown')}")
        
        # 캐시 저장
        if updated_count > 0:
            self._save_cache()
            
        logger.info(f"[Indices] Initialization complete. Updated {updated_count}/{len(self.INDICES)} indices.")

    def _fetch_index_data(self, key: str, period: str = "1y") -> Optional[Dict[str, Any]]:
        """
        yfinance에서 지수 데이터 수집
        """
        if key not in self.INDICES:
            return {"error": "Invalid index key"}
        
        meta = self.INDICES[key]
        ticker = meta["ticker"]
        
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                return {"error": "No data found"}
            
            # Chart.js 형식으로 변환
            chart_data = []
            current_price = 0
            prev_price = 0
            
            if not df.empty:
                current_price = float(df['Close'].iloc[-1])
                prev_price = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
                
                for date, row in df.iterrows():
                    chart_data.append({
                        "x": date.strftime("%Y-%m-%d"),
                        "y": round(float(row['Close']), 2)
                    })
            
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price else 0
            
            return {
                "meta": meta,
                "current_price": round(current_price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "history": chart_data
            }
            
        except Exception as e:
            logger.error(f"[Indices] Fetch error for {key}: {e}")
            return {"error": str(e)}

    def get_index_data(self, key: str, period: str = "1y") -> Dict[str, Any]:
        """
        특정 지수의 데이터 반환 (캐시 우선)
        """
        if key not in self.INDICES:
            return {"error": f"Invalid index key: {key}. Available: {list(self.INDICES.keys())}"}
        
        cache_key = f"{key}_{period}"
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # 캐시 확인
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if entry.get("date") == today_str:
                return entry["data"]
        
        # 캐시 미스 → 수집
        result = self._fetch_index_data(key, period)
        if result and "error" not in result:
            self._cache[cache_key] = {
                "date": today_str,
                "data": result
            }
            self._save_cache()
            
        return result

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """딕셔너리 스타일 접근 지원"""
        return self.get_index_data(key)

    def __contains__(self, key: str) -> bool:
        """in 연산자 지원"""
        return key in self.INDICES

    def keys(self):
        """사용 가능한 지수 키 목록"""
        return self.INDICES.keys()


# 전역 인스턴스
market_indices = MarketIndices()
