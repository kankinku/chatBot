"""
Market Data Client
yfinance 등을 이용한 외부 데이터 수집
"""
import logging
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from config.settings import get_settings

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MarketDataClient:
    """외부 마켓 데이터 연동 클라이언트"""
    
    def __init__(self):
        self.settings = get_settings()
        self._config = self.settings.load_yaml_config("external_data")
        
        market_config = self._config.get("market_data", {})
        self.provider = market_config.get("provider", "yfinance")
        self._ticker_map = market_config.get("tickers", {})
        
        # API Keys
        api_keys = market_config.get("api_keys", {})
        self.fred_api_key = os.getenv(api_keys.get("fred_api_key_env", "FRED_API_KEY"))
        
        # 간단한 인메모리 캐시 (Ticker -> DataFrame)
        self._cache: Dict[str, pd.DataFrame] = {}
        
    def get_ticker(self, entity_name: str) -> Optional[str]:
        """엔티티 이름으로 티커 조회"""
        return self._ticker_map.get(entity_name)

        
    def get_price_history(
        self, 
        ticker: str, 
        days: int = 365
    ) -> pd.DataFrame:
        """
        주가 데이터 조회 (최근 N일)
        """
        if ticker in self._cache:
            # TODO: 캐시 만료 체크
            return self._cache[ticker]
            
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = yf.download(
                ticker, 
                start=start_date.strftime("%Y-%m-%d"), 
                end=end_date.strftime("%Y-%m-%d"), 
                progress=False
            )
            
            if not data.empty:
                self._cache[ticker] = data
                
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """최신 가격 조회"""
        try:
            ticker_obj = yf.Ticker(ticker)
            # fast_info 사용권장
            price = ticker_obj.fast_info.get('last_price')
            if price is None:
                # Fallback to history
                hist = ticker_obj.history(period="1d")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
            return price
        except Exception as e:
            logger.error(f"Failed to fetch price for {ticker}: {e}")
            return None
            
    def get_market_indicators(self) -> Dict[str, float]:
        """주요 시장 지표 조회"""
        # 주요 지표 키 리스트 (설정 파일의 키와 일치해야 함)
        target_indicators = ["S&P500", "코스피", "금리", "VIX지수", "환율"]
        
        results = {}
        for name in target_indicators:
            ticker = self.get_ticker(name)
            if ticker:
                price = self.get_latest_price(ticker)
                if price is not None:
                    results[name] = price
        return results
