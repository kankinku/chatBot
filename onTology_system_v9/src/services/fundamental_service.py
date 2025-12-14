"""
Fundamental Analysis Service
재무제표 분석 및 점수화 서비스
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from src.core.logger import logger


class FundamentalAnalyzer:
    """재무제표 분석 및 점수화 클래스"""
    
    # 분석 기준 설정 (config로 분리 가능)
    METRICS_CONFIG = {
        "growth": {
            "revenue_growth": {"weight": 0.4, "good": 10, "bad": -5},
            "earnings_growth": {"weight": 0.3, "good": 15, "bad": -10},
            "asset_growth": {"weight": 0.3, "good": 8, "bad": -3},
        },
        "profitability": {
            "gross_margin": {"weight": 0.25, "good": 30, "bad": 10},
            "operating_margin": {"weight": 0.35, "good": 15, "bad": 0},
            "net_margin": {"weight": 0.2, "good": 10, "bad": 0},
            "roe": {"weight": 0.2, "good": 15, "bad": 5},
        },
        "stability": {
            "debt_to_equity": {"weight": 0.35, "good": 50, "bad": 150, "inverse": True},
            "current_ratio": {"weight": 0.35, "good": 1.5, "bad": 1.0},
            "interest_coverage": {"weight": 0.3, "good": 5, "bad": 1.5},
        },
    }
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = timedelta(hours=1)
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """종목 기본 정보 조회"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "ticker": ticker.upper(),
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "dividend_yield": info.get("dividendYield"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "beta": info.get("beta"),
                "currency": info.get("currency", "USD"),
            }
        except Exception as e:
            logger.error(f"[Fundamental] Failed to get stock info for {ticker}: {e}")
            raise ValueError(f"종목 정보를 찾을 수 없습니다: {ticker}")
    
    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """재무제표 데이터 조회 (연간 + 분기)"""
        try:
            stock = yf.Ticker(ticker)
            
            # 연간 재무제표
            income_annual = stock.income_stmt
            balance_annual = stock.balance_sheet
            cashflow_annual = stock.cashflow
            
            # 분기 재무제표
            income_quarterly = stock.quarterly_income_stmt
            balance_quarterly = stock.quarterly_balance_sheet
            
            return {
                "annual": {
                    "income_statement": self._df_to_dict(income_annual),
                    "balance_sheet": self._df_to_dict(balance_annual),
                    "cash_flow": self._df_to_dict(cashflow_annual),
                },
                "quarterly": {
                    "income_statement": self._df_to_dict(income_quarterly),
                    "balance_sheet": self._df_to_dict(balance_quarterly),
                }
            }
        except Exception as e:
            logger.error(f"[Fundamental] Failed to get financials for {ticker}: {e}")
            raise ValueError(f"재무제표를 가져올 수 없습니다: {ticker}")
    
    def _df_to_dict(self, df: pd.DataFrame) -> Dict:
        """DataFrame을 JSON 직렬화 가능한 dict로 변환"""
        if df is None or df.empty:
            return {}
        
        result = {}
        for col in df.columns:
            col_key = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
            col_data = {}
            for idx, val in df[col].items():
                if pd.notna(val):
                    col_data[str(idx)] = float(val) if isinstance(val, (int, float, np.number)) else val
            result[col_key] = col_data
        
        return result
    
    def calculate_metrics(self, ticker: str) -> Dict[str, Any]:
        """핵심 재무 지표 계산 (연도별 히트맵용)"""
        try:
            stock = yf.Ticker(ticker)
            income = stock.income_stmt
            balance = stock.balance_sheet
            
            if income.empty or balance.empty:
                raise ValueError("재무 데이터가 부족합니다")
            
            metrics_by_year = []
            years = list(income.columns)[:5]  # 최근 5년
            
            for i, year in enumerate(years):
                year_str = year.strftime("%Y") if hasattr(year, "strftime") else str(year)[:4]
                
                metrics = {"year": year_str}
                
                # 수익 관련
                revenue = self._safe_get(income, "Total Revenue", year)
                gross_profit = self._safe_get(income, "Gross Profit", year)
                operating_income = self._safe_get(income, "Operating Income", year)
                net_income = self._safe_get(income, "Net Income", year)
                
                # 자산 관련
                total_assets = self._safe_get(balance, "Total Assets", year)
                total_equity = self._safe_get(balance, "Total Equity Gross Minority Interest", year)
                if total_equity is None:
                    total_equity = self._safe_get(balance, "Stockholders Equity", year)
                total_debt = self._safe_get(balance, "Total Debt", year)
                current_assets = self._safe_get(balance, "Current Assets", year)
                current_liabilities = self._safe_get(balance, "Current Liabilities", year)
                
                # 전년도 데이터 (성장률 계산용)
                prev_revenue = None
                prev_earnings = None
                prev_assets = None
                if i + 1 < len(years):
                    prev_year = years[i + 1]
                    prev_revenue = self._safe_get(income, "Total Revenue", prev_year)
                    prev_earnings = self._safe_get(income, "Net Income", prev_year)
                    prev_assets = self._safe_get(balance, "Total Assets", prev_year)
                
                # 성장 지표
                metrics["revenue_growth"] = self._calc_growth(revenue, prev_revenue)
                metrics["earnings_growth"] = self._calc_growth(net_income, prev_earnings)
                metrics["asset_growth"] = self._calc_growth(total_assets, prev_assets)
                
                # 수익성 지표
                metrics["gross_margin"] = self._calc_ratio(gross_profit, revenue)
                metrics["operating_margin"] = self._calc_ratio(operating_income, revenue)
                metrics["net_margin"] = self._calc_ratio(net_income, revenue)
                metrics["roe"] = self._calc_ratio(net_income, total_equity)
                metrics["roa"] = self._calc_ratio(net_income, total_assets)
                
                # 안정성 지표
                metrics["debt_to_equity"] = self._calc_ratio(total_debt, total_equity)
                metrics["current_ratio"] = self._calc_ratio(current_assets, current_liabilities, as_pct=False)
                
                # 절대값
                metrics["revenue"] = revenue
                metrics["net_income"] = net_income
                metrics["total_assets"] = total_assets
                metrics["total_debt"] = total_debt
                
                metrics_by_year.append(metrics)
            
            return {
                "ticker": ticker.upper(),
                "metrics": metrics_by_year,
                "latest_year": metrics_by_year[0]["year"] if metrics_by_year else None
            }
            
        except Exception as e:
            logger.error(f"[Fundamental] Metrics calculation failed for {ticker}: {e}")
            raise ValueError(f"지표 계산 실패: {e}")
    
    def _safe_get(self, df: pd.DataFrame, key: str, col) -> Optional[float]:
        """DataFrame에서 안전하게 값 추출"""
        try:
            if key in df.index and col in df.columns:
                val = df.loc[key, col]
                if pd.notna(val):
                    return float(val)
        except Exception:
            pass
        return None
    
    def _calc_growth(self, current: Optional[float], previous: Optional[float]) -> Optional[float]:
        """성장률 계산"""
        if current is None or previous is None or previous == 0:
            return None
        return round(((current - previous) / abs(previous)) * 100, 2)
    
    def _calc_ratio(self, numerator: Optional[float], denominator: Optional[float], as_pct: bool = True) -> Optional[float]:
        """비율 계산"""
        if numerator is None or denominator is None or denominator == 0:
            return None
        ratio = numerator / denominator
        return round(ratio * 100, 2) if as_pct else round(ratio, 2)
    
    def calculate_scores(self, ticker: str) -> Dict[str, Any]:
        """종합 점수 계산 (성장성, 수익성, 안정성)"""
        try:
            metrics_data = self.calculate_metrics(ticker)
            latest = metrics_data["metrics"][0] if metrics_data["metrics"] else {}
            
            scores = {}
            
            # 성장성 점수
            growth_score = self._calc_category_score(latest, "growth")
            scores["growth"] = {
                "score": growth_score,
                "grade": self._get_grade(growth_score),
                "details": {
                    "revenue_growth": latest.get("revenue_growth"),
                    "earnings_growth": latest.get("earnings_growth"),
                    "asset_growth": latest.get("asset_growth"),
                }
            }
            
            # 수익성 점수
            profitability_score = self._calc_category_score(latest, "profitability")
            scores["profitability"] = {
                "score": profitability_score,
                "grade": self._get_grade(profitability_score),
                "details": {
                    "gross_margin": latest.get("gross_margin"),
                    "operating_margin": latest.get("operating_margin"),
                    "net_margin": latest.get("net_margin"),
                    "roe": latest.get("roe"),
                }
            }
            
            # 안정성 점수
            stability_score = self._calc_category_score(latest, "stability")
            scores["stability"] = {
                "score": stability_score,
                "grade": self._get_grade(stability_score),
                "details": {
                    "debt_to_equity": latest.get("debt_to_equity"),
                    "current_ratio": latest.get("current_ratio"),
                }
            }
            
            # 종합 점수
            total_score = round((growth_score + profitability_score + stability_score) / 3, 1)
            
            return {
                "ticker": ticker.upper(),
                "total_score": total_score,
                "total_grade": self._get_grade(total_score),
                "categories": scores,
                "updated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[Fundamental] Score calculation failed: {e}")
            raise ValueError(f"점수 계산 실패: {e}")
    
    def _calc_category_score(self, metrics: Dict, category: str) -> float:
        """카테고리별 점수 계산"""
        config = self.METRICS_CONFIG.get(category, {})
        total_weight = 0
        weighted_score = 0
        
        for metric_name, cfg in config.items():
            value = metrics.get(metric_name)
            if value is None:
                continue
            
            weight = cfg["weight"]
            good = cfg["good"]
            bad = cfg["bad"]
            inverse = cfg.get("inverse", False)
            
            # 점수 계산 (0-100 스케일)
            if inverse:
                # 낮을수록 좋음 (예: 부채비율)
                if value <= good:
                    score = 100
                elif value >= bad:
                    score = 0
                else:
                    score = 100 - ((value - good) / (bad - good)) * 100
            else:
                # 높을수록 좋음
                if value >= good:
                    score = 100
                elif value <= bad:
                    score = 0
                else:
                    score = ((value - bad) / (good - bad)) * 100
            
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 50.0  # 데이터 없으면 중간값
        
        return round(weighted_score / total_weight, 1)
    
    def _get_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C+"
        elif score >= 40:
            return "C"
        elif score >= 30:
            return "D"
        else:
            return "F"
    
    def get_convertible_bonds(self, ticker: str) -> Dict[str, Any]:
        """
        전환사채(CB) 정보 조회
        Note: yfinance는 CB 데이터를 직접 제공하지 않음
        SEC EDGAR 또는 별도 데이터 소스 필요
        현재는 기본 채권 정보만 제공
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            balance = stock.balance_sheet
            
            current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))
            
            # 총 부채 정보
            total_debt = None
            long_term_debt = None
            
            if not balance.empty:
                latest_col = balance.columns[0]
                total_debt = self._safe_get(balance, "Total Debt", latest_col)
                long_term_debt = self._safe_get(balance, "Long Term Debt", latest_col)
            
            # CB 관련 정보는 SEC 8-K/10-K 파싱 필요 (추후 구현)
            # 현재는 기본 부채 정보만 반환
            return {
                "ticker": ticker.upper(),
                "current_price": current_price,
                "currency": info.get("currency", "USD"),
                "debt_info": {
                    "total_debt": total_debt,
                    "long_term_debt": long_term_debt,
                },
                "convertible_bonds": [],  # TODO: SEC EDGAR 연동
                "message": "전환사채 상세 정보는 SEC EDGAR 연동 후 제공 예정",
            }
            
        except Exception as e:
            logger.error(f"[Fundamental] CB info fetch failed: {e}")
            return {
                "ticker": ticker.upper(),
                "convertible_bonds": [],
                "error": str(e)
            }
    
    def get_full_analysis(self, ticker: str) -> Dict[str, Any]:
        """전체 분석 데이터 반환 (단일 API 호출용)"""
        try:
            info = self.get_stock_info(ticker)
            metrics = self.calculate_metrics(ticker)
            scores = self.calculate_scores(ticker)
            cb_info = self.get_convertible_bonds(ticker)
            
            return {
                "success": True,
                "data": {
                    "info": info,
                    "metrics": metrics,
                    "scores": scores,
                    "convertible_bonds": cb_info,
                }
            }
        except Exception as e:
            logger.error(f"[Fundamental] Full analysis failed for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# 싱글톤 인스턴스
fundamental_analyzer = FundamentalAnalyzer()
