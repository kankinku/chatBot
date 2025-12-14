"""
Fundamental Analysis Routes
펀더멘털 분석 관련 API 엔드포인트
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from src.services.fundamental_service import fundamental_analyzer
from src.core.logger import logger

router = APIRouter()


@router.get("/search")
async def search_stock(q: str = Query(..., min_length=1, description="종목 티커 또는 이름")):
    """
    종목 검색 및 기본 정보 반환
    """
    try:
        # 티커로 직접 조회 시도
        ticker = q.upper().strip()
        info = fundamental_analyzer.get_stock_info(ticker)
        return {"success": True, "data": info}
    except Exception as e:
        logger.warning(f"[Fundamental] Search failed for '{q}': {e}")
        raise HTTPException(status_code=404, detail=f"종목을 찾을 수 없습니다: {q}")


@router.get("/info/{ticker}")
async def get_stock_info(ticker: str):
    """
    종목 기본 정보 조회
    """
    try:
        info = fundamental_analyzer.get_stock_info(ticker)
        return {"success": True, "data": info}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/metrics/{ticker}")
async def get_financial_metrics(ticker: str):
    """
    연도별 재무 지표 조회 (히트맵용)
    - 매출성장률, 영업이익률, ROE, 부채비율 등
    """
    try:
        metrics = fundamental_analyzer.calculate_metrics(ticker)
        return {"success": True, "data": metrics}
    except Exception as e:
        logger.error(f"[Fundamental] Metrics fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scores/{ticker}")
async def get_scores(ticker: str):
    """
    종합 점수 조회 (성장성, 수익성, 안정성)
    """
    try:
        scores = fundamental_analyzer.calculate_scores(ticker)
        return {"success": True, "data": scores}
    except Exception as e:
        logger.error(f"[Fundamental] Scores fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cb/{ticker}")
async def get_convertible_bonds(ticker: str):
    """
    전환사채(CB) 정보 조회
    """
    try:
        cb_info = fundamental_analyzer.get_convertible_bonds(ticker)
        return {"success": True, "data": cb_info}
    except Exception as e:
        logger.error(f"[Fundamental] CB info fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{ticker}")
async def get_full_analysis(ticker: str):
    """
    전체 분석 데이터 반환 (단일 API 호출)
    - 기본 정보, 재무 지표, 점수, CB 정보 포함
    """
    try:
        result = fundamental_analyzer.get_full_analysis(ticker)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "분석 실패"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Fundamental] Full analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/financials/{ticker}")
async def get_financial_statements(ticker: str):
    """
    상세 재무제표 데이터 조회
    """
    try:
        financials = fundamental_analyzer.get_financial_statements(ticker)
        return {"success": True, "data": financials}
    except Exception as e:
        logger.error(f"[Fundamental] Financials fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
