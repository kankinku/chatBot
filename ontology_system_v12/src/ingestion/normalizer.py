"""
Normalizer
원시 데이터를 표준 Observation/Event로 변환

책임:
- 단위 변환
- 타임존 정규화
- 결측값 처리
- 표준 스키마로 변환
"""
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
import re

from src.shared.schemas import Observation, Event

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """정규화 결과"""
    success: bool
    observations: List[Observation] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    original_count: int = 0
    normalized_count: int = 0


class Normalizer:
    """
    데이터 정규화기
    
    다양한 형식의 원시 데이터를 표준 스키마로 변환합니다.
    """
    
    # 단위 변환 규칙
    UNIT_CONVERSIONS = {
        # 백분율
        "percent": ("%", 1.0),
        "pct": ("%", 1.0),
        "%": ("%", 1.0),
        "percentage": ("%", 1.0),
        # basis points
        "bp": ("bps", 1.0),
        "bps": ("bps", 1.0),
        "basis_points": ("bps", 1.0),
        # 이자율 변환 (0.0433 -> 4.33%)
        "decimal_rate": ("%", 100.0),
        # 가격
        "usd": ("$", 1.0),
        "$": ("$", 1.0),
        "dollar": ("$", 1.0),
    }
    
    # 품질 플래그
    QUALITY_OK = "ok"
    QUALITY_ESTIMATED = "estimated"
    QUALITY_MISSING = "missing"
    QUALITY_ERROR = "error"
    
    def __init__(self, default_timezone: str = "UTC"):
        """
        Args:
            default_timezone: 기본 타임존
        """
        self.default_timezone = default_timezone
    
    def normalize_observations(
        self,
        raw_data: List[Dict[str, Any]],
        source_id: str,
        series_id: str,
        unit_hint: Optional[str] = None,
    ) -> NormalizationResult:
        """
        원시 데이터를 Observation 리스트로 변환
        
        Args:
            raw_data: 원시 데이터 리스트
            source_id: 소스 ID
            series_id: 시계열 ID
            unit_hint: 단위 힌트
        
        Returns:
            NormalizationResult
        """
        result = NormalizationResult(original_count=len(raw_data), success=True)
        
        for i, raw in enumerate(raw_data):
            try:
                obs = self._normalize_single_observation(
                    raw, source_id, series_id, unit_hint
                )
                if obs:
                    result.observations.append(obs)
                    result.normalized_count += 1
            except Exception as e:
                result.errors.append(f"Row {i}: {str(e)}")
                result.success = False
        
        if result.errors:
            logger.warning(f"Normalization errors: {len(result.errors)}")
        
        return result
    
    def _normalize_single_observation(
        self,
        raw: Dict[str, Any],
        source_id: str,
        series_id: str,
        unit_hint: Optional[str],
    ) -> Optional[Observation]:
        """단일 관측값 정규화"""
        # 타임스탬프 파싱
        timestamp = self._parse_timestamp(
            raw.get('timestamp') or raw.get('date') or raw.get('time')
        )
        if not timestamp:
            raise ValueError("Missing or invalid timestamp")
        
        # 값 파싱
        value = self._parse_value(raw.get('value') or raw.get('close') or raw.get('price'))
        if value is None:
            raise ValueError("Missing or invalid value")
        
        # 단위 변환
        raw_unit = unit_hint or raw.get('unit', '')
        unit, multiplier = self._get_unit_conversion(raw_unit)
        normalized_value = value * multiplier
        
        # 품질 플래그
        quality = self._determine_quality(raw, normalized_value)
        
        return Observation(
            series_id=series_id,
            timestamp=timestamp,
            value=normalized_value,
            unit=unit,
            source_id=source_id,
            is_revision=raw.get('is_revision', False),
            original_value=raw.get('original_value'),
            quality_flag=quality,
        )
    
    def normalize_events(
        self,
        raw_data: List[Dict[str, Any]],
        source_id: str,
        event_type: str = "general",
    ) -> NormalizationResult:
        """
        원시 데이터를 Event 리스트로 변환
        
        Args:
            raw_data: 원시 데이터 리스트
            source_id: 소스 ID
            event_type: 이벤트 유형
        
        Returns:
            NormalizationResult
        """
        result = NormalizationResult(original_count=len(raw_data), success=True)
        
        for i, raw in enumerate(raw_data):
            try:
                event = self._normalize_single_event(raw, source_id, event_type)
                if event:
                    result.events.append(event)
                    result.normalized_count += 1
            except Exception as e:
                result.errors.append(f"Row {i}: {str(e)}")
                result.success = False
        
        return result
    
    def _normalize_single_event(
        self,
        raw: Dict[str, Any],
        source_id: str,
        event_type: str,
    ) -> Optional[Event]:
        """단일 이벤트 정규화"""
        # 타임스탬프 파싱
        occurred_at = self._parse_timestamp(
            raw.get('occurred_at') or raw.get('timestamp') or raw.get('date')
        )
        if not occurred_at:
            raise ValueError("Missing or invalid occurred_at")
        
        # 제목
        title = raw.get('title') or raw.get('headline') or ""
        if not title:
            raise ValueError("Missing title")
        
        # 본문
        content = raw.get('content') or raw.get('body') or raw.get('text')
        
        return Event(
            source_id=source_id,
            external_id=raw.get('external_id') or raw.get('id'),
            occurred_at=occurred_at,
            title=self._clean_text(title),
            content=self._clean_text(content) if content else None,
            event_type=event_type,
            entities=raw.get('entities', []),
            sentiment=self._parse_sentiment(raw.get('sentiment')),
            importance=raw.get('importance', 0.5),
            tags=raw.get('tags', []),
            metadata=raw.get('metadata', {}),
        )
    
    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        """타임스탬프 파싱"""
        if value is None:
            return None
        
        if isinstance(value, datetime):
            return self._ensure_timezone(value)
        
        if isinstance(value, (int, float)):
            # Unix timestamp
            return datetime.fromtimestamp(value, tz=timezone.utc)
        
        if isinstance(value, str):
            # 다양한 포맷 시도
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y/%m/%d",
                "%d-%m-%Y",
                "%m/%d/%Y",
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(value, fmt)
                    return self._ensure_timezone(dt)
                except ValueError:
                    continue
            
            # ISO format 시도
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                pass
        
        return None
    
    def _ensure_timezone(self, dt: datetime) -> datetime:
        """타임존 보장"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    
    def _parse_value(self, value: Any) -> Optional[float]:
        """값 파싱"""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            if value != value:  # NaN 체크
                return None
            return float(value)
        
        if isinstance(value, str):
            # 숫자 추출
            cleaned = re.sub(r'[^\d.\-+eE]', '', value)
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    def _get_unit_conversion(self, unit: str) -> tuple:
        """단위 변환 규칙 조회"""
        unit_lower = unit.lower().strip()
        if unit_lower in self.UNIT_CONVERSIONS:
            return self.UNIT_CONVERSIONS[unit_lower]
        return (unit or None, 1.0)
    
    def _determine_quality(self, raw: Dict[str, Any], value: float) -> str:
        """품질 플래그 결정"""
        if raw.get('is_estimated', False):
            return self.QUALITY_ESTIMATED
        if raw.get('is_missing', False):
            return self.QUALITY_MISSING
        if raw.get('is_error', False):
            return self.QUALITY_ERROR
        return self.QUALITY_OK
    
    def _parse_sentiment(self, value: Any) -> Optional[float]:
        """감성 점수 파싱 (-1 ~ 1)"""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return max(-1.0, min(1.0, float(value)))
        
        if isinstance(value, str):
            sentiment_map = {
                'positive': 0.5,
                'negative': -0.5,
                'neutral': 0.0,
                'very_positive': 0.8,
                'very_negative': -0.8,
            }
            return sentiment_map.get(value.lower())
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 앞뒤 공백 제거
        return text.strip()
