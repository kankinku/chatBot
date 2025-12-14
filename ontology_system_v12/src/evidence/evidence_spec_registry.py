"""
Edge Evidence Spec Registry
관계 타입/패턴별 Evidence 스펙 관리

책임:
- Edge-Feature 매핑 정의
- 점수 규칙 관리
- 스펙 조회
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import re

from src.shared.schemas import EvidenceSpec, RegimeType

logger = logging.getLogger(__name__)


class EdgeEvidenceSpecRegistry:
    """
    Edge Evidence 스펙 레지스트리
    
    관계 타입(또는 특정 Edge)에 대해:
    - 필요한 Feature 목록
    - 점수 규칙 (임계값/방향/가중치)
    - lag, decay, regime applicability
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: evidence_specs.yaml 경로
        """
        self._specs: Dict[str, EvidenceSpec] = {}
        self._pattern_index: List[tuple] = []  # (패턴, spec_id) 리스트
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self._register_default_specs()
    
    def load_from_file(self, config_path: str) -> int:
        """YAML에서 스펙 로드"""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Evidence config not found: {config_path}")
            self._register_default_specs()
            return 0
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            specs = config.get('edge_evidence', [])
            loaded = 0
            
            for spec_data in specs:
                spec = self._parse_spec(spec_data)
                if spec:
                    self.register(spec)
                    loaded += 1
            
            logger.info(f"Loaded {loaded} evidence specs from {config_path}")
            return loaded
            
        except Exception as e:
            logger.error(f"Failed to load evidence config: {e}")
            self._register_default_specs()
            return 0
    
    def _parse_spec(self, data: Dict) -> Optional[EvidenceSpec]:
        """Dict에서 EvidenceSpec 파싱"""
        try:
            edge_pattern = data.get('edge_pattern', {})
            
            # regime_applicability 파싱
            regime_app = {}
            for k, v in data.get('regime_applicability', {}).items():
                try:
                    regime_app[k] = float(v)
                except:
                    pass
            
            return EvidenceSpec(
                edge_pattern=edge_pattern,
                evidence_features=data.get('evidence_features', []),
                thresholds=data.get('thresholds', {}),
                lag_days=data.get('lag_days', [0, 30, 60]),
                regime_applicability=regime_app,
                is_active=data.get('is_active', True),
            )
        except Exception as e:
            logger.error(f"Failed to parse evidence spec: {e}")
            return None
    
    def _register_default_specs(self) -> None:
        """기본 스펙 등록"""
        defaults = [
            # 금리 상승 → 성장주 하락
            EvidenceSpec(
                edge_pattern={
                    "head_type": "Indicator",
                    "head_contains": ["금리", "interest_rate", "SOFR", "fed_funds"],
                    "tail_type": "Asset",
                    "tail_contains": ["성장주", "growth_stock", "NASDAQ", "tech"],
                    "relation_type": "Affect",
                    "polarity": "-",
                },
                evidence_features=[
                    {"feature": "SOFR_ROC_30D", "direction": "positive", "weight": 0.4},
                    {"feature": "VIX_ROC_5D", "direction": "positive", "weight": 0.3},
                ],
                thresholds={
                    "strong_pro": 0.7,
                    "weak_pro": 0.3,
                    "neutral_low": -0.3,
                    "neutral_high": 0.3,
                    "weak_con": -0.3,
                    "strong_con": -0.7,
                },
                lag_days=[0, 30, 60],
                regime_applicability={
                    "risk_on": 0.8,
                    "risk_off": 1.2,
                    "inflation_up": 1.0,
                },
            ),
            # VIX 상승 → 주식 하락
            EvidenceSpec(
                edge_pattern={
                    "head_contains": ["VIX", "변동성", "volatility"],
                    "tail_contains": ["주식", "stock", "equity", "S&P", "SPY"],
                    "relation_type": "Affect",
                    "polarity": "-",
                },
                evidence_features=[
                    {"feature": "VIX_ZSCORE_30D", "direction": "positive", "weight": 0.5},
                    {"feature": "SPY_VOL_20D", "direction": "positive", "weight": 0.3},
                ],
                thresholds={
                    "strong_pro": 0.6,
                    "weak_pro": 0.2,
                },
                lag_days=[0, 5],
                regime_applicability={
                    "risk_on": 0.6,
                    "risk_off": 1.4,
                },
            ),
            # 인플레이션 → 금 상승
            EvidenceSpec(
                edge_pattern={
                    "head_contains": ["인플레", "CPI", "inflation"],
                    "tail_contains": ["금", "gold", "GLD"],
                    "relation_type": "Affect",
                    "polarity": "+",
                },
                evidence_features=[
                    {"feature": "CPI_YOY", "direction": "positive", "weight": 0.5},
                ],
                thresholds={
                    "strong_pro": 0.6,
                },
                lag_days=[0, 30, 90],
                regime_applicability={
                    "inflation_up": 1.3,
                    "disinflation": 0.7,
                },
            ),
        ]
        
        for spec in defaults:
            self.register(spec)
        
        logger.debug(f"Registered {len(defaults)} default evidence specs")
    
    def register(self, spec: EvidenceSpec) -> None:
        """스펙 등록"""
        self._specs[spec.spec_id] = spec
        
        # 패턴 인덱스 업데이트
        self._pattern_index.append((spec.edge_pattern, spec.spec_id))
        
        logger.debug(f"Registered evidence spec: {spec.spec_id}")
    
    def get(self, spec_id: str) -> Optional[EvidenceSpec]:
        """ID로 스펙 조회"""
        return self._specs.get(spec_id)
    
    def find_matching_spec(
        self,
        head_id: str,
        head_name: str,
        head_type: str,
        tail_id: str,
        tail_name: str,
        tail_type: str,
        relation_type: str,
        polarity: str,
    ) -> Optional[EvidenceSpec]:
        """
        Edge 정보로 매칭되는 스펙 찾기
        
        Returns:
            가장 잘 매칭되는 EvidenceSpec 또는 None
        """
        best_match = None
        best_score = 0
        
        for pattern, spec_id in self._pattern_index:
            score = self._match_pattern(
                pattern,
                head_id, head_name, head_type,
                tail_id, tail_name, tail_type,
                relation_type, polarity
            )
            
            if score > best_score:
                best_score = score
                best_match = self._specs[spec_id]
        
        return best_match if best_score > 0 else None
    
    def find_all_matching_specs(
        self,
        head_id: str,
        head_name: str,
        head_type: str,
        tail_id: str,
        tail_name: str,
        tail_type: str,
        relation_type: str,
        polarity: str,
    ) -> List[EvidenceSpec]:
        """매칭되는 모든 스펙 찾기"""
        matches = []
        
        for pattern, spec_id in self._pattern_index:
            score = self._match_pattern(
                pattern,
                head_id, head_name, head_type,
                tail_id, tail_name, tail_type,
                relation_type, polarity
            )
            
            if score > 0:
                matches.append((score, self._specs[spec_id]))
        
        # 점수순 정렬
        matches.sort(key=lambda x: x[0], reverse=True)
        return [spec for _, spec in matches]
    
    def _match_pattern(
        self,
        pattern: Dict,
        head_id: str,
        head_name: str,
        head_type: str,
        tail_id: str,
        tail_name: str,
        tail_type: str,
        relation_type: str,
        polarity: str,
    ) -> float:
        """패턴 매칭 점수 계산 (0~1)"""
        score = 0.0
        checks = 0
        
        # head_type 체크
        if "head_type" in pattern:
            checks += 1
            if pattern["head_type"].lower() in head_type.lower():
                score += 1
        
        # head_contains 체크
        if "head_contains" in pattern:
            checks += 1
            for keyword in pattern["head_contains"]:
                if keyword.lower() in head_name.lower() or keyword.lower() in head_id.lower():
                    score += 1
                    break
        
        # tail_type 체크
        if "tail_type" in pattern:
            checks += 1
            if pattern["tail_type"].lower() in tail_type.lower():
                score += 1
        
        # tail_contains 체크
        if "tail_contains" in pattern:
            checks += 1
            for keyword in pattern["tail_contains"]:
                if keyword.lower() in tail_name.lower() or keyword.lower() in tail_id.lower():
                    score += 1
                    break
        
        # relation_type 체크
        if "relation_type" in pattern:
            checks += 1
            if pattern["relation_type"].lower() == relation_type.lower():
                score += 1
        
        # polarity 체크
        if "polarity" in pattern:
            checks += 1
            if pattern["polarity"] == polarity:
                score += 1
        
        return score / checks if checks > 0 else 0
    
    def get_by_relation_type(self, relation_type: str) -> List[EvidenceSpec]:
        """관계 타입으로 스펙 조회"""
        result = []
        for pattern, spec_id in self._pattern_index:
            if pattern.get("relation_type", "").lower() == relation_type.lower():
                result.append(self._specs[spec_id])
        return result
    
    def list_all(self) -> List[EvidenceSpec]:
        """모든 스펙 목록"""
        return list(self._specs.values())
    
    def get_required_features(self) -> List[str]:
        """모든 스펙에 필요한 Feature 목록 (중복 제거)"""
        features = set()
        for spec in self._specs.values():
            for ef in spec.evidence_features:
                features.add(ef.get("feature", ""))
        return list(features)
    
    def get_stats(self) -> Dict:
        """통계"""
        relation_types = {}
        for pattern, _ in self._pattern_index:
            rt = pattern.get("relation_type", "unknown")
            relation_types[rt] = relation_types.get(rt, 0) + 1
        
        return {
            "total_specs": len(self._specs),
            "required_features": len(self.get_required_features()),
            "by_relation_type": relation_types,
        }
