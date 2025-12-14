"""
Feature Dependency Index
series → feature 의존성 맵 관리

책임:
- 의존성 그래프 구축
- 영향받는 Feature 조회 (증분 업데이트용)
- 역인덱스 유지
"""
import logging
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field

from src.shared.schemas import FeatureSpec
from src.features.feature_spec_registry import FeatureSpecRegistry

logger = logging.getLogger(__name__)


@dataclass
class DependencyNode:
    """의존성 노드"""
    id: str
    node_type: str  # "series" or "feature"
    depends_on: Set[str] = field(default_factory=set)   # 이 노드가 의존하는 것들
    depended_by: Set[str] = field(default_factory=set)  # 이 노드에 의존하는 것들


class FeatureDependencyIndex:
    """
    Feature 의존성 인덱스
    
    series → feature 의존성을 관리하여 증분 업데이트를 지원합니다.
    
    예: SOFR 업데이트 시 → SOFR_ROC_30D, SOFR_ZSCORE_90D 영향
    """
    
    def __init__(self, spec_registry: Optional[FeatureSpecRegistry] = None):
        """
        Args:
            spec_registry: Feature 스펙 레지스트리 (None이면 빈 상태)
        """
        self._nodes: Dict[str, DependencyNode] = {}
        self._series_to_features: Dict[str, Set[str]] = {}  # 빠른 조회용
        
        if spec_registry:
            self.build_from_registry(spec_registry)
    
    def build_from_registry(self, registry: FeatureSpecRegistry) -> int:
        """
        스펙 레지스트리에서 의존성 인덱스 구축
        
        Returns:
            구축된 의존성 수
        """
        self._nodes.clear()
        self._series_to_features.clear()
        
        dependency_count = 0
        
        for spec in registry.list_all():
            # Feature 노드 생성
            feature_node = self._ensure_node(spec.feature_id, "feature")
            
            # 입력 시계열에 대한 의존성 추가
            for series_id in spec.input_series:
                series_node = self._ensure_node(series_id, "series")
                
                # 양방향 연결
                feature_node.depends_on.add(series_id)
                series_node.depended_by.add(spec.feature_id)
                
                # 빠른 조회용 인덱스
                if series_id not in self._series_to_features:
                    self._series_to_features[series_id] = set()
                self._series_to_features[series_id].add(spec.feature_id)
                
                dependency_count += 1
        
        logger.info(f"Built dependency index: {dependency_count} dependencies")
        return dependency_count
    
    def _ensure_node(self, node_id: str, node_type: str) -> DependencyNode:
        """노드 생성 또는 조회"""
        if node_id not in self._nodes:
            self._nodes[node_id] = DependencyNode(id=node_id, node_type=node_type)
        return self._nodes[node_id]
    
    def add_dependency(
        self,
        feature_id: str,
        series_id: str,
    ) -> None:
        """의존성 추가"""
        feature_node = self._ensure_node(feature_id, "feature")
        series_node = self._ensure_node(series_id, "series")
        
        feature_node.depends_on.add(series_id)
        series_node.depended_by.add(feature_id)
        
        if series_id not in self._series_to_features:
            self._series_to_features[series_id] = set()
        self._series_to_features[series_id].add(feature_id)
    
    def remove_dependency(
        self,
        feature_id: str,
        series_id: str,
    ) -> bool:
        """의존성 제거"""
        if feature_id not in self._nodes or series_id not in self._nodes:
            return False
        
        feature_node = self._nodes[feature_id]
        series_node = self._nodes[series_id]
        
        feature_node.depends_on.discard(series_id)
        series_node.depended_by.discard(feature_id)
        
        if series_id in self._series_to_features:
            self._series_to_features[series_id].discard(feature_id)
        
        return True
    
    def get_affected_features(self, series_id: str) -> List[str]:
        """
        시계열 업데이트 시 영향받는 Feature 목록
        
        Args:
            series_id: 업데이트된 시계열 ID
        
        Returns:
            재계산이 필요한 Feature ID 리스트
        """
        return list(self._series_to_features.get(series_id, set()))
    
    def get_affected_features_batch(self, series_ids: List[str]) -> List[str]:
        """
        여러 시계열 업데이트 시 영향받는 Feature 목록 (중복 제거)
        
        Args:
            series_ids: 업데이트된 시계열 ID 리스트
        
        Returns:
            재계산이 필요한 Feature ID 리스트
        """
        affected = set()
        for series_id in series_ids:
            affected.update(self._series_to_features.get(series_id, set()))
        return list(affected)
    
    def get_dependencies(self, feature_id: str) -> List[str]:
        """Feature가 의존하는 시계열 목록"""
        node = self._nodes.get(feature_id)
        if not node:
            return []
        return list(node.depends_on)
    
    def get_dependents(self, series_id: str) -> List[str]:
        """시계열에 의존하는 Feature 목록"""
        node = self._nodes.get(series_id)
        if not node:
            return []
        return list(node.depended_by)
    
    def get_all_series(self) -> List[str]:
        """모든 시계열 ID"""
        return [
            node.id for node in self._nodes.values()
            if node.node_type == "series"
        ]
    
    def get_all_features(self) -> List[str]:
        """모든 Feature ID"""
        return [
            node.id for node in self._nodes.values()
            if node.node_type == "feature"
        ]
    
    def get_topological_order(self) -> List[str]:
        """
        토폴로지 정렬 순서로 Feature 목록 반환
        
        의존성이 없는 것부터 있는 것 순으로 정렬.
        계산 순서 결정에 사용.
        """
        # 현재는 단순 리턴 (Feature간 의존성이 없는 경우)
        # Feature가 다른 Feature에 의존하는 경우 확장 필요
        return self.get_all_features()
    
    def validate_dependencies(self, registry: FeatureSpecRegistry) -> List[str]:
        """
        의존성 유효성 검사
        
        Returns:
            에러 메시지 리스트 (빈 리스트면 유효)
        """
        errors = []
        
        for spec in registry.list_all():
            for series_id in spec.input_series:
                if series_id not in self._nodes:
                    errors.append(
                        f"Feature '{spec.feature_id}' depends on unknown series '{series_id}'"
                    )
        
        # 순환 의존성 체크 (현재는 Feature→Series만 있어서 불필요)
        
        return errors
    
    def get_stats(self) -> Dict:
        """통계"""
        series_count = sum(1 for n in self._nodes.values() if n.node_type == "series")
        feature_count = sum(1 for n in self._nodes.values() if n.node_type == "feature")
        
        total_deps = sum(len(n.depends_on) for n in self._nodes.values())
        
        # 가장 많이 사용되는 시계열
        most_used = sorted(
            self._series_to_features.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        
        return {
            "total_nodes": len(self._nodes),
            "series_count": series_count,
            "feature_count": feature_count,
            "total_dependencies": total_deps,
            "most_used_series": [
                {"series_id": s, "feature_count": len(f)}
                for s, f in most_used
            ],
        }
