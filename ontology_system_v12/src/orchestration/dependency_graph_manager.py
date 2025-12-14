"""
Dependency Graph Manager
series → feature → evidence → edge 의존성 그래프 관리

책임:
- 전체 의존성 그래프 구축
- 영향 범위 산출
- 역인덱스 유지
"""
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """노드 타입"""
    SERIES = "series"
    FEATURE = "feature"
    EVIDENCE = "evidence"
    EDGE = "edge"


@dataclass
class DependencyNode:
    """의존성 노드"""
    node_id: str
    node_type: NodeType
    upstream: Set[str] = field(default_factory=set)    # 이 노드가 의존하는 노드들
    downstream: Set[str] = field(default_factory=set)  # 이 노드에 의존하는 노드들


class DependencyGraphManager:
    """
    전체 의존성 그래프 관리자
    
    의존성 체인:
    Series → Feature → Evidence → Edge
    
    예:
    SOFR (series) → SOFR_ROC_30D (feature) → evidence_001 → edge_001
    """
    
    def __init__(self):
        self._nodes: Dict[str, DependencyNode] = {}
        
        # 타입별 인덱스
        self._by_type: Dict[NodeType, Set[str]] = {
            NodeType.SERIES: set(),
            NodeType.FEATURE: set(),
            NodeType.EVIDENCE: set(),
            NodeType.EDGE: set(),
        }
    
    def add_node(self, node_id: str, node_type: NodeType) -> DependencyNode:
        """노드 추가"""
        if node_id not in self._nodes:
            self._nodes[node_id] = DependencyNode(
                node_id=node_id,
                node_type=node_type,
            )
            self._by_type[node_type].add(node_id)
        return self._nodes[node_id]
    
    def add_dependency(
        self,
        from_id: str,
        from_type: NodeType,
        to_id: str,
        to_type: NodeType,
    ) -> None:
        """
        의존성 추가 (from → to, from이 to에 의존)
        
        Args:
            from_id: 의존하는 노드 ID
            from_type: 의존하는 노드 타입
            to_id: 의존되는 노드 ID (upstream)
            to_type: 의존되는 노드 타입
        """
        from_node = self.add_node(from_id, from_type)
        to_node = self.add_node(to_id, to_type)
        
        from_node.upstream.add(to_id)
        to_node.downstream.add(from_id)
    
    def get_affected_downstream(
        self,
        node_id: str,
        max_depth: int = 10,
    ) -> Dict[NodeType, Set[str]]:
        """
        특정 노드 변경 시 영향받는 모든 downstream 노드
        
        Args:
            node_id: 변경된 노드 ID
            max_depth: 최대 탐색 깊이
        
        Returns:
            {NodeType: Set[node_ids]} 형태의 영향받는 노드들
        """
        affected: Dict[NodeType, Set[str]] = {
            NodeType.SERIES: set(),
            NodeType.FEATURE: set(),
            NodeType.EVIDENCE: set(),
            NodeType.EDGE: set(),
        }
        
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)
            
            node = self._nodes.get(current_id)
            if not node:
                continue
            
            # 시작 노드는 affected에 추가하지 않음
            if current_id != node_id:
                affected[node.node_type].add(current_id)
            
            # downstream 노드들 탐색
            for downstream_id in node.downstream:
                if downstream_id not in visited:
                    queue.append((downstream_id, depth + 1))
        
        return affected
    
    def get_affected_downstream_batch(
        self,
        node_ids: List[str],
        max_depth: int = 10,
    ) -> Dict[NodeType, Set[str]]:
        """여러 노드 변경 시 영향받는 모든 downstream 노드"""
        result: Dict[NodeType, Set[str]] = {
            NodeType.SERIES: set(),
            NodeType.FEATURE: set(),
            NodeType.EVIDENCE: set(),
            NodeType.EDGE: set(),
        }
        
        for node_id in node_ids:
            affected = self.get_affected_downstream(node_id, max_depth)
            for node_type, ids in affected.items():
                result[node_type].update(ids)
        
        return result
    
    def get_upstream(self, node_id: str) -> Set[str]:
        """노드가 의존하는 upstream 노드들"""
        node = self._nodes.get(node_id)
        return node.upstream if node else set()
    
    def get_downstream(self, node_id: str) -> Set[str]:
        """노드에 의존하는 downstream 노드들"""
        node = self._nodes.get(node_id)
        return node.downstream if node else set()
    
    def get_nodes_by_type(self, node_type: NodeType) -> Set[str]:
        """타입별 노드 조회"""
        return self._by_type[node_type].copy()
    
    def build_from_registries(
        self,
        feature_specs: List[Dict],
        evidence_specs: List[Dict],
    ) -> Tuple[int, int]:
        """
        스펙 레지스트리들로부터 의존성 그래프 구축
        
        Args:
            feature_specs: Feature 스펙 리스트 (input_series 포함)
            evidence_specs: Evidence 스펙 리스트 (evidence_features 포함)
        
        Returns:
            (추가된 노드 수, 추가된 의존성 수)
        """
        node_count = 0
        dep_count = 0
        
        # Feature → Series 의존성
        for spec in feature_specs:
            feature_id = spec.get("feature_id", "")
            input_series = spec.get("input_series", [])
            
            for series_id in input_series:
                self.add_dependency(
                    feature_id, NodeType.FEATURE,
                    series_id, NodeType.SERIES
                )
                dep_count += 1
        
        # Evidence → Feature 의존성
        for spec in evidence_specs:
            spec_id = spec.get("spec_id", "")
            evidence_features = spec.get("evidence_features", [])
            
            for ef in evidence_features:
                feature_id = ef.get("feature", "")
                if feature_id:
                    self.add_dependency(
                        spec_id, NodeType.EVIDENCE,
                        feature_id, NodeType.FEATURE
                    )
                    dep_count += 1
        
        node_count = len(self._nodes)
        
        logger.info(f"Built dependency graph: {node_count} nodes, {dep_count} dependencies")
        return node_count, dep_count
    
    def add_edge_evidence_link(self, edge_id: str, evidence_spec_id: str) -> None:
        """Edge ↔ Evidence 연결"""
        self.add_dependency(
            edge_id, NodeType.EDGE,
            evidence_spec_id, NodeType.EVIDENCE
        )
    
    def get_topological_order(self) -> List[str]:
        """
        토폴로지 정렬
        
        upstream이 없는 노드부터 downstream 순으로 정렬
        증분 업데이트 시 계산 순서 결정에 사용
        """
        in_degree = {node_id: len(node.upstream) for node_id, node in self._nodes.items()}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            node = self._nodes.get(node_id)
            if node:
                for downstream_id in node.downstream:
                    in_degree[downstream_id] -= 1
                    if in_degree[downstream_id] == 0:
                        queue.append(downstream_id)
        
        return result
    
    def clear(self) -> None:
        """그래프 초기화"""
        self._nodes.clear()
        for node_set in self._by_type.values():
            node_set.clear()
    
    def get_stats(self) -> Dict:
        """통계"""
        total_deps = sum(len(n.upstream) for n in self._nodes.values())
        
        return {
            "total_nodes": len(self._nodes),
            "by_type": {t.value: len(ids) for t, ids in self._by_type.items()},
            "total_dependencies": total_deps,
        }
