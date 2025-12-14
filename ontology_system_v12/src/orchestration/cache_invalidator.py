"""
Cache Invalidator
업데이트된 입력에 따라 캐시/계산 결과 무효화

책임:
- 변경된 노드 기반 캐시 무효화
- 무효화 이벤트 발행
- 부분 재계산 트리거
"""
import logging
from typing import Dict, List, Set, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.orchestration.dependency_graph_manager import DependencyGraphManager, NodeType

logger = logging.getLogger(__name__)


class InvalidationReason(str, Enum):
    """무효화 사유"""
    DATA_UPDATE = "data_update"       # 새 데이터
    DATA_REVISION = "data_revision"   # 데이터 수정
    SPEC_CHANGE = "spec_change"       # 스펙 변경
    MANUAL = "manual"                 # 수동 무효화


@dataclass
class InvalidationEvent:
    """무효화 이벤트"""
    event_id: str
    timestamp: datetime
    reason: InvalidationReason
    source_nodes: List[str]           # 원인 노드들
    affected_features: Set[str]       # 영향받는 Features
    affected_evidence: Set[str]       # 영향받는 Evidence
    affected_edges: Set[str]          # 영향받는 Edges


class CacheInvalidator:
    """
    캐시 무효화 관리자
    
    의존성 그래프를 따라가며 영향받는 캐시/계산 결과를 무효화합니다.
    """
    
    def __init__(
        self,
        dependency_manager: DependencyGraphManager,
    ):
        """
        Args:
            dependency_manager: 의존성 그래프 관리자
        """
        self.dependency_manager = dependency_manager
        
        # 무효화 이벤트 리스너들
        self._listeners: List[Callable[[InvalidationEvent], None]] = []
        
        # 무효화 이력
        self._invalidation_history: List[InvalidationEvent] = []
        
        # 현재 무효화된 노드들 (재계산 대기 중)
        self._invalidated_features: Set[str] = set()
        self._invalidated_evidence: Set[str] = set()
        self._invalidated_edges: Set[str] = set()
    
    def invalidate(
        self,
        source_nodes: List[str],
        reason: InvalidationReason = InvalidationReason.DATA_UPDATE,
    ) -> InvalidationEvent:
        """
        노드 변경에 따른 영향 범위 무효화
        
        Args:
            source_nodes: 변경된 노드 ID 리스트 (보통 series)
            reason: 무효화 사유
        
        Returns:
            InvalidationEvent
        """
        # 영향받는 노드 산출
        affected = self.dependency_manager.get_affected_downstream_batch(source_nodes)
        
        # 무효화 상태 업데이트
        self._invalidated_features.update(affected[NodeType.FEATURE])
        self._invalidated_evidence.update(affected[NodeType.EVIDENCE])
        self._invalidated_edges.update(affected[NodeType.EDGE])
        
        # 이벤트 생성
        event = InvalidationEvent(
            event_id=f"INV_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self._invalidation_history)}",
            timestamp=datetime.now(),
            reason=reason,
            source_nodes=source_nodes,
            affected_features=affected[NodeType.FEATURE],
            affected_evidence=affected[NodeType.EVIDENCE],
            affected_edges=affected[NodeType.EDGE],
        )
        
        self._invalidation_history.append(event)
        
        # 리스너들에게 통지
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Listener error: {e}")
        
        logger.info(
            f"Invalidated: {len(affected[NodeType.FEATURE])} features, "
            f"{len(affected[NodeType.EVIDENCE])} evidence, "
            f"{len(affected[NodeType.EDGE])} edges"
        )
        
        return event
    
    def invalidate_by_type(
        self,
        node_type: NodeType,
        node_ids: List[str],
        reason: InvalidationReason = InvalidationReason.SPEC_CHANGE,
    ) -> InvalidationEvent:
        """특정 타입 노드 직접 무효화"""
        if node_type == NodeType.FEATURE:
            self._invalidated_features.update(node_ids)
        elif node_type == NodeType.EVIDENCE:
            self._invalidated_evidence.update(node_ids)
        elif node_type == NodeType.EDGE:
            self._invalidated_edges.update(node_ids)
        
        # 하위 영향도 반영
        affected = self.dependency_manager.get_affected_downstream_batch(node_ids)
        self._invalidated_features.update(affected[NodeType.FEATURE])
        self._invalidated_evidence.update(affected[NodeType.EVIDENCE])
        self._invalidated_edges.update(affected[NodeType.EDGE])
        
        event = InvalidationEvent(
            event_id=f"INV_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self._invalidation_history)}",
            timestamp=datetime.now(),
            reason=reason,
            source_nodes=node_ids,
            affected_features=self._invalidated_features.copy(),
            affected_evidence=self._invalidated_evidence.copy(),
            affected_edges=self._invalidated_edges.copy(),
        )
        
        self._invalidation_history.append(event)
        return event
    
    def get_pending_features(self) -> Set[str]:
        """재계산 대기 중인 Features"""
        return self._invalidated_features.copy()
    
    def get_pending_evidence(self) -> Set[str]:
        """재계산 대기 중인 Evidence"""
        return self._invalidated_evidence.copy()
    
    def get_pending_edges(self) -> Set[str]:
        """업데이트 대기 중인 Edges"""
        return self._invalidated_edges.copy()
    
    def mark_feature_computed(self, feature_ids: List[str]) -> None:
        """Feature 재계산 완료 표시"""
        for fid in feature_ids:
            self._invalidated_features.discard(fid)
    
    def mark_evidence_computed(self, evidence_ids: List[str]) -> None:
        """Evidence 재계산 완료 표시"""
        for eid in evidence_ids:
            self._invalidated_evidence.discard(eid)
    
    def mark_edge_updated(self, edge_ids: List[str]) -> None:
        """Edge 업데이트 완료 표시"""
        for eid in edge_ids:
            self._invalidated_edges.discard(eid)
    
    def add_listener(self, listener: Callable[[InvalidationEvent], None]) -> None:
        """무효화 이벤트 리스너 등록"""
        self._listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[InvalidationEvent], None]) -> None:
        """리스너 제거"""
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    def get_invalidation_history(
        self,
        limit: int = 100,
    ) -> List[InvalidationEvent]:
        """무효화 이력 조회"""
        return self._invalidation_history[-limit:]
    
    def clear_pending(self) -> tuple:
        """
        대기 중인 무효화 상태 초기화
        
        Returns:
            (features 수, evidence 수, edges 수)
        """
        counts = (
            len(self._invalidated_features),
            len(self._invalidated_evidence),
            len(self._invalidated_edges),
        )
        self._invalidated_features.clear()
        self._invalidated_evidence.clear()
        self._invalidated_edges.clear()
        return counts
    
    def get_stats(self) -> Dict:
        """통계"""
        return {
            "pending_features": len(self._invalidated_features),
            "pending_evidence": len(self._invalidated_evidence),
            "pending_edges": len(self._invalidated_edges),
            "total_invalidation_events": len(self._invalidation_history),
            "listeners_count": len(self._listeners),
        }
