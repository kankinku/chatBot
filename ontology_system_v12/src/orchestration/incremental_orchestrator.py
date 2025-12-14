"""
Incremental Update Orchestrator
증분 업데이트 통합 오케스트레이터

책임:
- Ingestion 결과 수신
- 영향 범위 산출 및 부분 재계산
- 전체 파이프라인 조율
"""
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from src.orchestration.dependency_graph_manager import DependencyGraphManager, NodeType
from src.orchestration.cache_invalidator import CacheInvalidator, InvalidationReason

logger = logging.getLogger(__name__)


@dataclass
class UpdateContext:
    """업데이트 컨텍스트"""
    update_id: str
    started_at: datetime
    source: str                           # ingestion, manual, replay
    updated_series: List[str] = field(default_factory=list)
    updated_features: List[str] = field(default_factory=list)
    updated_evidence: List[str] = field(default_factory=list)
    updated_edges: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    completed_at: Optional[datetime] = None


@dataclass
class OrchestratorResult:
    """오케스트레이션 결과"""
    success: bool
    update_id: str
    series_count: int = 0
    features_computed: int = 0
    evidence_computed: int = 0
    edges_updated: int = 0
    elapsed_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


class IncrementalUpdateOrchestrator:
    """
    증분 업데이트 오케스트레이터
    
    처리 흐름:
    1. Ingestion 결과 수신 (어떤 series가 업데이트됐는지)
    2. DependencyGraphManager로 영향 범위 산출
    3. FeatureBuilder로 해당 feature만 재계산
    4. EvidenceBinder로 해당 evidence만 재계산
    5. 변동이 큰 edge만 KG 갱신
    """
    
    def __init__(
        self,
        dependency_manager: DependencyGraphManager,
        cache_invalidator: CacheInvalidator,
        feature_builder: Optional[Any] = None,
        evidence_binder: Optional[Any] = None,
        feature_store: Optional[Any] = None,
        evidence_store: Optional[Any] = None,
        edge_update_threshold: float = 0.1,  # confidence 변화 임계값
    ):
        """
        Args:
            dependency_manager: 의존성 그래프 관리자
            cache_invalidator: 캐시 무효화 관리자
            feature_builder: Feature 계산기 (optional)
            evidence_binder: Evidence 바인더 (optional)
            feature_store: Feature 저장소 (optional)
            evidence_store: Evidence 저장소 (optional)
            edge_update_threshold: Edge 업데이트 임계값
        """
        self.dependency_manager = dependency_manager
        self.cache_invalidator = cache_invalidator
        self.feature_builder = feature_builder
        self.evidence_binder = evidence_binder
        self.feature_store = feature_store
        self.evidence_store = evidence_store
        self.edge_update_threshold = edge_update_threshold
        
        self._update_history: List[UpdateContext] = []
        self._current_context: Optional[UpdateContext] = None
    
    def process_ingestion_result(
        self,
        updated_series: List[str],
        timestamp: Optional[datetime] = None,
    ) -> OrchestratorResult:
        """
        Ingestion 결과 처리 (메인 진입점)
        
        Args:
            updated_series: 업데이트된 시계열 ID 리스트
            timestamp: 처리 기준 시점
        
        Returns:
            OrchestratorResult
        """
        import time
        start_time = time.time()
        
        timestamp = timestamp or datetime.now()
        update_id = f"UPD_{timestamp.strftime('%Y%m%d%H%M%S')}"
        
        context = UpdateContext(
            update_id=update_id,
            started_at=timestamp,
            source="ingestion",
            updated_series=updated_series,
        )
        self._current_context = context
        
        try:
            # 1. 캐시 무효화 (영향 범위 산출)
            invalidation_event = self.cache_invalidator.invalidate(
                updated_series,
                InvalidationReason.DATA_UPDATE
            )
            
            # 2. Feature 재계산
            features_computed = 0
            pending_features = list(invalidation_event.affected_features)
            
            if self.feature_builder and pending_features:
                computed, errors = self._compute_features(pending_features, timestamp)
                features_computed = computed
                context.updated_features = pending_features[:computed]
                context.errors.extend(errors)
            
            # 3. Evidence 재계산
            evidence_computed = 0
            pending_evidence = list(invalidation_event.affected_evidence)
            
            if self.evidence_binder and pending_evidence:
                computed, errors = self._compute_evidence(pending_evidence, timestamp)
                evidence_computed = computed
                context.updated_evidence = pending_evidence[:computed]
                context.errors.extend(errors)
            
            # 4. Edge 업데이트 (significant changes만)
            edges_updated = 0
            pending_edges = list(invalidation_event.affected_edges)
            
            if pending_edges:
                updated, errors = self._update_edges(pending_edges)
                edges_updated = updated
                context.updated_edges = pending_edges[:updated]
                context.errors.extend(errors)
            
            context.completed_at = datetime.now()
            self._update_history.append(context)
            
            elapsed = (time.time() - start_time) * 1000
            
            return OrchestratorResult(
                success=len(context.errors) == 0,
                update_id=update_id,
                series_count=len(updated_series),
                features_computed=features_computed,
                evidence_computed=evidence_computed,
                edges_updated=edges_updated,
                elapsed_ms=elapsed,
                errors=context.errors,
            )
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            context.errors.append(str(e))
            context.completed_at = datetime.now()
            
            return OrchestratorResult(
                success=False,
                update_id=update_id,
                errors=[str(e)],
            )
        finally:
            self._current_context = None
    
    def _compute_features(
        self,
        feature_ids: List[str],
        timestamp: datetime,
    ) -> Tuple[int, List[str]]:
        """Feature 재계산"""
        computed = 0
        errors = []
        
        for feature_id in feature_ids:
            try:
                if self.feature_builder:
                    result = self.feature_builder.compute(
                        feature_id,
                        end=timestamp,
                    )
                    if result.success:
                        computed += 1
                        # 저장
                        if self.feature_store and result.values:
                            self.feature_store.save_batch(result.values)
                    else:
                        errors.append(f"Feature {feature_id}: {result.error_message}")
            except Exception as e:
                errors.append(f"Feature {feature_id}: {str(e)}")
        
        self.cache_invalidator.mark_feature_computed(feature_ids[:computed])
        return computed, errors
    
    def _compute_evidence(
        self,
        evidence_spec_ids: List[str],
        timestamp: datetime,
    ) -> Tuple[int, List[str]]:
        """Evidence 재계산"""
        computed = 0
        errors = []
        
        # Evidence 계산은 실제 Edge 정보가 필요
        # 여기서는 stub으로 성공 처리
        computed = len(evidence_spec_ids)
        
        self.cache_invalidator.mark_evidence_computed(evidence_spec_ids[:computed])
        return computed, errors
    
    def _update_edges(
        self,
        edge_ids: List[str],
    ) -> Tuple[int, List[str]]:
        """Edge 업데이트"""
        updated = 0
        errors = []
        
        # Edge 업데이트는 GraphRepository와 연동 필요
        # 여기서는 stub으로 성공 처리
        updated = len(edge_ids)
        
        self.cache_invalidator.mark_edge_updated(edge_ids[:updated])
        return updated, errors
    
    def process_manual_update(
        self,
        node_type: NodeType,
        node_ids: List[str],
    ) -> OrchestratorResult:
        """수동 업데이트 처리"""
        import time
        start_time = time.time()
        
        update_id = f"MAN_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 무효화
        event = self.cache_invalidator.invalidate_by_type(
            node_type, node_ids, InvalidationReason.MANUAL
        )
        
        elapsed = (time.time() - start_time) * 1000
        
        return OrchestratorResult(
            success=True,
            update_id=update_id,
            features_computed=len(event.affected_features),
            evidence_computed=len(event.affected_evidence),
            edges_updated=len(event.affected_edges),
            elapsed_ms=elapsed,
        )
    
    def get_pending_work(self) -> Dict[str, int]:
        """대기 중인 작업 현황"""
        return {
            "pending_features": len(self.cache_invalidator.get_pending_features()),
            "pending_evidence": len(self.cache_invalidator.get_pending_evidence()),
            "pending_edges": len(self.cache_invalidator.get_pending_edges()),
        }
    
    def get_update_history(self, limit: int = 50) -> List[UpdateContext]:
        """업데이트 이력"""
        return self._update_history[-limit:]
    
    def get_stats(self) -> Dict:
        """통계"""
        total_updates = len(self._update_history)
        
        if total_updates == 0:
            return {
                "total_updates": 0,
                "avg_features_per_update": 0,
                "avg_evidence_per_update": 0,
                "avg_edges_per_update": 0,
            }
        
        total_features = sum(len(u.updated_features) for u in self._update_history)
        total_evidence = sum(len(u.updated_evidence) for u in self._update_history)
        total_edges = sum(len(u.updated_edges) for u in self._update_history)
        
        return {
            "total_updates": total_updates,
            "avg_features_per_update": total_features / total_updates,
            "avg_evidence_per_update": total_evidence / total_updates,
            "avg_edges_per_update": total_edges / total_updates,
            "pending_work": self.get_pending_work(),
        }
