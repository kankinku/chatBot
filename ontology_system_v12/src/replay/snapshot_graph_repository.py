"""
GraphRepository implementation that serves data from a SystemSnapshot.
"""
from typing import Any, Dict, List, Optional

from src.storage.graph_repository import GraphRepository
from src.replay.snapshot_manager import SystemSnapshot


class SnapshotGraphRepository(GraphRepository):
    """
    읽기 전용 스냅샷 그래프 저장소.
    """

    def __init__(self, snapshot: SystemSnapshot):
        self.snapshot = snapshot
        self._entities = {e.get("id") or e.get("entity_id"): e for e in snapshot.graph_entities}
        self._relations = snapshot.graph_relations

    # Mutations are no-op (read only)
    def upsert_entity(self, entity_id: str, labels: List[str], props: Dict[str, Any]) -> None:
        return

    def upsert_relation(self, src_id: str, rel_type: str, dst_id: str, props: Dict[str, Any]) -> None:
        return

    def get_entity(self, entity_id: str, *, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        return self._entities.get(entity_id)

    def get_relation(self, src_id: str, rel_type: str, dst_id: str, *, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        for rel in self._relations:
            if rel.get("src_id") == src_id and rel.get("dst_id") == dst_id and rel.get("rel_type") == rel_type:
                return rel
        return None

    def get_neighbors(self, entity_id: str, rel_type: Optional[str] = None, direction: str = "out", *, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        result = []
        for rel in self._relations:
            if rel_type and rel.get("rel_type") != rel_type:
                continue
            if direction in ("out", "both") and rel.get("src_id") == entity_id:
                result.append({"rel_type": rel.get("rel_type"), "dst_id": rel.get("dst_id"), "props": rel.get("props", {})})
            if direction in ("in", "both") and rel.get("dst_id") == entity_id:
                result.append({"rel_type": rel.get("rel_type"), "src_id": rel.get("src_id"), "props": rel.get("props", {})})
        return result

    def get_all_entities(self, *, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        return list(self._entities.values())

    def get_all_relations(self, *, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        return list(self._relations)

    def delete_entity(self, entity_id: str) -> bool:
        return False

    def delete_relation(self, src_id: str, rel_type: str, dst_id: str) -> bool:
        return False

    def clear(self) -> None:
        return

    def count_entities(self) -> int:
        return len(self._entities)

    def count_relations(self) -> int:
        return len(self._relations)
