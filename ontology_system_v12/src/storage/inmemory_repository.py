"""
In-Memory Graph Repository with as_of validation.
"""
from typing import Any, Dict, List, Optional
from collections import defaultdict

from src.storage.graph_repository import GraphRepository
from src.common.asof_context import validate_access


class InMemoryGraphRepository(GraphRepository):
    """In-Memory graph store."""
    
    def __init__(self) -> None:
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._relations: Dict[tuple, Dict[str, Any]] = {}
        self._edges_out: Dict[str, List[tuple]] = defaultdict(list)
        self._edges_in: Dict[str, List[tuple]] = defaultdict(list)
    
    def upsert_entity(
        self,
        entity_id: str,
        labels: List[str],
        props: Dict[str, Any],
    ) -> None:
        if entity_id in self._entities:
            existing = self._entities[entity_id]
            existing["labels"] = labels
            existing["props"].update(props)
        else:
            self._entities[entity_id] = {
                "labels": labels,
                "props": props,
            }
    
    def upsert_relation(
        self,
        src_id: str,
        rel_type: str,
        dst_id: str,
        props: Dict[str, Any],
    ) -> None:
        key = (src_id, rel_type, dst_id)
        self._relations[key] = props
        if (rel_type, dst_id) not in self._edges_out[src_id]:
            self._edges_out[src_id].append((rel_type, dst_id))
        if (rel_type, src_id) not in self._edges_in[dst_id]:
            self._edges_in[dst_id].append((rel_type, src_id))
    
    def get_entity(self, entity_id: str, *, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        validate_access(as_of)
        return self._entities.get(entity_id)
    
    def get_relation(self, src_id: str, rel_type: str, dst_id: str, *, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        validate_access(as_of)
        key = (src_id, rel_type, dst_id)
        props = self._relations.get(key)
        if props is None:
            return None
        return {
            "src_id": src_id,
            "rel_type": rel_type,
            "dst_id": dst_id,
            "props": props,
        }
    
    def get_neighbors(
        self,
        entity_id: str,
        rel_type: Optional[str] = None,
        direction: str = "out",
        *,
        as_of: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        validate_access(as_of)
        result = []
        if direction in ["out", "both"]:
            for r_type, dst in self._edges_out.get(entity_id, []):
                if rel_type and r_type != rel_type:
                    continue
                key = (entity_id, r_type, dst)
                result.append({"rel_type": r_type, "dst_id": dst, "props": self._relations.get(key, {})})
        if direction in ["in", "both"]:
            for r_type, src in self._edges_in.get(entity_id, []):
                if rel_type and r_type != rel_type:
                    continue
                key = (src, r_type, entity_id)
                result.append({"rel_type": r_type, "src_id": src, "props": self._relations.get(key, {})})
        return result
    
    def get_all_entities(self, *, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        validate_access(as_of)
        return [
            {"entity_id": eid, "labels": data["labels"], "props": data["props"]}
            for eid, data in self._entities.items()
        ]
    
    def get_all_relations(self, *, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        validate_access(as_of)
        return [
            {"src_id": src, "rel_type": rel_type, "dst_id": dst, "props": props}
            for (src, rel_type, dst), props in self._relations.items()
        ]
    
    def delete_entity(self, entity_id: str) -> bool:
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False
    
    def delete_relation(self, src_id: str, rel_type: str, dst_id: str) -> bool:
        key = (src_id, rel_type, dst_id)
        if key in self._relations:
            del self._relations[key]
            return True
        return False
    
    def clear(self) -> None:
        self._entities.clear()
        self._relations.clear()
        self._edges_out.clear()
        self._edges_in.clear()
    
    def count_entities(self) -> int:
        return len(self._entities)
    
    def count_relations(self) -> int:
        return len(self._relations)
