"""In-memory graph repository."""
from typing import Any, Dict, List, Optional
from collections import defaultdict

from chatbot.knowledge.storage.graph_repository import GraphRepository


class InMemoryGraphRepository(GraphRepository):
    """In-memory graph store."""

    def __init__(self) -> None:
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._relations: Dict[tuple, Dict[str, Any]] = {}
        self._edges_out: Dict[str, List[tuple]] = defaultdict(list)
        self._edges_in: Dict[str, List[tuple]] = defaultdict(list)

    def upsert_entity(self, entity_id: str, labels: List[str], props: Dict[str, Any]) -> None:
        if entity_id in self._entities:
            existing = self._entities[entity_id]
            existing["labels"] = labels
            existing["props"].update(props)
        else:
            self._entities[entity_id] = {"labels": labels, "props": props}

    def upsert_relation(self, src_id: str, rel_type: str, dst_id: str, props: Dict[str, Any]) -> None:
        key = (src_id, rel_type, dst_id)
        self._relations[key] = props
        if (rel_type, dst_id) not in self._edges_out[src_id]:
            self._edges_out[src_id].append((rel_type, dst_id))
        if (rel_type, src_id) not in self._edges_in[dst_id]:
            self._edges_in[dst_id].append((rel_type, src_id))

    def get_entity(self, entity_id: str, *, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        return self._entities.get(entity_id)

    def get_relation(
        self,
        src_id: str,
        rel_type: str,
        dst_id: str,
        *,
        as_of: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        key = (src_id, rel_type, dst_id)
        props = self._relations.get(key)
        if props is None:
            return None
        return {"src_id": src_id, "rel_type": rel_type, "dst_id": dst_id, "props": props}

    def get_neighbors(
        self,
        entity_id: str,
        rel_type: Optional[str] = None,
        direction: str = "out",
        *,
        as_of: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
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
        return [
            {"entity_id": eid, "labels": data["labels"], "props": data["props"]}
            for eid, data in self._entities.items()
        ]

    def get_all_relations(self, *, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        return [
            {"src_id": src, "rel_type": rel_type, "dst_id": dst, "props": props}
            for (src, rel_type, dst), props in self._relations.items()
        ]

    def delete_entity(self, entity_id: str) -> bool:
        if entity_id not in self._entities:
            return False

        incident = [
            (src_id, rel_type, dst_id)
            for (src_id, rel_type, dst_id) in list(self._relations)
            if src_id == entity_id or dst_id == entity_id
        ]
        for src_id, rel_type, dst_id in incident:
            self.delete_relation(src_id, rel_type, dst_id)

        del self._entities[entity_id]
        self._edges_out.pop(entity_id, None)
        self._edges_in.pop(entity_id, None)
        return True

    def delete_relation(self, src_id: str, rel_type: str, dst_id: str) -> bool:
        key = (src_id, rel_type, dst_id)
        if key not in self._relations:
            return False

        del self._relations[key]

        out_edge = (rel_type, dst_id)
        if out_edge in self._edges_out.get(src_id, []):
            self._edges_out[src_id].remove(out_edge)
        if not self._edges_out.get(src_id):
            self._edges_out.pop(src_id, None)

        in_edge = (rel_type, src_id)
        if in_edge in self._edges_in.get(dst_id, []):
            self._edges_in[dst_id].remove(in_edge)
        if not self._edges_in.get(dst_id):
            self._edges_in.pop(dst_id, None)

        return True

    def clear(self) -> None:
        self._entities.clear()
        self._relations.clear()
        self._edges_out.clear()
        self._edges_in.clear()

    def count_entities(self) -> int:
        return len(self._entities)

    def count_relations(self) -> int:
        return len(self._relations)
