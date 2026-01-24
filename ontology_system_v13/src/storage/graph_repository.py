"""
Graph Repository Interface
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class GraphRepository(ABC):
    """KnowledgeGraph 저장소 추상 인터페이스"""
    
    @abstractmethod
    def upsert_entity(
        self,
        entity_id: str,
        labels: List[str],
        props: Dict[str, Any],
    ) -> None:
        ...
    
    @abstractmethod
    def upsert_relation(
        self,
        src_id: str,
        rel_type: str,
        dst_id: str,
        props: Dict[str, Any],
    ) -> None:
        ...
    
    @abstractmethod
    def get_entity(self, entity_id: str, *, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        ...
    
    @abstractmethod
    def get_relation(
        self,
        src_id: str,
        rel_type: str,
        dst_id: str,
        *,
        as_of: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        ...
    
    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        rel_type: Optional[str] = None,
        direction: str = "out",
        *,
        as_of: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        ...
    
    @abstractmethod
    def get_all_entities(self, *, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        ...
    
    @abstractmethod
    def get_all_relations(self, *, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        ...
    
    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        ...
    
    @abstractmethod
    def delete_relation(
        self,
        src_id: str,
        rel_type: str,
        dst_id: str,
    ) -> bool:
        ...
    
    @abstractmethod
    def clear(self) -> None:
        ...
    
    @abstractmethod
    def count_entities(self) -> int:
        ...
    
    @abstractmethod
    def count_relations(self) -> int:
        ...
