from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

@dataclass
class SystemSnapshot:
    """
    System State Snapshot
    Encapsulates all state at a specific point in time 'as_of'.
    """
    snapshot_id: str
    as_of: datetime
    created_at: datetime

    # Data Summaries
    series_values: Dict[str, float] = field(default_factory=dict)
    feature_values: Dict[str, float] = field(default_factory=dict)
    evidence_scores: Dict[str, float] = field(default_factory=dict)
    regime_state: Dict[str, Any] = field(default_factory=dict)
    edge_confidences: Dict[str, float] = field(default_factory=dict)

    # Graph Snapshot (Internal Storage)
    graph_entities: List[Dict[str, Any]] = field(default_factory=list)
    graph_relations: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_graph_view(self):
        """
        Returns a view compatible with GraphRetrieval's domain expectation.
        Encapsulates the raw list-of-dicts structure.
        """
        return _SnapshotDomainView(self)

    def to_conf_map(self) -> Dict[str, float]:
        """
        Returns a map of relation_id -> confidence.
        """
        conf_map = {}
        for rel in self.graph_relations:
            props = rel.get("props", {})
            # Construct ID if missing
            rel_id = props.get("relation_id") or f"{rel.get('src_id')}_{rel.get('rel_type')}_{rel.get('dst_id')}"
            
            # Confidence Priority: props > top-level > default
            conf = float(props.get("domain_conf", rel.get("domain_conf", 0.5)))
            conf_map[rel_id] = conf
        return conf_map


class _SnapshotDomainView:
    """
    Adapter to make SystemSnapshot look like a DomainAdapter/GraphRepository
    for GraphRetrieval consumptions.
    """
    def __init__(self, snapshot: SystemSnapshot):
        self.snapshot = snapshot
        # Pre-index for faster lookup
        self._relations_map = {}
        for rel in snapshot.graph_relations:
            src = rel.get("src_id")
            dst = rel.get("dst_id")
            rtype = rel.get("rel_type")
            key = (src, dst, rtype)
            self._relations_map[key] = rel
    
    def get_relation_by_key(self, head_id: str, tail_id: str, relation_type: str):
        rel = self._relations_map.get((head_id, tail_id, relation_type))
        if rel:
             # Return a minimal object compatible with DomainRelation/Edge
             return _SnapshotRelationProxy(rel)
        
        # Fallback: partial match if needed (e.g. if key doesn't match exactly but we want to search)
        # For now, strict match is safer for Replay.
        return None

    def get_all_relations(self) -> Dict[str, Any]:
        """
        Returns {relation_id: ProxyObject}
        """
        result = {}
        for rel in self.snapshot.graph_relations:
            proxy = _SnapshotRelationProxy(rel)
            result[proxy.relation_id] = proxy
        return result

class _SnapshotRelationProxy:
    def __init__(self, rel_dict: Dict[str, Any]):
        self._data = rel_dict
        self.props = rel_dict.get("props", {})
    
    @property
    def relation_id(self) -> str:
        return self.props.get("relation_id") or f"{self.head_id}_{self.relation_type}_{self.tail_id}"
    
    @property
    def head_id(self) -> str:
        return self._data.get("src_id", "")
        
    @property
    def tail_id(self) -> str:
        return self._data.get("dst_id", "")
        
    @property
    def relation_type(self) -> str:
        return self._data.get("rel_type", "")
        
    @property
    def sign(self) -> str:
        return self.props.get("sign", self._data.get("sign", "+"))
        
    @property
    def domain_conf(self) -> float:
        return float(self.props.get("domain_conf", self._data.get("domain_conf", 0.5)))
        
    @property
    def evidence_count(self) -> int:
        return int(self.props.get("evidence_count", 1))
        
    @property
    def pcs_score(self) -> float:
        return float(self.props.get("pcs_score", 0.0))
