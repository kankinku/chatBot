from dataclasses import dataclass, field
from typing import Dict, List, Set, Any
import hashlib

@dataclass
class TopologyDigest:
    node_count: int
    edge_count: int
    top_hubs_hash: str  # Hash of sorted top-k node IDs by degree
    relation_type_counts: Dict[str, int]
    conflict_edge_count: int = 0
    drift_edge_count: int = 0

def digest(graph_data: Any) -> TopologyDigest:
    """
    Computes topology digest from a graph-like object (Snapshot or adapter).
    Supports:
    - Dict-based graph (snapshot.graph_relations)
    - Adapter-like object with get_all_relations()
    """
    relations = []
    
    # Identify input type
    if isinstance(graph_data, list): # List of dicts (snapshot.graph_relations)
        relations = graph_data
    elif hasattr(graph_data, "get_all_relations"):
        relations_dict = graph_data.get_all_relations()
        if isinstance(relations_dict, dict):
            relations = [{"src_id": v.head_id, "rel_type": v.relation_type, "dst_id": v.tail_id} for v in relations_dict.values()]
        elif isinstance(relations_dict, list):
             relations = relations_dict
    
    edge_count = len(relations)
    nodes = set()
    degree_map: Dict[str, int] = {}
    rel_counts: Dict[str, int] = {}
    
    for r in relations:
        src = r.get("src_id") or r.get("head_id")
        dst = r.get("dst_id") or r.get("tail_id")
        rtype = r.get("rel_type") or r.get("relation_type")
        
        if src: nodes.add(src)
        if dst: nodes.add(dst)
        
        if src: degree_map[src] = degree_map.get(src, 0) + 1
        if dst: degree_map[dst] = degree_map.get(dst, 0) + 1
        
        if rtype:
            rel_counts[rtype] = rel_counts.get(rtype, 0) + 1
            
    node_count = len(nodes)
    
    # Top 10 Hubs Hash
    top_hubs = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)[:10]
    hub_ids = sorted([h[0] for h in top_hubs])
    hubs_str = ",".join(hub_ids)
    hubs_hash = hashlib.md5(hubs_str.encode()).hexdigest()
    
    return TopologyDigest(
        node_count=node_count,
        edge_count=edge_count,
        top_hubs_hash=hubs_hash,
        relation_type_counts=rel_counts
    )

def diff(prev: TopologyDigest, curr: TopologyDigest) -> List[Any]:
    from src.scenario.contracts import TopologyDeltaItem
    
    changes = []
    
    if prev.node_count != curr.node_count:
        item = TopologyDeltaItem(
            change_type="NODE_COUNT_CHANGED",
            target_id="GRAPH",
            details=f"Node count {prev.node_count} -> {curr.node_count}"
        )
        changes.append(item)
        
    if prev.edge_count != curr.edge_count:
        item = TopologyDeltaItem(
            change_type="EDGE_COUNT_CHANGED",
            target_id="GRAPH",
            details=f"Edge count {prev.edge_count} -> {curr.edge_count}"
        )
        changes.append(item)
        
    if prev.top_hubs_hash != curr.top_hubs_hash:
        item = TopologyDeltaItem(
            change_type="HUB_STRUCTURE_CHANGED",
            target_id="GRAPH",
            details="Top K hubs composition changed"
        )
        changes.append(item)
        
    return changes
