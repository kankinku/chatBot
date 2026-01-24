"""Graph retrieval for domain relations."""
import logging
from typing import Optional, List, Dict, Set
from collections import deque

from src.reasoning.models import ParsedQuery, RetrievedPath, RetrievalResult
from src.domain.dynamic_update import DynamicDomainUpdate

logger = logging.getLogger(__name__)


class GraphRetrieval:
    """Domain-first graph retrieval."""

    def __init__(
        self,
        domain: Optional[DynamicDomainUpdate] = None,
        max_path_length: int = 4,
        max_paths: int = 10,
    ):
        self.domain = domain
        self.max_path_length = max_path_length
        self.max_paths = max_paths

    def set_limits(self, max_depth: Optional[int] = None, max_paths: Optional[int] = None) -> None:
        if max_depth is not None:
            self.max_path_length = max_depth
        if max_paths is not None:
            self.max_paths = max_paths

    def retrieve(self, parsed_query: ParsedQuery) -> RetrievalResult:
        direct_paths: List[RetrievedPath] = []
        indirect_paths: List[RetrievedPath] = []
        domain_count = 0
        total_edges = 0

        head = parsed_query.head_entity
        tail = parsed_query.tail_entity

        if not head or not self.domain:
            return RetrievalResult(
                query_id=parsed_query.query_id,
                direct_paths=[],
                indirect_paths=[],
            )

        all_paths = self._find_paths_bfs(
            start=head,
            end=tail,
            graph=self._build_domain_graph(),
            entity_names=parsed_query.entity_names,
            source="domain",
        )

        for path in all_paths:
            if path.path_length == 1:
                direct_paths.append(path)
            else:
                indirect_paths.append(path)
            domain_count += 1
            total_edges += len(path.edges)

        result = RetrievalResult(
            query_id=parsed_query.query_id,
            direct_paths=direct_paths,
            indirect_paths=indirect_paths[: self.max_paths],
            domain_paths_count=domain_count,
            total_edges_retrieved=total_edges,
        )

        logger.info(
            f"Retrieved: {len(direct_paths)} direct, {len(indirect_paths)} indirect, "
            f"domain={domain_count}"
        )

        return result

    def _build_domain_graph(self) -> Dict[str, List[Dict]]:
        graph: Dict[str, List[Dict]] = {}
        if not self.domain:
            return graph
        for rel in self.domain.get_all_relations().values():
            graph.setdefault(rel.head_id, []).append({
                "tail": rel.tail_id,
                "relation_id": rel.relation_id,
                "sign": rel.sign,
                "domain_conf": rel.domain_conf,
                "evidence_count": rel.evidence_count,
                "relation_type": rel.relation_type,
            })
        return graph

    def _find_paths_bfs(
        self,
        start: str,
        end: Optional[str],
        graph: Dict[str, List[Dict]],
        entity_names: Dict[str, str],
        source: str,
    ) -> List[RetrievedPath]:
        if end is None or start == end:
            return []

        paths: List[RetrievedPath] = []
        queue = deque([(start, [start], [])])
        visited_paths: Set[tuple] = set()

        while queue and len(paths) < self.max_paths:
            current, path, edges = queue.popleft()
            if len(path) > self.max_path_length:
                continue

            for edge_info in graph.get(current, []):
                next_node = edge_info["tail"]
                if next_node in path:
                    continue

                new_path = path + [next_node]
                new_edges = edges + [{
                    "relation_id": edge_info["relation_id"],
                    "head": current,
                    "tail": next_node,
                    "sign": edge_info["sign"],
                    "domain_conf": edge_info["domain_conf"],
                    "evidence_count": edge_info.get("evidence_count", 1),
                    "relation_type": edge_info.get("relation_type", "Affect"),
                    "source": source,
                }]

                if next_node == end:
                    path_key = tuple(new_path)
                    if path_key not in visited_paths:
                        visited_paths.add(path_key)
                        paths.append(RetrievedPath(
                            nodes=new_path,
                            node_names=[entity_names.get(n, n) for n in new_path],
                            edges=new_edges,
                            source=source,
                            path_length=len(new_path) - 1,
                        ))
                else:
                    queue.append((next_node, new_path, new_edges))

        return paths
