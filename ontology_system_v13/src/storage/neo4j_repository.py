"""Neo4j graph repository."""
import logging
from typing import Any, Dict, List, Optional

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

from src.storage.graph_repository import GraphRepository

logger = logging.getLogger(__name__)


class Neo4jGraphRepository(GraphRepository):
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        if GraphDatabase is None:
            raise ImportError("neo4j driver is not installed")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def upsert_entity(self, entity_id: str, labels: List[str], props: Dict[str, Any]) -> None:
        with self.driver.session(database=self.database) as session:
            labels_str = ":".join(labels)
            session.run(
                f"MERGE (n:{labels_str} {{id: }}) SET n += ",
                id=entity_id,
                props=props,
            )

    def upsert_relation(self, src_id: str, rel_type: str, dst_id: str, props: Dict[str, Any]) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                f"""
                MATCH (s {{id: }}), (d {{id: }})
                MERGE (s)-[r:{rel_type}]->(d)
                SET r += 
                """,
                src_id=src_id,
                dst_id=dst_id,
                props=props,
            )

    def get_entity(self, entity_id: str, *, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            res = session.run(
                "MATCH (n {id:}) RETURN labels(n) AS labels, properties(n) AS props",
                id=entity_id,
            )
            rec = res.single()
            if not rec:
                return None
            return {"labels": rec["labels"], "props": rec["props"]}

    def get_relation(
        self,
        src_id: str,
        rel_type: str,
        dst_id: str,
        *,
        as_of: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            res = session.run(
                """
                MATCH (s {id:})-[r:%s]->(d {id:})
                RETURN properties(r) AS props
                """ % rel_type,
                src=src_id,
                dst=dst_id,
            )
            rec = res.single()
            if not rec:
                return None
            return {"src_id": src_id, "rel_type": rel_type, "dst_id": dst_id, "props": rec["props"]}

    def get_neighbors(
        self,
        entity_id: str,
        rel_type: Optional[str] = None,
        direction: str = "out",
        *,
        as_of: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        dir_arrow = "->" if direction == "out" else "<-"
        rel_filter = f":{rel_type}" if rel_type else ""
        query = (
            "MATCH (s {id:})-[r" + rel_filter + "]" + dir_arrow +
            "(t) RETURN type(r) AS rel_type, properties(r) AS props, t.id AS target_id"
        )
        results = []
        with self.driver.session(database=self.database) as session:
            res = session.run(query, id=entity_id)
            for rec in res:
                entry = {"rel_type": rec["rel_type"], "props": rec["props"]}
                if direction == "out":
                    entry["dst_id"] = rec["target_id"]
                else:
                    entry["src_id"] = rec["target_id"]
                results.append(entry)
        return results

    def get_all_entities(self, *, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            res = session.run("MATCH (n) RETURN n.id AS id, labels(n) AS labels, properties(n) AS props")
            return [{"entity_id": r["id"], "labels": r["labels"], "props": r["props"]} for r in res]

    def get_all_relations(self, *, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            res = session.run(
                "MATCH (s)-[r]->(d) RETURN s.id AS src_id, type(r) AS rel_type, d.id AS dst_id, properties(r) AS props"
            )
            return [
                {"src_id": r["src_id"], "rel_type": r["rel_type"], "dst_id": r["dst_id"], "props": r["props"]}
                for r in res
            ]

    def delete_entity(self, entity_id: str) -> bool:
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n {id:}) DETACH DELETE n", id=entity_id)
        return True

    def delete_relation(self, src_id: str, rel_type: str, dst_id: str) -> bool:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MATCH (s {id:})-[r:%s]->(d {id:}) DELETE r" % rel_type,
                src=src_id,
                dst=dst_id,
            )
        return True

    def clear(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def count_entities(self) -> int:
        with self.driver.session(database=self.database) as session:
            res = session.run("MATCH (n) RETURN count(n) AS cnt")
            return res.single()["cnt"]

    def count_relations(self) -> int:
        with self.driver.session(database=self.database) as session:
            res = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            return res.single()["cnt"]
