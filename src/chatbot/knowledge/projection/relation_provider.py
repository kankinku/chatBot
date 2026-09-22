"""Read-only relation-provider adapters for graph reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from chatbot.knowledge.domain.dynamic_update import DynamicDomainUpdate

from .models import ScenarioProjection


class RelationLike(Protocol):
    relation_id: str
    head_id: str
    tail_id: str
    relation_type: str
    sign: str
    domain_conf: float
    evidence_count: int


class RelationProvider(Protocol):
    def get_all_relations(self) -> Mapping[str, RelationLike]: ...


class DynamicRelationProvider:
    def __init__(self, domain: DynamicDomainUpdate):
        self.domain = domain

    def get_all_relations(self):
        return self.domain.get_all_relations()


@dataclass(frozen=True)
class ReasoningRelation:
    relation_id: str
    head_id: str
    tail_id: str
    relation_type: str
    sign: str
    domain_conf: float
    evidence_count: int


class ProjectedRelationProvider:
    """Expose active projected relations through the existing reasoning read shape."""

    def __init__(self, projection: ScenarioProjection):
        self.projection = projection

    def get_all_relations(self) -> dict[str, ReasoningRelation]:
        result: dict[str, ReasoningRelation] = {}
        for relation in self.projection.relations:
            if (
                not relation.active
                or relation.projected_weight <= 0.0
                or relation.projected_sign not in {"+", "-"}
            ):
                continue
            result[relation.relation_id] = ReasoningRelation(
                relation_id=relation.relation_id,
                head_id=relation.head_id,
                tail_id=relation.tail_id,
                relation_type=relation.relation_type,
                sign=relation.projected_sign,
                domain_conf=relation.projected_weight,
                # projected_weight already contains the canonical evidence
                # score. Passing the original evidence count would make the
                # existing EdgeWeightFusion apply its evidence bonus again.
                evidence_count=0,
            )
        return result
