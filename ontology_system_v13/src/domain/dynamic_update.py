"""
Dynamic domain update.
"""
import logging
import math
from typing import Dict, Optional
from datetime import datetime

from src.bootstrap import get_domain_kg_adapter
from src.domain.kg_adapter import DomainKGAdapter
from src.domain.models import (
    DomainCandidate,
    DynamicRelation,
    DynamicUpdateResult,
    DomainAction,
)
from src.storage.transaction_manager import Transaction

logger = logging.getLogger(__name__)


class DynamicDomainUpdate:
    """Updates dynamic relations and persists them via DomainKGAdapter."""

    def __init__(
        self,
        kg_adapter: Optional[DomainKGAdapter] = None,
        initial_conf: float = 0.5,
        conf_increase_rate: float = 0.05,
        conf_decrease_rate: float = 0.08,
        decay_rate: float = 0.98,
        decay_days: int = 30,
    ):
        self.kg_adapter = kg_adapter or get_domain_kg_adapter()
        self.initial_conf = initial_conf
        self.conf_increase_rate = conf_increase_rate
        self.conf_decrease_rate = conf_decrease_rate
        self.decay_rate = decay_rate
        self.decay_days = decay_days

    def update(
        self,
        candidate: DomainCandidate,
        tx: Optional[Transaction] = None,
    ) -> DynamicUpdateResult:
        relation = self.kg_adapter.get_relation(
            candidate.head_canonical_id,
            candidate.tail_canonical_id,
            candidate.relation_type,
        )

        if relation is None:
            return self._create_new_relation(candidate, tx)
        return self._update_existing_relation(relation, candidate, tx)

    def _create_new_relation(
        self,
        candidate: DomainCandidate,
        tx: Optional[Transaction],
    ) -> DynamicUpdateResult:
        relation = DynamicRelation(
            head_id=candidate.head_canonical_id,
            head_name=candidate.head_canonical_name,
            tail_id=candidate.tail_canonical_id,
            tail_name=candidate.tail_canonical_name,
            relation_type=candidate.relation_type,
            sign=candidate.polarity,
            domain_conf=self.initial_conf,
            evidence_count=1,
            origin=candidate.evidence_source,
            semantic_tags=[candidate.semantic_tag],
        )

        self.kg_adapter.upsert_relation(relation, tx=tx)
        logger.info(f"Created new dynamic relation: {relation.relation_id}")

        return DynamicUpdateResult(
            candidate_id=candidate.candidate_id,
            relation_id=relation.relation_id,
            action=DomainAction.CREATE_NEW,
            domain_conf=relation.domain_conf,
            evidence_count=relation.evidence_count,
            is_new=True,
        )

    def _update_existing_relation(
        self,
        relation: DynamicRelation,
        candidate: DomainCandidate,
        tx: Optional[Transaction],
    ) -> DynamicUpdateResult:
        previous_conf = relation.domain_conf
        previous_evidence = relation.evidence_count

        decayed = self._apply_decay(relation)

        if candidate.polarity == relation.sign or candidate.polarity == "unknown":
            relation = self._strengthen_relation(relation, candidate)
        else:
            relation = self._weaken_relation(relation)

        if candidate.semantic_tag not in relation.semantic_tags:
            relation.semantic_tags.append(candidate.semantic_tag)

        self.kg_adapter.upsert_relation(relation, tx=tx)
        logger.info(
            f"Updated dynamic relation: {relation.relation_id}, "
            f"conf: {previous_conf:.3f} -> {relation.domain_conf:.3f}"
        )

        return DynamicUpdateResult(
            candidate_id=candidate.candidate_id,
            relation_id=relation.relation_id,
            action=DomainAction.UPDATE_EXISTING,
            domain_conf=relation.domain_conf,
            evidence_count=relation.evidence_count,
            decayed=decayed,
            previous_conf=previous_conf,
            previous_evidence_count=previous_evidence,
        )

    def _apply_decay(self, relation: DynamicRelation) -> bool:
        now = datetime.now()
        days_elapsed = (now - relation.last_update).days

        if days_elapsed < self.decay_days:
            return False

        decay_periods = days_elapsed // self.decay_days
        decay_factor = self.decay_rate ** decay_periods

        relation.domain_conf *= decay_factor
        relation.decay_applied = True

        logger.debug(f"Applied decay to {relation.relation_id}: factor={decay_factor:.4f}")
        return True

    def _strengthen_relation(
        self,
        relation: DynamicRelation,
        candidate: DomainCandidate,
    ) -> DynamicRelation:
        relation.evidence_count += 1
        relation.last_update = datetime.now()

        increase = self.conf_increase_rate / math.sqrt(relation.evidence_count)
        relation.domain_conf = min(0.95, relation.domain_conf + increase)

        return relation

    def _weaken_relation(self, relation: DynamicRelation) -> DynamicRelation:
        relation.conflict_count += 1
        relation.last_update = datetime.now()
        relation.domain_conf = max(0.1, relation.domain_conf - self.conf_decrease_rate)
        return relation

    def get_relation(self, relation_id: str) -> Optional[DynamicRelation]:
        return self.kg_adapter.get_relation_by_id(relation_id)

    def get_relation_by_key(
        self,
        head_id: str,
        tail_id: str,
        relation_type: str,
    ) -> Optional[DynamicRelation]:
        return self.kg_adapter.get_relation(head_id, tail_id, relation_type)

    def get_all_relations(self) -> Dict[str, DynamicRelation]:
        return self.kg_adapter.get_all_relations()

    def get_relations_for_entity(self, entity_id: str) -> list:
        all_rels = self.kg_adapter.get_all_relations()
        return [
            rel for rel in all_rels.values()
            if rel.head_id == entity_id or rel.tail_id == entity_id
        ]
