"""
Domain pipeline (intake -> static guard -> dynamic update).
"""
import logging
from typing import List, Optional, Dict

from src.bootstrap import get_transaction_manager
from src.storage.transaction_manager import Transaction
from src.shared.models import RawEdge, ResolvedEntity
from src.validation.models import ValidationResult
from src.domain.models import DomainProcessResult, DomainAction
from src.domain.intake import DomainCandidateIntake
from src.domain.static_guard import StaticDomainGuard
from src.domain.dynamic_update import DynamicDomainUpdate

logger = logging.getLogger(__name__)


class DomainPipeline:
    """Domain processing pipeline with transaction support."""

    def __init__(self):
        self.intake = DomainCandidateIntake()
        self.static_guard = StaticDomainGuard()
        self.dynamic_update = DynamicDomainUpdate()

        self.tx_manager = get_transaction_manager()

        self._stats = {
            "total": 0,
            "domain_accepted": 0,
            "logged": 0,
            "static_matched": 0,
            "static_conflict": 0,
            "new_relations": 0,
            "updated_relations": 0,
        }

    def process(
        self,
        edge: RawEdge,
        validation_result: ValidationResult,
        resolved_entities: List[ResolvedEntity],
        tx: Optional[Transaction] = None,
    ) -> DomainProcessResult:
        self._stats["total"] += 1

        candidate = self.intake.process(edge, validation_result, resolved_entities)
        if candidate is None:
            self._stats["logged"] += 1
            return DomainProcessResult(
                candidate_id="",
                raw_edge_id=edge.raw_edge_id,
                final_destination="log",
            )

        static_result = self.static_guard.check(candidate)
        if static_result.static_conflict:
            self._stats["static_conflict"] += 1
            return DomainProcessResult(
                candidate_id=candidate.candidate_id,
                raw_edge_id=edge.raw_edge_id,
                final_destination="log",
                intake_result=candidate,
                static_result=static_result,
            )

        if static_result.action == DomainAction.STRENGTHEN_STATIC:
            self._stats["static_matched"] += 1

        dynamic_result = self.dynamic_update.update(candidate, tx=tx)

        if dynamic_result.is_new:
            self._stats["new_relations"] += 1
        else:
            self._stats["updated_relations"] += 1

        self._stats["domain_accepted"] += 1

        return DomainProcessResult(
            candidate_id=candidate.candidate_id,
            raw_edge_id=edge.raw_edge_id,
            final_destination="domain",
            intake_result=candidate,
            static_result=static_result,
            dynamic_result=dynamic_result,
            domain_relation_id=dynamic_result.relation_id,
        )

    def process_batch(
        self,
        edges: List[RawEdge],
        validation_results: Dict[str, ValidationResult],
        resolved_entities: List[ResolvedEntity],
    ) -> List[DomainProcessResult]:
        results = []
        with self.tx_manager.transaction() as tx:
            for edge in edges:
                v_result = validation_results.get(edge.raw_edge_id)
                if v_result and v_result.validation_passed:
                    result = self.process(edge, v_result, resolved_entities, tx=tx)
                    results.append(result)

        logger.info(f"Domain batch complete: {self._stats}")
        return results

    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()

    def get_dynamic_domain(self) -> DynamicDomainUpdate:
        return self.dynamic_update

    def reset_stats(self) -> None:
        for key in self._stats:
            self._stats[key] = 0
