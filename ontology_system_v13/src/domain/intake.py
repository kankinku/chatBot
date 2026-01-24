"""
Domain candidate intake module.
"""
import logging
from typing import Optional, List
from datetime import datetime

from src.shared.models import RawEdge, ResolvedEntity
from src.validation.models import ValidationResult, ValidationDestination
from src.domain.models import DomainCandidate

logger = logging.getLogger(__name__)


class DomainCandidateIntake:
    """Convert validated edges into domain candidates."""

    def process(
        self,
        edge: RawEdge,
        validation_result: ValidationResult,
        resolved_entities: List[ResolvedEntity],
    ) -> Optional[DomainCandidate]:
        if validation_result.destination != ValidationDestination.DOMAIN_CANDIDATE:
            logger.debug(f"Edge {edge.raw_edge_id} is not a domain candidate")
            return None

        entity_map = {e.entity_id: e for e in resolved_entities}
        head_entity = entity_map.get(edge.head_entity_id)
        tail_entity = entity_map.get(edge.tail_entity_id)

        if not head_entity or not tail_entity:
            logger.warning(f"Missing entities for edge {edge.raw_edge_id}")
            return None

        if not self._is_domain_relevant(edge):
            logger.info(f"Edge {edge.raw_edge_id} is not domain relevant")
            return None

        polarity = self._normalize_polarity(edge, validation_result)

        semantic_tag = "unknown"
        if validation_result.semantic_result:
            semantic_tag = validation_result.semantic_result.semantic_tag.value

        candidate = DomainCandidate(
            raw_edge_id=edge.raw_edge_id,
            head_canonical_id=head_entity.canonical_id or head_entity.entity_id,
            head_canonical_name=head_entity.canonical_name or edge.head_canonical_name or "",
            tail_canonical_id=tail_entity.canonical_id or tail_entity.entity_id,
            tail_canonical_name=tail_entity.canonical_name or edge.tail_canonical_name or "",
            relation_type=edge.relation_type,
            polarity=polarity,
            semantic_tag=semantic_tag,
            combined_conf=validation_result.combined_conf,
            student_conf=edge.student_conf or 0.0,
            timestamp=datetime.now(),
            freq_count=1,
            evidence_source="student",
            fragment_text=edge.fragment_text,
        )

        logger.info(f"Created domain candidate: {candidate.candidate_id}")
        return candidate

    def _is_domain_relevant(self, edge: RawEdge) -> bool:
        """Lightweight relevance check."""
        text = edge.fragment_text or ""
        if len(text.strip()) < 10:
            return False
        return True

    def _normalize_polarity(
        self,
        edge: RawEdge,
        validation_result: ValidationResult,
    ) -> str:
        """Normalize polarity using sign validation if available."""
        if validation_result.sign_result:
            return validation_result.sign_result.polarity_final

        polarity = edge.polarity_guess
        if hasattr(polarity, "value"):
            polarity = polarity.value

        if polarity in ["+", "-", "neutral"]:
            return polarity

        return "unknown"
