"""
Static domain guard for known fixed rules.
"""
import logging
from typing import Dict, Any, Optional, List

from src.domain.models import DomainCandidate, StaticGuardResult, DomainAction
from config.settings import get_settings

logger = logging.getLogger(__name__)


class StaticDomainGuard:
    """Check candidates against static domain rules."""

    def __init__(self):
        self.settings = get_settings()
        self._static_rules = self._load_static_rules()
        self._rules_map = self._build_rules_map()

    def _load_static_rules(self) -> List[Dict[str, Any]]:
        """Load static rules from config."""
        try:
            data = self.settings.load_yaml_config("static_domain")
            return data.get("static_rules", [])
        except FileNotFoundError:
            logger.warning("Static domain config not found")
            return []

    def _build_rules_map(self) -> Dict[tuple, Dict[str, Any]]:
        """Index rules by (head, tail)."""
        rules_map = {}
        for rule in self._static_rules:
            head = rule.get("head")
            tail = rule.get("tail")
            if head and tail:
                rules_map[(head, tail)] = rule
        return rules_map

    def check(self, candidate: DomainCandidate) -> StaticGuardResult:
        """Check candidate against static rules."""
        head_id = candidate.head_canonical_id
        tail_id = candidate.tail_canonical_id

        static_rule = self._rules_map.get((head_id, tail_id))

        if static_rule is None:
            return StaticGuardResult(
                candidate_id=candidate.candidate_id,
                static_pass=True,
                static_conflict=False,
                action=DomainAction.CREATE_NEW,
            )

        static_polarity = static_rule.get("polarity")
        static_relation = static_rule.get("relation")
        static_certainty = static_rule.get("certainty", 1.0)

        if static_polarity and candidate.polarity != "unknown":
            if candidate.polarity != static_polarity:
                logger.warning(
                    f"Static conflict: {candidate.candidate_id} "
                    f"expected {static_polarity}, got {candidate.polarity}"
                )
                return StaticGuardResult(
                    candidate_id=candidate.candidate_id,
                    static_pass=False,
                    static_conflict=True,
                    action=DomainAction.REJECT_TO_LOG,
                    conflict_rule_id=static_rule.get("rule_id"),
                    expected_polarity=static_polarity,
                    actual_polarity=candidate.polarity,
                    conflict_reason=f"Polarity conflict with static rule: {static_rule.get('description', '')}",
                )

        if static_relation and candidate.relation_type != static_relation:
            if static_certainty >= 0.95:
                logger.debug(
                    "Relation type mismatch but allowing: "
                    f"static={static_relation}, candidate={candidate.relation_type}"
                )

        logger.info(f"Static match: {candidate.candidate_id} matches rule {static_rule.get('rule_id')}")
        return StaticGuardResult(
            candidate_id=candidate.candidate_id,
            static_pass=True,
            static_conflict=False,
            action=DomainAction.STRENGTHEN_STATIC,
            conflict_rule_id=static_rule.get("rule_id"),
            expected_polarity=static_polarity,
            actual_polarity=candidate.polarity,
        )

    def get_static_rule(self, head_id: str, tail_id: str) -> Optional[Dict[str, Any]]:
        """Return a static rule by head/tail."""
        return self._rules_map.get((head_id, tail_id))

    def is_static_relation(self, head_id: str, tail_id: str) -> bool:
        """Check if a head/tail pair is static."""
        return (head_id, tail_id) in self._rules_map
