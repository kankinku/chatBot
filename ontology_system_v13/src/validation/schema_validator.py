"""
Schema validator for structural constraints.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from src.shared.models import RawEdge, ResolvedEntity
from src.validation.models import SchemaValidationResult
from config.settings import get_settings

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validates basic schema constraints for extracted edges.
    """

    def __init__(self):
        self.settings = get_settings()
        self._validation_schema = self._load_validation_schema()
        self._relation_types = self._load_relation_types()
        self._allowed_combinations = self._build_allowed_set()
        self._forbidden_combinations = self._build_forbidden_set()

    def _load_validation_schema(self) -> Dict[str, Any]:
        """Load schema rules."""
        try:
            return self.settings.load_yaml_config("validation_schema")
        except FileNotFoundError:
            logger.warning("Validation schema not found, using permissive mode")
            return {"validation_rules": {}}

    def _load_relation_types(self) -> Set[str]:
        """Load known relation types."""
        try:
            data = self.settings.load_yaml_config("relation_types")
            return set(data.get("relation_types", {}).keys())
        except FileNotFoundError:
            return {
                "Affect",
                "Cause",
                "DependOn",
                "TemporalBefore",
                "TemporalAfter",
                "CorrelateWith",
                "PartOf",
                "ConditionalOn",
            }

    def _build_allowed_set(self) -> Set[Tuple[str, str, str]]:
        """Allowed (head_type, tail_type, relation) triples."""
        allowed = set()
        rules = self._validation_schema.get("validation_rules", {})

        for combo in rules.get("allowed_combinations", []):
            head_type = combo.get("head_type")
            tail_type = combo.get("tail_type")
            for rel in combo.get("relations", []):
                allowed.add((head_type, tail_type, rel))

        return allowed

    def _build_forbidden_set(self) -> Dict[Tuple[str, str, str], str]:
        """Forbidden (head_type, tail_type, relation) triples."""
        forbidden = {}
        rules = self._validation_schema.get("validation_rules", {})

        for combo in rules.get("forbidden_combinations", []):
            head_type = combo.get("head_type")
            tail_type = combo.get("tail_type")
            reason = combo.get("reason", "Forbidden combination")
            for rel in combo.get("relations", []):
                forbidden[(head_type, tail_type, rel)] = reason

        return forbidden

    def validate(
        self,
        edge: RawEdge,
        resolved_entities: List[ResolvedEntity],
    ) -> SchemaValidationResult:
        """Run schema validation for a raw edge."""
        errors = []

        entity_map = {e.entity_id: e for e in resolved_entities}

        has_required = self._check_required_fields(edge)
        if not has_required:
            errors.append("missing_required_fields")

        relation_valid = edge.relation_type in self._relation_types
        if not relation_valid:
            errors.append(f"invalid_relation_type:{edge.relation_type}")

        entity_pair_valid = True
        head_entity = entity_map.get(edge.head_entity_id)
        tail_entity = entity_map.get(edge.tail_entity_id)

        if head_entity and tail_entity:
            head_type = head_entity.canonical_type
            tail_type = tail_entity.canonical_type

            if head_type and tail_type:
                combo = (head_type, tail_type, edge.relation_type)

                if combo in self._forbidden_combinations:
                    entity_pair_valid = False
                    reason = self._forbidden_combinations[combo]
                    errors.append(f"forbidden_entity_pair:{reason}")
                elif self._allowed_combinations and combo not in self._allowed_combinations:
                    entity_pair_valid = False
                    errors.append(
                        f"entity_pair_not_allowed:{head_type}:{tail_type}:{edge.relation_type}"
                    )
            else:
                logger.debug(
                    "Entity types missing; skipping type-combination validation."
                )
        else:
            if not head_entity:
                errors.append(f"head_entity_not_found:{edge.head_entity_id}")
            if not tail_entity:
                errors.append(f"tail_entity_not_found:{edge.tail_entity_id}")
            entity_pair_valid = False

        no_self_loop = edge.head_entity_id != edge.tail_entity_id
        if not no_self_loop:
            errors.append("self_loop_detected")

        schema_valid = has_required and relation_valid and entity_pair_valid and no_self_loop

        result = SchemaValidationResult(
            edge_id=edge.raw_edge_id,
            schema_valid=schema_valid,
            schema_errors=errors,
            has_required_fields=has_required,
            relation_type_valid=relation_valid,
            entity_pair_valid=entity_pair_valid,
            no_self_loop=no_self_loop,
        )

        if not schema_valid:
            logger.info(f"Schema validation failed for {edge.raw_edge_id}: {errors}")

        return result

    def _check_required_fields(self, edge: RawEdge) -> bool:
        """Check required fields exist."""
        required_fields = [
            edge.head_entity_id,
            edge.tail_entity_id,
            edge.relation_type,
            edge.fragment_id,
        ]
        return all(f is not None and f != "" for f in required_fields)
