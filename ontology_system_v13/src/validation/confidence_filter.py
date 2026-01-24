"""Confidence-based routing for validated edges."""
import logging

from src.shared.models import RawEdge
from src.validation.models import (
    SchemaValidationResult,
    SignValidationResult,
    SemanticValidationResult,
    ValidationResult,
    ValidationDestination,
    SignTag,
    SemanticTag,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ConfidenceFilter:
    """Route validated edges to domain or drop based on confidence."""

    def __init__(self):
        self.settings = get_settings()
        self._thresholds = self._load_thresholds()
        self._weights = self._load_weights()

    def _load_thresholds(self) -> dict:
        try:
            data = self.settings.load_yaml_config("validation_schema")
            rules = data.get("validation_rules", {})
            return rules.get("confidence_thresholds", {"domain_candidate": 0.55})
        except FileNotFoundError:
            return {"domain_candidate": 0.55}

    def _load_weights(self) -> dict:
        try:
            data = self.settings.load_yaml_config("validation_schema")
            rules = data.get("validation_rules", {})
            return rules.get("confidence_weights", {
                "student_conf": 0.4,
                "sign_score": 0.3,
                "semantic_conf": 0.3,
            })
        except FileNotFoundError:
            return {"student_conf": 0.4, "sign_score": 0.3, "semantic_conf": 0.3}

    def filter(
        self,
        edge: RawEdge,
        schema_result: SchemaValidationResult,
        sign_result: SignValidationResult,
        semantic_result: SemanticValidationResult,
    ) -> ValidationResult:
        rejection_reasons = []

        if not schema_result.schema_valid:
            rejection_reasons.append("schema_invalid")

        allowed_sign_tags = {SignTag.CONFIDENT, SignTag.AMBIGUOUS}
        if sign_result.sign_tag not in allowed_sign_tags:
            rejection_reasons.append(f"sign_tag:{sign_result.sign_tag.value}")

        allowed_semantic_tags = {
            SemanticTag.SEM_CONFIDENT,
            SemanticTag.SEM_WEAK,
            SemanticTag.SEM_AMBIGUOUS,
        }
        if semantic_result.semantic_tag not in allowed_semantic_tags:
            rejection_reasons.append(f"semantic_tag:{semantic_result.semantic_tag.value}")

        student_conf = edge.student_conf if edge.student_conf else 0.0
        sign_score = sign_result.sign_consistency_score
        semantic_conf = semantic_result.semantic_confidence

        combined_conf = (
            self._weights["student_conf"] * student_conf +
            self._weights["sign_score"] * sign_score +
            self._weights["semantic_conf"] * semantic_conf
        )

        if rejection_reasons:
            return ValidationResult(
                edge_id=edge.raw_edge_id,
                validation_passed=False,
                destination=ValidationDestination.DROP_LOG,
                combined_conf=combined_conf,
                student_conf=student_conf,
                sign_score=sign_score,
                semantic_conf=semantic_conf,
                schema_result=schema_result,
                sign_result=sign_result,
                semantic_result=semantic_result,
                rejection_reason=rejection_reasons[0],
                rejection_details=rejection_reasons,
            )

        domain_threshold = self._thresholds["domain_candidate"]
        if combined_conf >= domain_threshold:
            destination = ValidationDestination.DOMAIN_CANDIDATE
        else:
            return ValidationResult(
                edge_id=edge.raw_edge_id,
                validation_passed=False,
                destination=ValidationDestination.DROP_LOG,
                combined_conf=combined_conf,
                student_conf=student_conf,
                sign_score=sign_score,
                semantic_conf=semantic_conf,
                schema_result=schema_result,
                sign_result=sign_result,
                semantic_result=semantic_result,
                rejection_reason="low_confidence",
                rejection_details=[f"combined_conf:{combined_conf:.3f} < {domain_threshold}"],
            )

        logger.info(
            f"Edge {edge.raw_edge_id} passed validation -> {destination.value} "
            f"(conf={combined_conf:.3f})"
        )

        return ValidationResult(
            edge_id=edge.raw_edge_id,
            validation_passed=True,
            destination=destination,
            combined_conf=combined_conf,
            student_conf=student_conf,
            sign_score=sign_score,
            semantic_conf=semantic_conf,
            schema_result=schema_result,
            sign_result=sign_result,
            semantic_result=semantic_result,
        )

    def set_thresholds(self, domain_threshold: float = None) -> None:
        if domain_threshold is not None:
            self._thresholds["domain_candidate"] = float(domain_threshold)
