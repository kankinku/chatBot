"""
Semantic validator with local heuristics and optional LLM checks.
"""
import logging
from typing import Optional, Dict, Any, List

from src.shared.models import RawEdge, ResolvedEntity
from src.validation.models import SemanticValidationResult, SemanticTag
from src.llm.ollama_client import OllamaClient
from config.settings import get_settings

logger = logging.getLogger(__name__)


class SemanticValidator:
    """Checks semantic plausibility for an extracted edge."""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.settings = get_settings()
        self.llm_client = llm_client
        self._semantic_patterns = self._load_semantic_patterns()

    def _load_semantic_patterns(self) -> Dict[str, List[str]]:
        try:
            data = self.settings.load_yaml_config("static_domain")
            return data.get("semantic_patterns", {})
        except FileNotFoundError:
            return {
                "exaggeration": ["always", "never", "guaranteed"],
                "correlation_as_causation": ["correlates with", "goes with", "moves with"],
                "weak_evidence": ["maybe", "might", "rumor", "alleged"],
            }

    def validate(
        self,
        edge: RawEdge,
        fragment_text: str,
        resolved_entities: List[ResolvedEntity],
        domain_kg: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> SemanticValidationResult:
        """Run semantic validation for an edge."""
        has_exaggeration = self._check_exaggeration(fragment_text)
        is_correlation = self._check_correlation_as_causation(fragment_text, edge)
        has_weak = self._check_weak_evidence(fragment_text)

        domain_conflict = False
        if domain_kg:
            domain_conflict = self._check_domain_conflict(edge, domain_kg)

        llm_judgement = None
        if use_llm and self.llm_client:
            llm_judgement = self._get_llm_judgement(edge, fragment_text)

        semantic_tag, semantic_conf = self._determine_semantic_tag(
            has_exaggeration=has_exaggeration,
            is_correlation=is_correlation,
            has_weak=has_weak,
            domain_conflict=domain_conflict,
            llm_judgement=llm_judgement,
        )

        return SemanticValidationResult(
            edge_id=edge.raw_edge_id,
            semantic_tag=semantic_tag,
            semantic_confidence=semantic_conf,
            has_exaggeration=has_exaggeration,
            is_correlation_as_causation=is_correlation,
            has_weak_evidence=has_weak,
            domain_conflict=domain_conflict,
            llm_judgement=llm_judgement,
        )

    def _check_exaggeration(self, text: str) -> bool:
        """Detect exaggeration phrases."""
        patterns = self._semantic_patterns.get("exaggeration", [])
        return any(p in text for p in patterns)

    def _check_correlation_as_causation(self, text: str, edge: RawEdge) -> bool:
        """Detect correlation phrasing for a causal relation."""
        if edge.relation_type != "Cause":
            return False

        patterns = self._semantic_patterns.get("correlation_as_causation", [])
        return any(p in text for p in patterns)

    def _check_weak_evidence(self, text: str) -> bool:
        """Detect weak-evidence phrasing."""
        patterns = self._semantic_patterns.get("weak_evidence", [])
        return any(p in text for p in patterns)

    def _check_domain_conflict(self, edge: RawEdge, domain_kg: Dict[str, Any]) -> bool:
        """Basic conflict check against existing domain edges."""
        existing = domain_kg.get("edges", {})

        for edge_info in existing.values():
            if (
                edge_info.get("head") == edge.head_canonical_name
                and edge_info.get("tail") == edge.tail_canonical_name
            ):
                existing_polarity = edge_info.get("polarity")
                new_polarity = str(edge.polarity_guess) if edge.polarity_guess else None

                if existing_polarity and new_polarity and existing_polarity != new_polarity:
                    return True

        return False

    def _get_llm_judgement(self, edge: RawEdge, text: str) -> Optional[str]:
        """Ask the LLM for a semantic judgement."""
        prompt = f"""Assess if the relation is supported by the sentence.

Sentence: "{text}"
Relation: {edge.head_canonical_name} --[{edge.relation_type}]--> {edge.tail_canonical_name}

Return JSON:
{{"judgement": "valid|weak|spurious|wrong|ambiguous", "reason": "..."}}"""

        try:
            result = self.llm_client.generate_json(prompt=prompt, temperature=0.1)
            return result.get("judgement")
        except Exception as e:
            logger.warning(f"LLM semantic judgement failed: {e}")
            return None

    def _determine_semantic_tag(
        self,
        has_exaggeration: bool,
        is_correlation: bool,
        has_weak: bool,
        domain_conflict: bool,
        llm_judgement: Optional[str],
    ) -> tuple:
        """Map signals to a semantic tag and confidence."""
        if domain_conflict:
            return SemanticTag.SEM_WRONG, 0.2

        if llm_judgement == "wrong":
            return SemanticTag.SEM_WRONG, 0.25

        if is_correlation or llm_judgement == "spurious":
            return SemanticTag.SEM_SPURIOUS, 0.35

        if llm_judgement == "valid" and not has_exaggeration and not has_weak:
            return SemanticTag.SEM_CONFIDENT, 0.85

        if has_exaggeration or has_weak:
            return SemanticTag.SEM_WEAK, 0.5

        if llm_judgement == "weak":
            return SemanticTag.SEM_WEAK, 0.5

        if llm_judgement == "ambiguous":
            return SemanticTag.SEM_AMBIGUOUS, 0.55

        if llm_judgement == "valid":
            return SemanticTag.SEM_CONFIDENT, 0.75

        return SemanticTag.SEM_AMBIGUOUS, 0.5
