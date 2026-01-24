"""
Sign validator for relation polarity.
"""
import logging
from typing import Optional, Dict, Any, List

from src.shared.models import RawEdge, ResolvedEntity, Polarity
from src.validation.models import SignValidationResult, SignTag
from src.llm.ollama_client import OllamaClient
from config.settings import get_settings

logger = logging.getLogger(__name__)


class SignValidator:
    """Validate polarity signals for an edge."""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.settings = get_settings()
        self.llm_client = llm_client
        self._static_domain = self._load_static_domain()
        self._sign_patterns = self._load_sign_patterns()

        self._static_rules_map = self._build_static_rules_map()

    def _load_static_domain(self) -> Dict[str, Any]:
        try:
            return self.settings.load_yaml_config("static_domain")
        except FileNotFoundError:
            logger.warning("Static domain not found")
            return {}

    def _load_sign_patterns(self) -> Dict[str, List[str]]:
        patterns = self._static_domain.get("sign_patterns", {})
        return {
            "positive": patterns.get("positive", []),
            "negative": patterns.get("negative", []),
            "inverse": patterns.get("inverse", []),
        }

    def _build_static_rules_map(self) -> Dict[tuple, Dict[str, Any]]:
        """Index static rules by (head_canonical, tail_canonical)."""
        rules_map = {}
        for rule in self._static_domain.get("static_rules", []):
            key = (rule.get("head"), rule.get("tail"))
            rules_map[key] = rule
        return rules_map

    def validate(
        self,
        edge: RawEdge,
        fragment_text: str,
        resolved_entities: List[ResolvedEntity],
        use_llm: bool = True,
    ) -> SignValidationResult:
        """Run sign validation."""
        entity_map = {e.entity_id: e for e in resolved_entities}
        head_entity = entity_map.get(edge.head_entity_id)
        tail_entity = entity_map.get(edge.tail_entity_id)

        pattern_polarity = self._estimate_from_patterns(fragment_text)

        domain_polarity = None
        conflict_with_static = False
        static_certainty = 0.0

        if head_entity and tail_entity:
            head_canonical = head_entity.canonical_id
            tail_canonical = tail_entity.canonical_id

            if head_canonical and tail_canonical:
                static_rule = self._static_rules_map.get((head_canonical, tail_canonical))
                if static_rule:
                    domain_polarity = static_rule.get("polarity")
                    static_certainty = static_rule.get("certainty", 0.8)

                    student_pol = self._normalize_polarity(edge.polarity_guess)
                    if domain_polarity and student_pol:
                        if student_pol != domain_polarity and student_pol != "unknown":
                            conflict_with_static = True
                            logger.warning(
                                f"Static domain conflict: {edge.raw_edge_id}, "
                                f"student={student_pol}, domain={domain_polarity}"
                            )

        llm_polarity = None
        if use_llm and self.llm_client and pattern_polarity is None:
            llm_polarity = self._get_llm_polarity(fragment_text, edge)

        polarity_final, sign_tag, consistency_score = self._determine_final_sign(
            student_polarity=self._normalize_polarity(edge.polarity_guess),
            pattern_polarity=pattern_polarity,
            domain_polarity=domain_polarity,
            llm_polarity=llm_polarity,
            conflict_with_static=conflict_with_static,
            static_certainty=static_certainty,
        )

        return SignValidationResult(
            edge_id=edge.raw_edge_id,
            polarity_final=polarity_final,
            sign_tag=sign_tag,
            sign_consistency_score=consistency_score,
            pattern_polarity=pattern_polarity,
            domain_polarity=domain_polarity,
            llm_polarity=llm_polarity,
            conflict_with_static=conflict_with_static,
        )

    def _normalize_polarity(self, polarity) -> Optional[str]:
        """Normalize polarity to a string."""
        if polarity is None:
            return None
        if isinstance(polarity, Polarity):
            if polarity == Polarity.POSITIVE:
                return "+"
            if polarity == Polarity.NEGATIVE:
                return "-"
            if polarity == Polarity.NEUTRAL:
                return "neutral"
            return "unknown"
        return str(polarity)

    def _estimate_from_patterns(self, text: str) -> Optional[str]:
        """Estimate polarity based on sign patterns."""
        text_lower = text.lower()

        for pattern in self._sign_patterns.get("inverse", []):
            if pattern in text_lower:
                return "-"

        pos_count = sum(1 for p in self._sign_patterns.get("positive", []) if p in text_lower)
        neg_count = sum(1 for p in self._sign_patterns.get("negative", []) if p in text_lower)

        if pos_count > 0 and neg_count == 0:
            return "+"
        if neg_count > 0 and pos_count == 0:
            return "-"
        if pos_count > 0 and neg_count > 0:
            return None

        return None

    def _get_llm_polarity(self, text: str, edge: RawEdge) -> Optional[str]:
        """Ask the LLM for polarity judgement."""
        prompt = f"""Determine the direction of influence from the sentence.

Sentence: "{text}"
Relation: "{edge.head_canonical_name}" -> "{edge.tail_canonical_name}"

Return JSON:
{{"polarity": "+|-|neutral|unknown", "confidence": 0.0-1.0}}"""

        try:
            result = self.llm_client.generate_json(prompt=prompt, temperature=0.1)
            return result.get("polarity")
        except Exception as e:
            logger.warning(f"LLM polarity check failed: {e}")
            return None

    def _determine_final_sign(
        self,
        student_polarity: Optional[str],
        pattern_polarity: Optional[str],
        domain_polarity: Optional[str],
        llm_polarity: Optional[str],
        conflict_with_static: bool,
        static_certainty: float,
    ) -> tuple:
        """Combine signals into final polarity."""
        if conflict_with_static and static_certainty >= 0.9:
            return domain_polarity or "unknown", SignTag.SUSPECT, 0.3

        sources = [p for p in [student_polarity, pattern_polarity, domain_polarity, llm_polarity] if p and p != "unknown"]

        if not sources:
            return "unknown", SignTag.UNKNOWN, 0.0

        if len(set(sources)) == 1:
            return sources[0], SignTag.CONFIDENT, 0.9

        if domain_polarity:
            matching = sum(1 for s in sources if s == domain_polarity)
            if matching >= len(sources) / 2:
                return domain_polarity, SignTag.CONFIDENT, 0.8
            return domain_polarity, SignTag.AMBIGUOUS, 0.6

        from collections import Counter
        counter = Counter(sources)
        most_common = counter.most_common(1)[0]

        if most_common[1] > len(sources) / 2:
            return most_common[0], SignTag.AMBIGUOUS, 0.5

        return student_polarity or "unknown", SignTag.AMBIGUOUS, 0.4
