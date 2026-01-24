"""
Named Entity Recognition (NER) module.
"""
import re
import logging
from typing import List, Optional, Dict, Any

from src.shared.models import EntityCandidate
from src.shared.error_framework import ExtractionError, ErrorSeverity
from src.llm.llm_client import LLMClient, LLMRequest
from src.bootstrap import get_llm_gateway
from config.settings import get_settings

logger = logging.getLogger(__name__)


class NERStudent:
    """Extract entity candidates from fragment text."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.settings = get_settings()
        self.llm_client = llm_client or get_llm_gateway()
        self._entity_types = self._load_entity_types()
        self._alias_hints = self._load_alias_hints()

    def _load_entity_types(self) -> Dict[str, Any]:
        """Load entity types configuration."""
        try:
            return self.settings.load_yaml_config("entity_types")
        except FileNotFoundError:
            logger.warning("Entity types config not found, using defaults")
            return {"entity_types": {}}

    def _load_alias_hints(self) -> Dict[str, str]:
        """Load alias hints to boost rule-based matching."""
        hints = {}
        try:
            alias_config = self.settings.load_yaml_config("alias_dictionary")
            for entry in alias_config.get("aliases", []):
                entity_type = entry.get("type", "Unknown")
                for alias in entry.get("synonyms", []):
                    hints[alias.lower()] = entity_type
        except FileNotFoundError:
            logger.warning("Alias dictionary not found")
        return hints

    def extract(self, fragment_text: str, fragment_id: str, use_llm: bool = True) -> List[EntityCandidate]:
        """Extract entity candidates."""
        if not fragment_text.strip():
            return []

        if use_llm and self.llm_client and self.llm_client.health_check():
            try:
                candidates = self._extract_with_llm(fragment_text, fragment_id)
                if candidates:
                    return candidates
            except Exception as e:
                logger.warning(f"LLM extraction failed, falling back to rules: {e}")

        return self._extract_rule_based(fragment_text, fragment_id)

    def _extract_with_llm(
        self,
        fragment_text: str,
        fragment_id: str,
    ) -> List[EntityCandidate]:
        """Use LLM to extract entities."""
        type_list = ", ".join(self._entity_types.get("entity_types", {}).keys())

        system_prompt = (
            "You are an expert Named Entity Recognition system. "
            f"Extract entities of types: {type_list}. "
            "Output strictly in JSON format: "
            "{'entities': [{'surface_text': '...', 'type': '...', "
            "'normalized_name': '...', 'confidence': 0.0-1.0}]}"
        )

        prompt = f"Text: {fragment_text}"

        try:
            request = LLMRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                json_mode=True,
            )

            result = self.llm_client.generate_json(request)

            entities = []
            for ent_data in result.get("entities", []):
                surface_text = ent_data.get("surface_text", "")
                if not surface_text:
                    continue

                span_start = fragment_text.find(surface_text)
                span_end = span_start + len(surface_text) if span_start != -1 else 0

                if span_start == -1:
                    continue

                entities.append(EntityCandidate(
                    surface_text=surface_text,
                    type_guess=ent_data.get("type", "Unknown"),
                    normalized_name_guess=ent_data.get("normalized_name"),
                    span_start=span_start,
                    span_end=span_end,
                    student_conf=float(ent_data.get("confidence", 0.5)),
                    fragment_id=fragment_id,
                ))

            return entities

        except Exception as e:
            raise ExtractionError(
                message=f"LLM extraction failed: {str(e)}",
                extractor="NERStudent",
                text_preview=fragment_text,
                severity=ErrorSeverity.MEDIUM
            )

    def _extract_rule_based(
        self,
        fragment_text: str,
        fragment_id: str,
    ) -> List[EntityCandidate]:
        """Rule-based extraction (fallback)."""
        entities = []
        text_lower = fragment_text.lower()

        for alias, entity_type in self._alias_hints.items():
            if alias in text_lower:
                pattern = re.compile(re.escape(alias), re.IGNORECASE)
                for match in pattern.finditer(fragment_text):
                    entities.append(EntityCandidate(
                        surface_text=match.group(),
                        type_guess=entity_type,
                        normalized_name_guess=None,
                        span_start=match.start(),
                        span_end=match.end(),
                        student_conf=0.8,
                        fragment_id=fragment_id,
                    ))

        entities.extend(self._extract_by_patterns(fragment_text, fragment_id))

        return entities

    def _extract_by_patterns(
        self,
        fragment_text: str,
        fragment_id: str,
    ) -> List[EntityCandidate]:
        """Simple pattern-based extraction."""
        entities = []

        for match in re.finditer(r'\d+\.?\d*%?', fragment_text):
            entities.append(EntityCandidate(
                surface_text=match.group(),
                type_guess="Quantity",
                span_start=match.start(),
                span_end=match.end(),
                student_conf=0.8,
                fragment_id=fragment_id,
            ))

        return entities
