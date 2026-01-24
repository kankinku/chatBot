"""
Entity resolution module.
"""
import logging
from typing import List, Optional, Dict, Any, Tuple
from difflib import SequenceMatcher

from src.shared.models import EntityCandidate, ResolvedEntity, ResolutionMode
from src.shared.exceptions import EntityResolutionError
from config.settings import get_settings

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolve entity candidates to canonical entities."""

    def __init__(
        self,
        static_domain_kg: Optional[Dict[str, Any]] = None,
        dynamic_domain_kg: Optional[Dict[str, Any]] = None,
        custom_aliases: Optional[Dict[str, str]] = None,
    ):
        self.settings = get_settings()

        self._alias_table = self._build_alias_table()
        self._static_domain = static_domain_kg or {}
        self._dynamic_domain = dynamic_domain_kg or {}
        self._custom_aliases = custom_aliases or {}

        self._stats = {
            "dictionary_match": 0,
            "static_domain": 0,
            "dynamic_domain": 0,
            "custom_alias": 0,
            "fuzzy_match": 0,
            "ambiguous": 0,
            "new_entity": 0,
        }

    def _build_alias_table(self) -> Dict[str, Dict[str, Any]]:
        """Build alias lookup table from config."""
        alias_table = {}
        try:
            alias_data = self.settings.load_yaml_config("alias_dictionary")
            for entity_key, entity_info in alias_data.get("entities", {}).items():
                canonical_id = entity_key
                canonical_name = entity_info.get("canonical_name", entity_key)
                entity_type = entity_info.get("type", "Unknown")
                subtype = entity_info.get("subtype")

                for alias in entity_info.get("aliases", []):
                    alias_lower = alias.lower().strip()
                    alias_table[alias_lower] = {
                        "canonical_id": canonical_id,
                        "canonical_name": canonical_name,
                        "canonical_type": entity_type,
                        "canonical_subtype": subtype,
                    }

                alias_table[canonical_name.lower()] = {
                    "canonical_id": canonical_id,
                    "canonical_name": canonical_name,
                    "canonical_type": entity_type,
                    "canonical_subtype": subtype,
                }

        except FileNotFoundError:
            logger.warning("Alias dictionary not found, resolution will be limited")

        logger.info(f"Built alias table with {len(alias_table)} entries")
        return alias_table

    def resolve(
        self,
        candidates: List[EntityCandidate],
    ) -> List[ResolvedEntity]:
        """Resolve a list of candidates into canonical entities."""
        resolved = []

        for candidate in candidates:
            try:
                resolved_entity = self._resolve_single(candidate)
                resolved.append(resolved_entity)
            except EntityResolutionError as e:
                logger.warning(f"Resolution failed for {candidate.surface_text}: {e}")
                resolved.append(ResolvedEntity(
                    entity_id=candidate.entity_id,
                    resolution_mode=ResolutionMode.NEW_ENTITY,
                    resolution_conf=0.0,
                    is_new_entity_candidate=True,
                    surface_text=candidate.surface_text,
                    fragment_id=candidate.fragment_id,
                ))

        logger.info(f"Resolved {len(resolved)} entities. Stats: {self._stats}")
        return resolved

    def _resolve_single(self, candidate: EntityCandidate) -> ResolvedEntity:
        """Resolve a single candidate by priority."""
        surface_lower = candidate.surface_text.lower().strip()

        if surface_lower in self._alias_table:
            match = self._alias_table[surface_lower]
            self._stats["dictionary_match"] += 1
            return ResolvedEntity(
                entity_id=candidate.entity_id,
                canonical_id=match["canonical_id"],
                canonical_name=match["canonical_name"],
                canonical_type=match["canonical_type"],
                resolution_mode=ResolutionMode.DICTIONARY_MATCH,
                resolution_conf=0.95,
                surface_text=candidate.surface_text,
                fragment_id=candidate.fragment_id,
            )

        static_match = self._match_in_domain(surface_lower, self._static_domain)
        if static_match:
            self._stats["static_domain"] += 1
            return ResolvedEntity(
                entity_id=candidate.entity_id,
                canonical_id=static_match["id"],
                canonical_name=static_match["name"],
                canonical_type=static_match.get("type"),
                resolution_mode=ResolutionMode.STATIC_DOMAIN,
                resolution_conf=0.9,
                surface_text=candidate.surface_text,
                fragment_id=candidate.fragment_id,
            )

        dynamic_match = self._match_in_domain(surface_lower, self._dynamic_domain)
        if dynamic_match:
            self._stats["dynamic_domain"] += 1
            return ResolvedEntity(
                entity_id=candidate.entity_id,
                canonical_id=dynamic_match["id"],
                canonical_name=dynamic_match["name"],
                canonical_type=dynamic_match.get("type"),
                resolution_mode=ResolutionMode.DYNAMIC_DOMAIN,
                resolution_conf=0.85,
                surface_text=candidate.surface_text,
                fragment_id=candidate.fragment_id,
            )

        if surface_lower in self._custom_aliases:
            canonical_name = self._custom_aliases[surface_lower]
            self._stats["custom_alias"] += 1
            return ResolvedEntity(
                entity_id=candidate.entity_id,
                canonical_id=f"CUSTOM_{canonical_name.replace(' ', '_')}",
                canonical_name=canonical_name,
                resolution_mode=ResolutionMode.CUSTOM_ALIAS,
                resolution_conf=0.8,
                surface_text=candidate.surface_text,
                fragment_id=candidate.fragment_id,
            )

        fuzzy_result = self._fuzzy_match(surface_lower)
        if fuzzy_result:
            if len(fuzzy_result) == 1:
                match, conf = fuzzy_result[0]
                self._stats["fuzzy_match"] += 1
                return ResolvedEntity(
                    entity_id=candidate.entity_id,
                    canonical_id=match["canonical_id"],
                    canonical_name=match["canonical_name"],
                    canonical_type=match.get("canonical_type"),
                    resolution_mode=ResolutionMode.FUZZY_MATCH,
                    resolution_conf=conf,
                    surface_text=candidate.surface_text,
                    fragment_id=candidate.fragment_id,
                )

            self._stats["ambiguous"] += 1
            return ResolvedEntity(
                entity_id=candidate.entity_id,
                resolution_mode=ResolutionMode.AMBIGUOUS,
                resolution_conf=0.5,
                candidate_ids=[m["canonical_id"] for m, _ in fuzzy_result],
                candidate_confs=[c for _, c in fuzzy_result],
                surface_text=candidate.surface_text,
                fragment_id=candidate.fragment_id,
            )

        self._stats["new_entity"] += 1
        return ResolvedEntity(
            entity_id=candidate.entity_id,
            resolution_mode=ResolutionMode.NEW_ENTITY,
            resolution_conf=0.0,
            is_new_entity_candidate=True,
            surface_text=candidate.surface_text,
            fragment_id=candidate.fragment_id,
        )

    def _match_in_domain(
        self,
        surface: str,
        domain_kg: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Match by name in a domain KG dict."""
        for entity_id, entity_info in domain_kg.items():
            if isinstance(entity_info, dict):
                name = entity_info.get("name", "").lower()
                if surface == name:
                    return {"id": entity_id, "name": entity_info.get("name"), "type": entity_info.get("type")}
        return None

    def _fuzzy_match(
        self,
        surface: str,
    ) -> Optional[List[Tuple[Dict[str, Any], float]]]:
        """Fuzzy match against known aliases."""
        threshold = self.settings.extraction.fuzzy_match_threshold
        matches = []

        for alias, canonical_info in self._alias_table.items():
            similarity = SequenceMatcher(None, surface, alias).ratio()
            if similarity >= threshold:
                matches.append((canonical_info, similarity))

        if not matches:
            return None

        matches.sort(key=lambda x: x[1], reverse=True)

        top_conf = matches[0][1]
        close_matches = [(m, c) for m, c in matches if top_conf - c <= 0.05]

        return close_matches[:3]

    def add_custom_alias(self, alias: str, canonical_name: str):
        """Add a custom alias mapping."""
        self._custom_aliases[alias.lower().strip()] = canonical_name
        logger.info(f"Added custom alias: {alias} -> {canonical_name}")

    def get_stats(self) -> Dict[str, int]:
        """Return resolution stats."""
        return self._stats.copy()

    def reset_stats(self):
        """Reset resolution stats."""
        for key in self._stats:
            self._stats[key] = 0
