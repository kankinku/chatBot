"""Conclusion synthesis for reasoning results."""
import logging
from typing import Optional

from src.reasoning.models import ParsedQuery, ReasoningResult, ReasoningConclusion, ReasoningDirection
from src.llm.llm_client import LLMClient, LLMRequest

logger = logging.getLogger(__name__)


class ConclusionSynthesizer:
    """Builds a natural language conclusion from reasoning results."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client

    def synthesize(self, parsed_query: ParsedQuery, reasoning_result: ReasoningResult) -> ReasoningConclusion:
        conclusion_text = self._generate_conclusion_text(parsed_query, reasoning_result)
        explanation_text = self._generate_explanation(reasoning_result)
        strongest_path_desc = self._describe_strongest_path(reasoning_result)
        evidence_summary = self._summarize_evidence(reasoning_result)

        if self.llm_client:
            conclusion_text = self._polish_with_llm(conclusion_text, parsed_query.original_query)

        return ReasoningConclusion(
            query_id=parsed_query.query_id,
            original_query=parsed_query.original_query,
            conclusion_text=conclusion_text,
            explanation_text=explanation_text,
            direction=reasoning_result.direction,
            confidence=reasoning_result.confidence,
            strongest_path_description=strongest_path_desc,
            evidence_summary=evidence_summary,
            reasoning_result=reasoning_result,
        )

    def _generate_conclusion_text(self, query: ParsedQuery, result: ReasoningResult) -> str:
        head_name = query.entity_names.get(query.head_entity, query.head_entity or "")
        tail_name = query.entity_names.get(query.tail_entity, query.tail_entity or "")

        direction = result.direction
        if direction == ReasoningDirection.POSITIVE:
            dir_text = "positive influence"
        elif direction == ReasoningDirection.NEGATIVE:
            dir_text = "negative influence"
        elif direction == ReasoningDirection.NEUTRAL:
            dir_text = "neutral influence"
        else:
            dir_text = "unknown influence"

        conf_text = self._confidence_text(result.confidence)

        if not tail_name:
            return f"{head_name}: {dir_text} ({conf_text})"
        return f"{head_name} -> {tail_name}: {dir_text} ({conf_text})"

    def _confidence_text(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "very high confidence"
        if confidence >= 0.6:
            return "high confidence"
        if confidence >= 0.4:
            return "medium confidence"
        if confidence >= 0.2:
            return "low confidence"
        return "very low confidence"

    def _generate_explanation(self, result: ReasoningResult) -> str:
        lines = [
            f"Paths used: {len(result.paths_used)}",
            f"Evidence (+/-): {result.positive_evidence:.3f} / {result.negative_evidence:.3f}",
        ]
        if result.conflicting_paths > 0:
            lines.append(f"Conflicting paths: {result.conflicting_paths}")
        return "\n".join(lines)

    def _describe_strongest_path(self, result: ReasoningResult) -> str:
        if not result.strongest_path:
            return "no path"

        path = result.strongest_path
        parts = []
        for i, node in enumerate(path.node_names):
            if i < len(path.edge_signs):
                sign = "+" if path.edge_signs[i] == "+" else "-"
                parts.append(f"{node}({sign})")
            else:
                parts.append(node)
        return " -> ".join(parts)

    def _summarize_evidence(self, result: ReasoningResult) -> str:
        total = result.positive_evidence + result.negative_evidence
        if total == 0:
            return "no evidence"
        pos_ratio = result.positive_evidence / total * 100
        neg_ratio = result.negative_evidence / total * 100
        return f"positive {pos_ratio:.1f}%, negative {neg_ratio:.1f}%"

    def _polish_with_llm(self, text: str, original_query: str) -> str:
        try:
            prompt = (
                "Rewrite the conclusion to be concise and natural. "
                "Do not change the direction or confidence.\n"
                f"Query: {original_query}\n"
                f"Conclusion: {text}"
            )
            request = LLMRequest(prompt=prompt, temperature=0.3)
            response = self.llm_client.generate(request)
            return response.content.strip() or text
        except Exception as e:
            logger.warning(f"LLM polish failed: {e}")
            return text
