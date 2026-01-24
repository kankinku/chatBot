"""
Fragment extraction module.
"""
import re
import logging
from typing import List, Optional

from src.shared.models import Fragment, QualityTag
from src.shared.exceptions import FragmentExtractionError
from src.llm.ollama_client import OllamaClient
from config.settings import get_settings

logger = logging.getLogger(__name__)


NOISE_PATTERNS = [
    r'^(lol|omg|wow|hmm|uh|um|hey)$',
    r'^(ok|okay|fine|sure)[\.\!\?]?$',
    r'^[\!\?\.]+$',
]

CAUSAL_PATTERNS = [
    r'.+\bif\b.+\bthen\b.+',
    r'.+\bif\b.+',
    r'.+\bwhen\b.+',
    r'.+\bbecause\b.+',
    r'.+\bdue to\b.+',
    r'.+\bas a result\b.+',
    r'.+\b(leads to|results in|causes)\b.+',
]


class FragmentExtractor:
    """Split raw text into relation-bearing fragments."""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.settings = get_settings().extraction
        self.llm_client = llm_client
        self._noise_regex = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]
        self._causal_regex = [re.compile(p, re.IGNORECASE) for p in CAUSAL_PATTERNS]

    def extract(
        self,
        raw_text: str,
        doc_id: str,
        use_llm: bool = True,
    ) -> List[Fragment]:
        """Extract fragments from raw text."""
        if not raw_text or not raw_text.strip():
            raise FragmentExtractionError(
                message="Empty raw text provided",
                doc_id=doc_id,
                recoverable=False,
            )

        try:
            if use_llm and self.llm_client:
                fragments = self._extract_with_llm(raw_text, doc_id)
            else:
                fragments = self._extract_rule_based(raw_text, doc_id)

            fragments = [self._tag_quality(f) for f in fragments]
            fragments = self._filter_by_length(fragments)

            logger.info(f"Extracted {len(fragments)} fragments from doc {doc_id}")
            return fragments

        except FragmentExtractionError:
            raise
        except Exception as e:
            raise FragmentExtractionError(
                message=f"Fragment extraction failed: {str(e)}",
                doc_id=doc_id,
                raw_text_preview=raw_text,
                recoverable=True,
            )

    def _extract_with_llm(self, raw_text: str, doc_id: str) -> List[Fragment]:
        """LLM-based fragment extraction."""
        system_prompt = """You are a text segmentation expert.
Split the input into minimal fragments that each express a single relation or claim.

Rules:
1. Each fragment should be self-contained and meaningful.
2. Do not split conditional or causal constructions (e.g., "if ... then ...", "because ...").
3. Separate pure emotion or filler into its own fragment.
4. Keep fragments concise and focused.

Return JSON:
{
  "fragments": [
    {"text": "fragment text", "quality": "informative"},
    {"text": "fragment text", "quality": "noisy"}
  ]
}

Quality values: informative, noisy, unclear, emotional, incomplete"""

        prompt = f"""Split the following text into fragments:

{raw_text}"""

        try:
            result = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
            )

            fragments = []
            for frag_data in result.get("fragments", []):
                text = frag_data.get("text", "").strip()
                if not text:
                    continue

                start = raw_text.find(text)
                end = start + len(text) if start >= 0 else None

                quality_str = frag_data.get("quality", "informative")
                try:
                    quality = QualityTag(quality_str)
                except ValueError:
                    quality = QualityTag.INFORMATIVE

                fragment = Fragment(
                    text=text,
                    doc_id=doc_id,
                    quality_tag=quality,
                    source_start=start if start >= 0 else None,
                    source_end=end,
                )
                fragments.append(fragment)

            return fragments

        except Exception as e:
            logger.warning(f"LLM extraction failed, falling back to rule-based: {e}")
            return self._extract_rule_based(raw_text, doc_id)

    def _extract_rule_based(self, raw_text: str, doc_id: str) -> List[Fragment]:
        """Rule-based fragment extraction (fallback)."""
        sentences = self._split_sentences(raw_text)

        fragments = []
        current_pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            start = raw_text.find(sentence, current_pos)
            if start >= 0:
                current_pos = start + len(sentence)

            sub_fragments = self._split_multiple_relations(sentence)

            for sub_text in sub_fragments:
                fragment = Fragment(
                    text=sub_text,
                    doc_id=doc_id,
                    source_start=start if start >= 0 else None,
                    source_end=start + len(sub_text) if start >= 0 else None,
                )
                fragments.append(fragment)

        return fragments

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        pattern = r'(?<![0-9])[\.\?\!]+(?=\s|$)'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_multiple_relations(self, sentence: str) -> List[str]:
        """Split a sentence when multiple relations appear."""
        connectors = [", and ", ", but ", "; "]

        for pattern in self._causal_regex:
            if pattern.match(sentence):
                return [sentence]

        for conn in connectors:
            if conn in sentence:
                parts = sentence.split(conn)
                if len(parts) == 2 and all(len(p.strip()) > 10 for p in parts):
                    return [p.strip() for p in parts]

        return [sentence]

    def _tag_quality(self, fragment: Fragment) -> Fragment:
        """Assign a quality tag to a fragment."""
        text = fragment.text.strip()

        for pattern in self._noise_regex:
            if pattern.match(text):
                fragment.quality_tag = QualityTag.NOISY
                return fragment

        if len(text) < self.settings.min_fragment_length:
            fragment.quality_tag = QualityTag.INCOMPLETE

        if text.endswith("?") and "?" not in text[:-1]:
            fragment.quality_tag = QualityTag.UNCLEAR

        return fragment

    def _filter_by_length(self, fragments: List[Fragment]) -> List[Fragment]:
        """Filter fragments by length thresholds."""
        filtered = []
        for fragment in fragments:
            text_len = len(fragment.text.strip())
            if text_len < self.settings.min_fragment_length:
                logger.debug(f"Fragment too short, skipping: {fragment.text[:30]}...")
                continue
            if text_len > self.settings.max_fragment_length:
                logger.warning(f"Fragment too long: {fragment.text[:50]}...")
            filtered.append(fragment)
        return filtered
