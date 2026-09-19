"""
Extraction pipeline: fragments -> entities -> relations.
"""
import hashlib
import logging
import time
from typing import List, Optional

from chatbot.knowledge.shared.models import (
    Fragment, EntityCandidate, ResolvedEntity, RawEdge, ExtractionResult
)
from chatbot.knowledge.shared.exceptions import ExtractionError
from chatbot.knowledge.llm.ollama_client import OllamaClient
from .fragment_extractor import FragmentExtractor
from .ner_student import NERStudent
from .entity_resolver import EntityResolver
from .relation_extractor import RelationExtractor

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """End-to-end extraction pipeline."""

    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        use_llm: bool = True,
    ):
        self.use_llm = use_llm
        self.llm_client = llm_client

        self.fragment_extractor = FragmentExtractor(llm_client=llm_client)
        self.ner_student = NERStudent(llm_client=llm_client)
        self.entity_resolver = EntityResolver()
        self.relation_extractor = RelationExtractor(llm_client=llm_client)

    def process(
        self,
        raw_text: str,
        doc_id: str,
        source_uri: Optional[str] = None,
    ) -> ExtractionResult:
        """Process a single document and preserve source provenance."""
        start_time = time.time()
        source_uri = source_uri or f"document:{doc_id}"
        source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        warnings = []
        error_count = 0

        all_fragments: List[Fragment] = []
        all_entity_candidates: List[EntityCandidate] = []
        all_resolved: List[ResolvedEntity] = []
        all_edges: List[RawEdge] = []

        try:
            logger.info(f"[Pipeline] Step 1: Fragment extraction for {doc_id}")
            fragments = self.fragment_extractor.extract(
                raw_text=raw_text,
                doc_id=doc_id,
                use_llm=self.use_llm,
            )
            all_fragments = fragments
            logger.info(f"  Extracted {len(fragments)} fragments")

            for fragment in fragments:
                try:
                    entity_candidates = self.ner_student.extract(
                        fragment_text=fragment.text,
                        fragment_id=fragment.fragment_id,
                        use_llm=self.use_llm,
                    )
                    all_entity_candidates.extend(entity_candidates)

                    if not entity_candidates:
                        continue

                    resolved_entities = self.entity_resolver.resolve(entity_candidates)
                    all_resolved.extend(resolved_entities)

                    raw_edges = self.relation_extractor.extract(
                        fragment_text=fragment.text,
                        fragment_id=fragment.fragment_id,
                        resolved_entities=resolved_entities,
                        use_llm=self.use_llm,
                    )
                    all_edges.extend(raw_edges)

                except Exception as e:
                    error_count += 1
                    warnings.append(f"Fragment {fragment.fragment_id}: {str(e)}")
                    logger.warning(f"Error processing fragment: {e}")
                    continue

        except ExtractionError as e:
            error_count += 1
            warnings.append(f"Pipeline error: {str(e)}")
            logger.error(f"Pipeline failed: {e}")

        processing_time = (time.time() - start_time) * 1000

        result = ExtractionResult(
            doc_id=doc_id,
            source_uri=source_uri,
            source_hash=source_hash,
            fragments=all_fragments,
            entity_candidates=all_entity_candidates,
            resolved_entities=all_resolved,
            raw_edges=all_edges,
            processing_time_ms=processing_time,
            error_count=error_count,
            warning_messages=warnings,
        )

        logger.info(
            f"[Pipeline] Complete: {len(all_fragments)} fragments, "
            f"{len(all_entity_candidates)} entities, {len(all_edges)} edges, "
            f"{processing_time:.2f}ms"
        )

        return result

    def process_batch(
        self,
        documents: List[dict],
    ) -> List[ExtractionResult]:
        """Process a list of documents."""
        results = []
        for doc in documents:
            doc_id = doc.get("doc_id", f"doc_{len(results)}")
            text = doc.get("text", "")
            source_uri = doc.get("source_uri")

            result = self.process(
                raw_text=text,
                doc_id=doc_id,
                source_uri=source_uri,
            )
            results.append(result)

        return results
