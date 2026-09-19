"""End-to-end extraction -> validation -> domain -> evidence projection."""

from __future__ import annotations

from chatbot.knowledge.domain.pipeline import DomainPipeline
from chatbot.knowledge.extraction.pipeline import ExtractionPipeline
from chatbot.knowledge.validation.pipeline import ValidationPipeline

from .models import EvidenceProcessResult
from .projector import EvidenceProjector


class EvidenceProvenancePipeline:
    """Run the existing Knowledge Core and project its results into provenance."""

    def __init__(
        self,
        *,
        use_llm: bool = True,
        extraction_pipeline: ExtractionPipeline | None = None,
        validation_pipeline: ValidationPipeline | None = None,
        domain_pipeline: DomainPipeline | None = None,
        projector: EvidenceProjector | None = None,
    ):
        self.extraction = extraction_pipeline or ExtractionPipeline(
            use_llm=use_llm
        )
        self.validation = validation_pipeline or ValidationPipeline(
            use_llm=use_llm
        )
        self.domain = domain_pipeline or DomainPipeline()
        self.projector = projector or EvidenceProjector()

    def process(
        self,
        raw_text: str,
        doc_id: str,
        *,
        source_uri: str | None = None,
    ) -> EvidenceProcessResult:
        extraction = self.extraction.process(
            raw_text=raw_text,
            doc_id=doc_id,
            source_uri=source_uri,
        )
        fragment_texts = {
            fragment.fragment_id: fragment.text
            for fragment in extraction.fragments
        }
        validation_results = self.validation.validate_batch(
            extraction.raw_edges,
            extraction.resolved_entities,
            fragment_texts=fragment_texts,
        )
        validation_by_edge = {
            result.edge_id: result
            for result in validation_results
        }
        domain_results = self.domain.process_batch(
            extraction.raw_edges,
            validation_by_edge,
            extraction.resolved_entities,
        )
        projection = self.projector.project(
            extraction,
            validation_results=validation_results,
            domain_results=domain_results,
        )
        return EvidenceProcessResult(
            extraction=extraction,
            validation_results=validation_results,
            domain_results=domain_results,
            projection=projection,
        )
