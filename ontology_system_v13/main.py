"""Ontology System - main entry point."""
import json
import logging
from pathlib import Path
import argparse

from src.extraction import ExtractionPipeline
from src.validation import ValidationPipeline
from src.validation.models import ValidationDestination
from src.domain import DomainPipeline
from src.reasoning import ReasoningPipeline
from src.bootstrap import build_llm_client
from config.settings import get_settings
from src.shared.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def run_pipeline(use_llm: bool, llm_client):
    extraction = ExtractionPipeline(llm_client=llm_client, use_llm=use_llm)
    validation = ValidationPipeline(llm_client=llm_client, use_llm=use_llm)
    domain = DomainPipeline()
    reasoning = ReasoningPipeline(
        domain=domain.dynamic_update,
        llm_client=llm_client,
        ner=extraction.ner_student,
        resolver=extraction.entity_resolver,
    )

    sample_path = Path(__file__).parent / "data" / "samples" / "sample_documents.json"
    if not sample_path.exists():
        print("Sample data not found. Exiting.")
        return reasoning

    with open(sample_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    if not documents:
        print("No sample documents. Add documents to data/samples/sample_documents.json.")
        return reasoning

    print("\n" + "=" * 70)
    print("== ONTOLOGY SYSTEM - CORE PIPELINE")
    print("=" * 70)
    print(f"Documents: {len(documents)}")
    print(f"LLM Mode: {'Ollama' if use_llm else 'Rule-based'}")

    print("\n" + "-" * 70)
    print("[PHASE 1]: Knowledge Collection")
    print("-" * 70)

    for doc in documents:
        doc_id = doc.get("doc_id")
        text = doc.get("text", "")
        if not text:
            continue
        print(f"\n[DOC] {doc_id}: {text[:40]}...")
        ext = extraction.process(raw_text=text, doc_id=doc_id)
        if not ext.raw_edges:
            continue
        vals = validation.validate_batch(
            edges=ext.raw_edges,
            resolved_entities=ext.resolved_entities,
        )
        val_map = {v.edge_id: v for v in vals}
        for edge in ext.raw_edges:
            v = val_map.get(edge.raw_edge_id)
            if not v or not v.validation_passed:
                continue
            if v.destination == ValidationDestination.DOMAIN_CANDIDATE:
                dom_result = domain.process(edge, v, ext.resolved_entities)
                if dom_result.final_destination == "domain":
                    print(f"   [DOMAIN] {edge.head_canonical_name} -> {edge.tail_canonical_name}")

    dyn = domain.get_dynamic_domain()
    print("\n[STATS] Knowledge Collected:")
    print(f"   Domain KG: {len(dyn.get_all_relations())} relations")

    return reasoning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_llm", action="store_true", help="Disable LLM usage")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    llm_client = build_llm_client()
    use_llm = False if args.no_llm else llm_client.health_check()
    if use_llm:
        logger.info(f"[OK] Ollama connected: {settings.ollama.model_name}")
    else:
        logger.warning("[WARN] LLM not available, using rule-based mode")
        llm_client = None

    run_pipeline(use_llm, llm_client)


if __name__ == "__main__":
    main()
