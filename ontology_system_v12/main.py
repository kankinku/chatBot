"""
Ontology System - main entry point with optional replay training.
"""
import json
import logging
from pathlib import Path
import argparse
from datetime import datetime, timedelta

from src.extraction import ExtractionPipeline
from src.validation import ValidationPipeline
from src.validation.models import ValidationDestination
from src.domain import DomainPipeline
from src.personal import PersonalPipeline
from src.reasoning import ReasoningPipeline
from src.bootstrap import build_llm_client
from config.settings import get_settings

from src.shared.logging_setup import setup_logging
from src.replay.snapshot_manager import SnapshotManager
from src.replay.replay_runner import ReplayRunner
from src.learning.replay_training_orchestrator import ReplayTrainingOrchestrator

logger = logging.getLogger(__name__)


def run_pipeline(use_llm: bool, llm_client) -> None:
    # Extraction/Validation/Domain/Personal 준비
    extraction = ExtractionPipeline(llm_client=llm_client, use_llm=use_llm)
    validation = ValidationPipeline(llm_client=llm_client, use_llm=use_llm)
    domain = DomainPipeline()
    personal = PersonalPipeline(
        user_id="default_user",
        static_guard=domain.static_guard,
        dynamic_domain=domain.dynamic_update,
    )
    reasoning = ReasoningPipeline(
        domain=domain.dynamic_update,
        personal=personal.get_pkg(),
        llm_client=llm_client,
        ner=extraction.ner_student,
        resolver=extraction.entity_resolver,
    )
    snapshot_manager = SnapshotManager(
        graph_repo=domain.dynamic_update.kg_adapter._repo if hasattr(domain.dynamic_update, "kg_adapter") else None
    )

    # 샘플 데이터
    sample_path = Path(__file__).parent / "data" / "samples" / "sample_documents.json"
    if not sample_path.exists():
        print("Sample data not found. Running minimal test.")
        test_text = "금리가 인상되면 성장주는 하락할까?"
        result = extraction.process(raw_text=test_text, doc_id="TEST")
        print(f"Test: {len(result.fragments)} fragments")
        return

    with open(sample_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"\n{'='*70}")
    print("== ONTOLOGY SYSTEM - 5 SECTOR PIPELINE")
    print(f"{'='*70}")
    print(f"Documents: {len(documents)}")
    print(f"LLM Mode: {'Ollama' if use_llm else 'Rule-based'}")

    # Phase 1: Knowledge Collection
    print(f"\n{'-'*70}")
    print("[PHASE 1]: Knowledge Collection")
    print(f"{'-'*70}")

    for doc in documents[:3]:
        doc_id = doc.get("doc_id")
        text = doc.get("text", "")
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
                else:
                    if dom_result.intake_result:
                        personal.process_from_domain_rejection(
                            dom_result.intake_result, dom_result
                        )
            elif v.destination == ValidationDestination.PERSONAL_CANDIDATE:
                personal.process_from_validation(edge, v, ext.resolved_entities)

    dyn = domain.get_dynamic_domain()
    pkg = personal.get_pkg()
    print(f"\n[STATS] Knowledge Collected:")
    print(f"   Domain KG: {len(dyn.get_all_relations())} relations")
    print(f"   Personal KG: {len(pkg.get_all_relations())} relations")

    # Phase 2: Reasoning
    print(f"\n{'='*70}")
    print("[PHASE 2]: Reasoning")
    print(f"{'='*70}")
    test_queries = [
        "금리가 오르면 성장주는 어떻게 되나?",
        "유가가 급락하면 경기방어주는 오르나?",
        "미국 금리가 내리면 환율은?",
    ]
    for query in test_queries:
        print(f"\n[Q]: {query}")
        scenario = reasoning.reason(query)
        conc = scenario.conclusion
        print(f"   [A]: {conc.text}")
        print(f"   [DIR] Direction: {conc.direction}, Confidence: {conc.confidence:.2f}")
        print(f"   [PATH] Path: {conc.strongest_path}")
        print(f"   [BREAK] {len(scenario.break_conditions.conditions)} break conditions")

    return snapshot_manager, reasoning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_replay_train", action="store_true", help="리플레이 기반 정책 학습 실행")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    llm_client = build_llm_client()
    use_llm = llm_client.health_check()
    if use_llm:
        logger.info(f"[OK] Ollama connected: {settings.ollama.model_name}")
    else:
        logger.warning("[FAIL] Ollama not available, using rule-based mode")
        llm_client = None

    snapshot_manager, reasoning = run_pipeline(use_llm, llm_client)

    if args.run_replay_train:
        # Replay-based policy training
        replay_runner = ReplayRunner(snapshot_manager=snapshot_manager, reasoning_engine=reasoning)
        orchestrator = ReplayTrainingOrchestrator(replay_runner=replay_runner)
        start = datetime.now() - timedelta(days=10)
        end = datetime.now() - timedelta(days=1)
        result = orchestrator.run(start_date=start, end_date=end, query="테스트 질의", lookahead_days=5)
        print(f"[REPLAY TRAIN] avg_reward={result['avg_reward']:.3f}, active_policy={result['active_policy']}")


if __name__ == "__main__":
    main()
