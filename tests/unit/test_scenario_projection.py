"""Scenario/regime projection contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from chatbot.knowledge.evidence import EvidenceLedger, EvidenceProjection
from chatbot.knowledge.ingestion import IngestionRecord, IngestionState, IngestionStateStore
from chatbot.knowledge.projection import (
    InjectRelationAssumption,
    ProjectedRelationProvider,
    ProjectionBase,
    ProjectionBaseResolver,
    ProjectionBaseState,
    ProjectionBaseState,
    ProjectionService,
    ProjectionStore,
    RegimeRule,
    RegimeSpec,
    RelationDisableAssumption,
    RelationScaleAssumption,
    RelationSelector,
    RelationSignOverrideAssumption,
    ScenarioProjectionEngine,
    ScenarioShock,
    ScenarioSpec,
    SensitivityThreshold,
)
from chatbot.knowledge.reasoning.edge_fusion import EdgeWeightFusion
from chatbot.knowledge.reasoning.graph_retrieval import GraphRetrieval
from chatbot.knowledge.reasoning.models import ParsedQuery
from chatbot.knowledge.replay import KnowledgeReplayService, KnowledgeReplayStore
from chatbot.knowledge.replay.models import state_digest
from chatbot.knowledge.workspace.hashing import hash_value
from chatbot.knowledge.workspace.models import (
    Confidence,
    MetaRelation,
    NodeKind,
    SourceRef,
    WorkspaceEdge,
    WorkspaceGraph,
    WorkspaceNode,
)


UTC = timezone.utc


def _edge(
    source: str,
    target: str,
    relation: MetaRelation,
    ref: SourceRef,
    *,
    semantic_type: str | None = None,
) -> WorkspaceEdge:
    payload = {
        "source": source,
        "target": target,
        "relation": relation.value,
        "semantic_type": semantic_type,
    }
    return WorkspaceEdge(
        source=source,
        target=target,
        relation=relation,
        confidence=Confidence.VALIDATED,
        content_hash=hash_value(payload),
        sources=[ref],
        semantic_type=semantic_type,
    )


def _state(
    relations: tuple[tuple[str, str, str, str, float], ...] = (
        ("A", "B", "Affect", "+", 0.9),
        ("B", "C", "Affect", "-", 0.8),
        ("A", "C", "Affect", "+", 0.7),
        ("C", "A", "Affect", "+", 0.6),
        ("A", "D", "Affect", "neutral", 0.75),
    ),
) -> IngestionState:
    source_uri = "file:scenario.txt"
    source_hash = hashlib.sha256(repr(relations).encode("utf-8")).hexdigest()
    ref = SourceRef(path=source_uri, hash=source_hash)
    document_id = "document:scenario"
    nodes = [
        WorkspaceNode(
            id=document_id,
            label="scenario",
            kind=NodeKind.DOCUMENT,
            content_hash=source_hash,
            sources=[ref],
            properties={"doc_id": "scenario", "source_uri": source_uri},
        )
    ]
    edges = []

    entity_ids = sorted(
        {item for relation in relations for item in relation[:2]}
    )
    for entity_id in entity_ids:
        props = {"stable_key": entity_id, "canonical_name": entity_id}
        nodes.append(
            WorkspaceNode(
                id=f"entity:{entity_id}",
                label=entity_id,
                kind=NodeKind.ENTITY,
                content_hash=hash_value(props),
                sources=[ref],
                properties=props,
            )
        )

    for index, (head, tail, relation_type, sign, quality) in enumerate(relations):
        relation_props = {
            "head": head,
            "tail": tail,
            "relation_type": relation_type,
        }
        relation_id = "domain-relation:" + hash_value(relation_props)[:24]
        assertion_props = {
            "relation_type": relation_type,
            "head": head,
            "tail": tail,
            "polarity_final": sign,
            "combined_conf": quality,
            "semantic_tag": "sem_confident",
        }
        assertion_id = f"assertion:{index:02d}"
        nodes.extend(
            [
                WorkspaceNode(
                    id=relation_id,
                    label=f"{head} {relation_type} {tail}",
                    kind=NodeKind.DOMAIN_RELATION,
                    content_hash=hash_value(relation_props),
                    sources=[ref],
                    properties=relation_props,
                ),
                WorkspaceNode(
                    id=assertion_id,
                    label=f"assert {index}",
                    kind=NodeKind.ASSERTION,
                    content_hash=hash_value(assertion_props),
                    sources=[ref],
                    properties=assertion_props,
                ),
            ]
        )
        edges.append(
            _edge(
                relation_id,
                assertion_id,
                MetaRelation.SUPPORTED_BY,
                ref,
            )
        )

    graph = WorkspaceGraph(nodes=nodes, edges=edges)
    projection = EvidenceProjection(
        source_uri=source_uri,
        source_hash=source_hash,
        document_node_id=document_id,
        projection_hash=hash_value(graph.to_dict()),
        graph=graph,
    )
    ledger = EvidenceLedger()
    ledger.upsert(projection)
    record = IngestionRecord(
        doc_id="scenario",
        source_uri=source_uri,
        source_hash=source_hash,
        projection_hash=projection.projection_hash,
        processor_stamp="a" * 64,
    )
    state = IngestionState(records={source_uri: record}, ledger=ledger)
    state.validate()
    return state


def _base(state: IngestionState | None = None) -> ProjectionBaseState:
    state = state or _state()
    return ProjectionBaseState(
        metadata=ProjectionBase(
            state_digest=state_digest(state),
            origin="current_state",
        ),
        _state=state,
    )


def _relation(projection, head: str, tail: str, relation_type: str = "Affect"):
    return next(
        item
        for item in projection.relations
        if item.stable_key == (head, tail, relation_type)
    )


def test_deterministic_identity_and_labels_do_not_change_semantics():
    base = _base()
    scenario_a = ScenarioSpec(
        assumptions=(
            RelationScaleAssumption(
                RelationSelector(head_id="A", tail_id="B", relation_type="Affect"),
                0.5,
            ),
        ),
        label="A",
        description="first label",
    )
    scenario_b = ScenarioSpec(
        assumptions=scenario_a.assumptions,
        label="B",
        description="different text",
    )
    regime_a = RegimeSpec(
        rules=(RegimeRule(RelationSelector(relation_type="Affect"), 0.8),),
        label="one",
    )
    regime_b = RegimeSpec(rules=regime_a.rules, label="two")

    first = ScenarioProjectionEngine().project(base, scenario_a, regime_a)
    second = ScenarioProjectionEngine().project(base, scenario_b, regime_b)

    assert scenario_a.scenario_spec_id == scenario_b.scenario_spec_id
    assert regime_a.regime_spec_id == regime_b.regime_spec_id
    assert first.projection_id == second.projection_id
    assert first.to_dict() == second.to_dict()


def test_base_or_engine_change_changes_projection_identity():
    scenario = ScenarioSpec()
    first = ScenarioProjectionEngine().project(_base(_state()), scenario)
    changed = ScenarioProjectionEngine().project(
        _base(_state((("A", "B", "Affect", "+", 0.2),))),
        scenario,
    )
    versioned = ScenarioProjectionEngine(
        engine_version="scenario-projection-v2"
    ).project(_base(_state()), scenario)

    assert first.projection_id != changed.projection_id
    assert first.projection_id != versioned.projection_id


def test_regime_and_scenario_modify_projected_weight_not_evidence_score():
    base = _base()
    selector = RelationSelector(head_id="A", tail_id="B", relation_type="Affect")
    scenario = ScenarioSpec(
        assumptions=(RelationScaleAssumption(selector, 0.5),)
    )
    regime = RegimeSpec(rules=(RegimeRule(selector, 0.5),))

    baseline = ScenarioProjectionEngine().project(base, ScenarioSpec())
    projected = ScenarioProjectionEngine().project(base, scenario, regime)
    before = _relation(baseline, "A", "B")
    after = _relation(projected, "A", "B")

    assert before.evidence_score == after.evidence_score
    assert before.base_weight == after.base_weight
    assert after.projected_weight == pytest.approx(before.base_weight * 0.25)
    assert after.scenario_multiplier == 0.5
    assert after.regime_multiplier == 0.5


def test_multiplier_product_is_bounded_and_disable_is_explainable():
    selector = RelationSelector(relation_type="Affect")
    scenario = ScenarioSpec(
        assumptions=(
            RelationScaleAssumption(selector, 2.0),
            RelationScaleAssumption(
                RelationSelector(head_id="A", tail_id="B", relation_type="Affect"),
                1.5,
            ),
            RelationDisableAssumption(
                RelationSelector(head_id="B", tail_id="C", relation_type="Affect")
            ),
        )
    )
    regime = RegimeSpec(
        rules=(
            RegimeRule(selector, 2.0),
            RegimeRule(
                RelationSelector(head_id="A", tail_id="B", relation_type="Affect"),
                1.5,
            ),
        )
    )
    projection = ScenarioProjectionEngine().project(_base(), scenario, regime)
    ab = _relation(projection, "A", "B")
    bc = _relation(projection, "B", "C")

    assert ab.scenario_multiplier == 2.0
    assert ab.regime_multiplier == 2.0
    assert ab.projected_weight <= 1.0
    assert bc.active is False
    assert bc.projected_weight == 0.0


def test_sign_override_preserves_base_sign_and_conflicts_fail_closed():
    selector = RelationSelector(head_id="B", tail_id="C", relation_type="Affect")
    projection = ScenarioProjectionEngine().project(
        _base(),
        ScenarioSpec(
            assumptions=(RelationSignOverrideAssumption(selector, "+"),)
        ),
    )
    relation = _relation(projection, "B", "C")

    assert relation.base_sign == "-"
    assert relation.projected_sign == "+"

    with pytest.raises(ValueError, match="conflicting sign overrides"):
        ScenarioProjectionEngine().project(
            _base(),
            ScenarioSpec(
                assumptions=(
                    RelationSignOverrideAssumption(selector, "+"),
                    RelationSignOverrideAssumption(selector, "-"),
                )
            ),
        )


def test_duplicate_inputs_fail_closed():
    selector = RelationSelector(relation_type="Affect")
    assumption = RelationScaleAssumption(selector, 0.5)
    shock = ScenarioShock("A", "+", 1.0)
    rule = RegimeRule(selector, 0.5)

    with pytest.raises(ValueError, match="duplicate scenario assumption"):
        ScenarioSpec(assumptions=(assumption, assumption))
    with pytest.raises(ValueError, match="duplicate scenario shock"):
        ScenarioSpec(shocks=(shock, shock))
    with pytest.raises(ValueError, match="duplicate regime rule"):
        RegimeSpec(rules=(rule, rule))


def test_injected_relation_requires_existing_entities_and_no_collision():
    engine = ScenarioProjectionEngine()
    base = _base()

    with pytest.raises(ValueError, match="existing base entities"):
        engine.project(
            base,
            ScenarioSpec(
                assumptions=(
                    InjectRelationAssumption("A", "UNKNOWN", "Cause", "+", 0.5),
                )
            ),
        )

    with pytest.raises(ValueError, match="collides"):
        engine.project(
            base,
            ScenarioSpec(
                assumptions=(
                    InjectRelationAssumption("A", "B", "Affect", "+", 0.5),
                )
            ),
        )

    projection = engine.project(
        base,
        ScenarioSpec(
            assumptions=(
                InjectRelationAssumption("B", "D", "Cause", "+", 0.4),
            )
        ),
    )
    injected = _relation(projection, "B", "D", "Cause")
    assert injected.origin == "hypothesis"
    assert injected.evidence_score is None
    assert injected.source_refs == ()
    assert injected.base_weight == 0.4


def test_shock_sign_and_path_impact_use_projected_graph_weights():
    projection = ScenarioProjectionEngine().project(
        _base(),
        ScenarioSpec(
            shocks=(ScenarioShock("A", "+", 0.8),),
            max_depth=2,
            max_paths=50,
        ),
    )
    ab = _relation(projection, "A", "B")
    bc = _relation(projection, "B", "C")
    two_hop = next(
        item
        for item in projection.impacts
        if item.path_entities == ("A", "B", "C")
    )

    assert two_hop.direction == "-"
    assert two_hop.impact_value == pytest.approx(
        0.8 * ab.projected_weight * bc.projected_weight
    )


def test_neutral_edge_does_not_propagate_until_sign_override():
    shock = ScenarioShock("A", "+", 1.0)
    base = _base()

    no_override = ScenarioProjectionEngine().project(
        base,
        ScenarioSpec(shocks=(shock,), max_depth=1),
    )
    assert not any(item.target_entity_id == "D" for item in no_override.impacts)

    override = ScenarioProjectionEngine().project(
        base,
        ScenarioSpec(
            assumptions=(
                RelationSignOverrideAssumption(
                    RelationSelector(
                        head_id="A",
                        tail_id="D",
                        relation_type="Affect",
                    ),
                    "+",
                ),
            ),
            shocks=(shock,),
            max_depth=1,
        ),
    )
    assert any(item.target_entity_id == "D" for item in override.impacts)


def test_cycles_depth_and_path_caps_are_enforced():
    base = _base()
    shock = ScenarioShock("A", "+", 1.0)
    depth_one = ScenarioProjectionEngine().project(
        base,
        ScenarioSpec(shocks=(shock,), max_depth=1, max_paths=100),
    )
    assert all(len(item.path_relation_ids) == 1 for item in depth_one.impacts)

    capped = ScenarioProjectionEngine().project(
        base,
        ScenarioSpec(shocks=(shock,), max_depth=5, max_paths=2),
    )
    assert len(capped.impacts) == 2
    assert all(
        len(item.path_entities) == len(set(item.path_entities))
        for item in capped.impacts
    )


def test_node_summary_uses_strongest_paths_not_sum():
    projection = ScenarioProjectionEngine().project(
        _base(),
        ScenarioSpec(
            shocks=(ScenarioShock("A", "+", 1.0),),
            max_depth=2,
            max_paths=100,
        ),
    )
    c_impacts = [
        item.impact_value
        for item in projection.impacts
        if item.target_entity_id == "C" and item.direction == "+"
    ]
    summary = next(
        item for item in projection.node_summaries if item.entity_id == "C"
    )
    assert summary.strongest_positive == max(c_impacts, default=0.0)
    assert summary.strongest_positive != pytest.approx(sum(c_impacts)) if len(c_impacts) > 1 else True


def test_explicit_sensitivity_only_evaluates_supplied_thresholds():
    selector = RelationSelector(head_id="A", tail_id="B", relation_type="Affect")
    projection = ScenarioProjectionEngine().project(
        _base(),
        ScenarioSpec(
            sensitivity_thresholds=(
                SensitivityThreshold(selector, "<", 0.9),
            )
        ),
    )
    assert len(projection.sensitivity) == 1
    item = projection.sensitivity[0]
    assert item.threshold == 0.9
    assert item.triggered == (item.projected_weight < 0.9)

    no_threshold = ScenarioProjectionEngine().project(_base(), ScenarioSpec())
    assert no_threshold.sensitivity == ()


def test_projection_does_not_mutate_canonical_state():
    state = _state()
    before = json.dumps(state.to_dict(), sort_keys=True)
    ScenarioProjectionEngine().project(
        _base(state),
        ScenarioSpec(
            assumptions=(
                RelationDisableAssumption(RelationSelector(relation_type="Affect")),
            ),
            shocks=(ScenarioShock("A", "+", 1.0),),
        ),
    )
    assert json.dumps(state.to_dict(), sort_keys=True) == before


def test_snapshot_and_as_of_resolve_exact_historical_state(tmp_path: Path):
    replay_store = KnowledgeReplayStore(tmp_path / "replay")
    t1 = datetime(2026, 9, 22, 1, 0, tzinfo=UTC)
    first = replay_store.record(
        _state((("A", "B", "Affect", "+", 0.9),)),
        committed_at=t1,
        origin="bootstrap",
    )
    replay_store.record(
        _state((("A", "B", "Affect", "-", 0.9),)),
        committed_at=t1 + timedelta(hours=1),
        origin="ingestion_commit",
    )
    current_store = IngestionStateStore(tmp_path / "current.json")
    current_store.save(_state((("A", "B", "Affect", "-", 0.9),)))
    service = ProjectionService(
        base_resolver=ProjectionBaseResolver(
            replay_service=KnowledgeReplayService(replay_store),
            current_state_store=current_store,
        )
    )

    historical = service.project_snapshot(first.snapshot_id, ScenarioSpec())
    as_of = service.project_as_of(t1 + timedelta(minutes=30), ScenarioSpec())

    assert _relation(historical, "A", "B").base_sign == "+"
    assert _relation(as_of, "A", "B").base_sign == "+"
    assert historical.base.snapshot_id == first.snapshot_id


def test_as_of_before_first_snapshot_is_unavailable(tmp_path: Path):
    replay_store = KnowledgeReplayStore(tmp_path / "replay")
    t1 = datetime(2026, 9, 22, 1, 0, tzinfo=UTC)
    replay_store.record(_state(), committed_at=t1, origin="bootstrap")
    service = ProjectionService(
        base_resolver=ProjectionBaseResolver(
            replay_service=KnowledgeReplayService(replay_store),
            current_state_store=IngestionStateStore(tmp_path / "state.json"),
        )
    )
    with pytest.raises(LookupError, match="no replay snapshot"):
        service.project_as_of(t1 - timedelta(seconds=1), ScenarioSpec())


def test_current_state_normalizes_to_latest_snapshot_when_digest_matches(tmp_path: Path):
    state = _state()
    replay_store = KnowledgeReplayStore(tmp_path / "replay")
    snapshot = replay_store.record(
        state,
        committed_at=datetime(2026, 9, 22, 1, 0, tzinfo=UTC),
        origin="bootstrap",
    )
    current_store = IngestionStateStore(tmp_path / "state.json")
    current_store.save(state)
    resolver = ProjectionBaseResolver(
        replay_service=KnowledgeReplayService(replay_store),
        current_state_store=current_store,
    )
    current = resolver.current()

    assert current.metadata.origin == "replay_snapshot"
    assert current.metadata.snapshot_id == snapshot.snapshot_id


def test_projection_store_is_idempotent_and_detects_corruption(tmp_path: Path):
    state = _state()
    replay_store = KnowledgeReplayStore(tmp_path / "replay")
    snapshot = replay_store.record(
        state,
        committed_at=datetime(2026, 9, 22, 1, 0, tzinfo=UTC),
        origin="bootstrap",
    )
    resolver = ProjectionBaseResolver(
        replay_service=KnowledgeReplayService(replay_store),
        current_state_store=IngestionStateStore(tmp_path / "state.json"),
    )
    projection = ScenarioProjectionEngine().project(
        resolver.snapshot(snapshot.snapshot_id),
        ScenarioSpec(),
    )
    store = ProjectionStore(tmp_path / "projections")
    first = store.save(projection)
    second = store.save(projection)

    assert first == second
    assert store.verify(projection.projection_id)["valid"] is True

    value = json.loads(first.read_text(encoding="utf-8"))
    value["relations"][0]["projected_weight"] = 0.123
    first.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="output digest mismatch"):
        store.load(projection.projection_id)


def test_transient_current_projection_cannot_be_persisted(tmp_path: Path):
    projection = ScenarioProjectionEngine().project(_base(), ScenarioSpec())
    with pytest.raises(ValueError, match="snapshot-backed"):
        ProjectionStore(tmp_path / "projections").save(projection)


def test_projected_relation_provider_integrates_with_graph_retrieval():
    projection = ScenarioProjectionEngine().project(_base(), ScenarioSpec())
    retrieval = GraphRetrieval(
        domain=ProjectedRelationProvider(projection),
        max_path_length=3,
        max_paths=10,
    )
    result = retrieval.retrieve(
        ParsedQuery(
            original_query="A to C",
            query_entities=["A", "C"],
            entity_names={"A": "A", "B": "B", "C": "C"},
            head_entity="A",
            tail_entity="C",
        )
    )

    assert result.domain_paths_count >= 2
    assert any(path.nodes == ["A", "C"] for path in result.direct_paths)
    assert any(path.nodes == ["A", "B", "C"] for path in result.indirect_paths)


def test_projection_package_has_no_vector_retrieval_dependency():
    from chatbot.knowledge.projection import engine as projection_engine

    source = Path(projection_engine.__file__).read_text(encoding="utf-8")
    assert "chromadb" not in source
    assert "VectorRetriever" not in source


def test_projection_store_rejects_semantic_identity_tampering(tmp_path: Path):
    state = _state()
    replay_store = KnowledgeReplayStore(tmp_path / "replay")
    snapshot = replay_store.record(
        state,
        committed_at=datetime(2026, 9, 22, 1, 0, tzinfo=UTC),
        origin="bootstrap",
    )
    resolver = ProjectionBaseResolver(
        replay_service=KnowledgeReplayService(replay_store),
        current_state_store=IngestionStateStore(tmp_path / "state.json"),
    )
    projection = ScenarioProjectionEngine().project(
        resolver.snapshot(snapshot.snapshot_id),
        ScenarioSpec(),
    )
    store = ProjectionStore(tmp_path / "projections")
    path = store.save(projection)

    value = json.loads(path.read_text(encoding="utf-8"))
    value["base"]["state_digest"] = "0" * 64
    value["trace"]["base"]["state_digest"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic identity mismatch"):
        store.load(projection.projection_id)


def test_projected_relation_provider_does_not_double_count_evidence_bonus():
    projection = ScenarioProjectionEngine().project(_base(), ScenarioSpec())
    expected = _relation(projection, "A", "C")
    retrieval = GraphRetrieval(
        domain=ProjectedRelationProvider(projection),
        max_path_length=2,
        max_paths=10,
    )
    result = retrieval.retrieve(
        ParsedQuery(
            original_query="A to C",
            query_entities=["A", "C"],
            entity_names={"A": "A", "C": "C"},
            head_entity="A",
            tail_entity="C",
        )
    )
    direct = next(path for path in result.direct_paths if path.nodes == ["A", "C"])
    fused = EdgeWeightFusion().fuse_path(direct)

    assert fused.fused_edges[0].evidence_count == 0
    assert fused.fused_edges[0].final_weight == pytest.approx(
        expected.projected_weight
    )


def test_projection_base_state_rejects_mismatched_digest():
    state = _state()
    with pytest.raises(ValueError, match="projection base state digest mismatch"):
        ProjectionBaseState(
            metadata=ProjectionBase(
                state_digest="0" * 64,
                origin="current_state",
            ),
            _state=state,
        )


def test_projection_base_state_freezes_defensive_state_copy():
    state = _state()
    base = ProjectionBaseState(
        metadata=ProjectionBase(
            state_digest=state_digest(state),
            origin="current_state",
        ),
        _state=state,
    )
    state.records.clear()

    assert base.state.records
    projection = ScenarioProjectionEngine().project(base, ScenarioSpec())
    assert projection.relations


def test_projection_store_detects_evidence_version_trace_tampering(tmp_path: Path):
    state = _state()
    replay_store = KnowledgeReplayStore(tmp_path / "replay")
    snapshot = replay_store.record(
        state,
        committed_at=datetime(2026, 9, 22, 1, 0, tzinfo=UTC),
        origin="bootstrap",
    )
    resolver = ProjectionBaseResolver(
        replay_service=KnowledgeReplayService(replay_store),
        current_state_store=IngestionStateStore(tmp_path / "state.json"),
    )
    projection = ScenarioProjectionEngine().project(
        resolver.snapshot(snapshot.snapshot_id),
        ScenarioSpec(),
    )
    store = ProjectionStore(tmp_path / "projections")
    path = store.save(projection)

    value = json.loads(path.read_text(encoding="utf-8"))
    value["trace"]["evidence_score_versions"] = ["tampered-version"]
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="projection evidence score versions mismatch",
    ):
        store.load(projection.projection_id)
