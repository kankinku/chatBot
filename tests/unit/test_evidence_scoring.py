"""Evidence score aggregation contracts."""

from __future__ import annotations

from chatbot.knowledge.evidence.scoring import (
    SCORE_VERSION,
    EvidenceScoreAggregator,
    EvidenceScorePolicy,
)
from chatbot.knowledge.workspace.models import (
    NodeKind,
    SourceRef,
    WorkspaceNode,
)


def _assertion(
    node_id: str,
    source_uri: str,
    quality: float | None,
    *,
    semantic_conf: float | None = None,
    student_conf: float | None = None,
) -> WorkspaceNode:
    props = {}
    if quality is not None:
        props["combined_conf"] = quality
    if semantic_conf is not None:
        props["semantic_conf"] = semantic_conf
    if student_conf is not None:
        props["student_conf"] = student_conf
    return WorkspaceNode(
        id=node_id,
        label=node_id,
        kind=NodeKind.ASSERTION,
        content_hash=f"hash-{node_id}",
        sources=[SourceRef(path=source_uri, hash="a" * 64)],
        properties=props,
    )


def test_repeated_assertions_from_same_source_do_not_increase_score():
    scorer = EvidenceScoreAggregator()
    strong = _assertion("a1", "file://one.txt", 0.9)
    repeated = _assertion("a2", "file://one.txt", 0.7)

    single = scorer.score([strong], [])
    duplicate = scorer.score([strong, repeated], [])

    assert duplicate.score == single.score
    assert duplicate.support_strength == single.support_strength == 0.9
    assert duplicate.support_source_count == 1
    assert duplicate.support_assertion_count == 2


def test_independent_sources_raise_support_score():
    scorer = EvidenceScoreAggregator()
    one = _assertion("a1", "file://one.txt", 0.9)
    two = _assertion("a2", "file://two.txt", 0.9)

    single = scorer.score([one], [])
    diverse = scorer.score([one, two], [])

    assert diverse.support_source_count == 2
    assert diverse.support_strength == 1.8
    assert diverse.support_score > single.support_score
    assert diverse.score > single.score


def test_higher_quality_support_raises_relation_confidence():
    scorer = EvidenceScoreAggregator()
    weak = scorer.score(
        [_assertion("weak", "file://one.txt", 0.2)],
        [],
    )
    strong = scorer.score(
        [_assertion("strong", "file://one.txt", 0.9)],
        [],
    )

    assert strong.support_strength > weak.support_strength
    assert strong.score > weak.score


def test_conflicting_sources_reduce_relation_confidence():
    scorer = EvidenceScoreAggregator()
    support = _assertion("support", "file://support.txt", 0.9)
    weak_conflict = _assertion("weak-con", "file://weak.txt", 0.2)
    strong_conflict = _assertion("strong-con", "file://strong.txt", 0.9)

    clean = scorer.score([support], [])
    weak = scorer.score([support], [weak_conflict])
    strong = scorer.score([support], [strong_conflict])

    assert clean.score > weak.score > strong.score
    assert strong.conflict_source_count == 1
    assert strong.conflict_score > weak.conflict_score


def test_scoring_is_order_independent():
    scorer = EvidenceScoreAggregator()
    assertions = [
        _assertion("a1", "file://one.txt", 0.7),
        _assertion("a2", "file://two.txt", 0.8),
        _assertion("a3", "file://one.txt", 0.9),
    ]
    conflicts = [
        _assertion("c1", "file://three.txt", 0.4),
        _assertion("c2", "file://four.txt", 0.3),
    ]

    forward = scorer.score(assertions, conflicts)
    reverse = scorer.score(
        list(reversed(assertions)),
        list(reversed(conflicts)),
    )

    assert forward == reverse
    assert forward.version == SCORE_VERSION


def test_missing_combined_confidence_uses_component_average():
    scorer = EvidenceScoreAggregator()
    assertion = _assertion(
        "component",
        "file://one.txt",
        None,
        semantic_conf=0.8,
        student_conf=0.6,
    )

    assert scorer.assertion_quality(assertion) == 0.7


def test_missing_all_quality_fields_uses_policy_fallback():
    scorer = EvidenceScoreAggregator(
        EvidenceScorePolicy(fallback_quality=0.35)
    )
    assertion = _assertion("fallback", "file://one.txt", None)

    assert scorer.assertion_quality(assertion) == 0.35


def test_policy_rejects_invalid_bounds():
    try:
        EvidenceScorePolicy(minimum=0.8, maximum=0.2)
    except ValueError as exc:
        assert "score bounds" in str(exc)
    else:
        raise AssertionError("invalid score bounds must be rejected")
