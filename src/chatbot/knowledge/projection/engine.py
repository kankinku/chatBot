"""Pure deterministic scenario/regime projection engine."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from chatbot.knowledge.evidence import RelationEvidenceViewBuilder
from chatbot.knowledge.workspace.hashing import hash_value
from chatbot.knowledge.workspace.models import NodeKind

from .base_view import ProjectionBaseState
from .models import (
    PROJECTION_ENGINE_VERSION,
    InjectRelationAssumption,
    NodeImpactSummary,
    ProjectedImpact,
    ProjectedRelation,
    ProjectionDependency,
    ProjectionTrace,
    RegimeSpec,
    RelationDisableAssumption,
    RelationScaleAssumption,
    RelationSignOverrideAssumption,
    ScenarioProjection,
    ScenarioSpec,
    SensitivityResult,
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _relation_id(head_id: str, tail_id: str, relation_type: str) -> str:
    return "prel_" + hash_value(
        {
            "head_id": head_id,
            "tail_id": tail_id,
            "relation_type": relation_type,
        }
    )[:24]


def _combine_sign(left: str, right: str) -> str:
    if left not in {"+", "-"} or right not in {"+", "-"}:
        return "unknown"
    return "+" if left == right else "-"


@dataclass(frozen=True)
class _RelationSeed:
    relation_id: str
    head_id: str
    head_name: str
    tail_id: str
    tail_name: str
    relation_type: str
    origin: str
    base_relation_node_id: str | None
    base_sign: str | None
    initial_sign: str
    evidence_score: object | None
    base_weight: float
    support_assertion_ids: tuple[str, ...]
    conflict_assertion_ids: tuple[str, ...]
    source_refs: tuple


class ScenarioProjectionEngine:
    def __init__(
        self,
        *,
        engine_version: str = PROJECTION_ENGINE_VERSION,
        view_builder: RelationEvidenceViewBuilder | None = None,
    ):
        if not engine_version:
            raise ValueError("projection engine version must not be empty")
        self.engine_version = engine_version
        self.view_builder = view_builder or RelationEvidenceViewBuilder()

    def project(
        self,
        base: ProjectionBaseState,
        scenario: ScenarioSpec,
        regime: RegimeSpec | None = None,
    ) -> ScenarioProjection:
        graph = base.merged_graph()
        evidence_relations = [
            item
            for item in self.view_builder.build(graph)
            if item.support_count > 0
        ]
        entity_ids, entity_names = self._entity_catalog(graph, evidence_relations)
        seeds = [
            _RelationSeed(
                relation_id=_relation_id(
                    item.head_id,
                    item.tail_id,
                    item.relation_type,
                ),
                head_id=item.head_id,
                head_name=item.head_name,
                tail_id=item.tail_id,
                tail_name=item.tail_name,
                relation_type=item.relation_type,
                origin="evidence",
                base_relation_node_id=item.relation_node_id,
                base_sign=item.sign,
                initial_sign=item.sign,
                evidence_score=item.score,
                base_weight=item.score.score,
                support_assertion_ids=item.support_assertion_ids,
                conflict_assertion_ids=item.conflict_assertion_ids,
                source_refs=item.source_refs,
            )
            for item in evidence_relations
        ]

        existing_keys = {
            (seed.head_id, seed.tail_id, seed.relation_type)
            for seed in seeds
        }
        injected_keys: set[tuple[str, str, str]] = set()
        dependencies: list[ProjectionDependency] = []

        for assumption in scenario.assumptions:
            if not isinstance(assumption, InjectRelationAssumption):
                continue
            if assumption.head_id not in entity_ids or assumption.tail_id not in entity_ids:
                raise ValueError(
                    "injected relations may reference only existing base entities"
                )
            if assumption.stable_key in existing_keys:
                raise ValueError(
                    "injected relation collides with an evidence-backed relation"
                )
            if assumption.stable_key in injected_keys:
                raise ValueError("duplicate injected relation stable key")
            injected_keys.add(assumption.stable_key)
            relation_id = _relation_id(*assumption.stable_key)
            seeds.append(
                _RelationSeed(
                    relation_id=relation_id,
                    head_id=assumption.head_id,
                    head_name=entity_names.get(
                        assumption.head_id,
                        assumption.head_id,
                    ),
                    tail_id=assumption.tail_id,
                    tail_name=entity_names.get(
                        assumption.tail_id,
                        assumption.tail_id,
                    ),
                    relation_type=assumption.relation_type,
                    origin="hypothesis",
                    base_relation_node_id=None,
                    base_sign=None,
                    initial_sign=assumption.sign,
                    evidence_score=None,
                    base_weight=assumption.assumed_weight,
                    support_assertion_ids=(),
                    conflict_assertion_ids=(),
                    source_refs=(),
                )
            )
            dependencies.append(
                ProjectionDependency(
                    kind="hypothetical_relation_injected",
                    relation_id=relation_id,
                    input_id=assumption.assumption_id,
                    detail="hypothetical relation injected without evidence mutation",
                )
            )

        projected: list[ProjectedRelation] = []
        matched_assumptions: set[str] = set()
        matched_rules: set[str] = set()

        for seed in sorted(seeds, key=lambda item: item.relation_id):
            scenario_multiplier = 1.0
            regime_multiplier = 1.0
            active = True
            projected_sign = seed.initial_sign
            applied_assumptions: list[str] = []
            applied_rules: list[str] = []
            sign_overrides: dict[str, str] = {}

            for assumption in scenario.assumptions:
                if isinstance(assumption, InjectRelationAssumption):
                    if seed.origin == "hypothesis" and (
                        seed.head_id,
                        seed.tail_id,
                        seed.relation_type,
                    ) == assumption.stable_key:
                        applied_assumptions.append(assumption.assumption_id)
                        matched_assumptions.add(assumption.assumption_id)
                    continue

                if not assumption.selector.matches(
                    relation_node_id=seed.base_relation_node_id,
                    head_id=seed.head_id,
                    tail_id=seed.tail_id,
                    relation_type=seed.relation_type,
                ):
                    continue

                matched_assumptions.add(assumption.assumption_id)
                applied_assumptions.append(assumption.assumption_id)
                if isinstance(assumption, RelationScaleAssumption):
                    scenario_multiplier *= assumption.multiplier
                    dependencies.append(
                        ProjectionDependency(
                            kind="scenario_multiplier",
                            relation_id=seed.relation_id,
                            input_id=assumption.assumption_id,
                            detail=f"multiplier={assumption.multiplier}",
                        )
                    )
                elif isinstance(assumption, RelationDisableAssumption):
                    active = False
                    dependencies.append(
                        ProjectionDependency(
                            kind="relation_disabled",
                            relation_id=seed.relation_id,
                            input_id=assumption.assumption_id,
                            detail="relation disabled by scenario assumption",
                        )
                    )
                elif isinstance(assumption, RelationSignOverrideAssumption):
                    sign_overrides[assumption.assumption_id] = assumption.sign

            if len(set(sign_overrides.values())) > 1:
                raise ValueError(
                    f"conflicting sign overrides for relation {seed.relation_id}"
                )
            if sign_overrides:
                projected_sign = next(iter(sign_overrides.values()))
                for assumption_id in sorted(sign_overrides):
                    dependencies.append(
                        ProjectionDependency(
                            kind="relation_sign_override",
                            relation_id=seed.relation_id,
                            input_id=assumption_id,
                            detail=f"projected_sign={projected_sign}",
                        )
                    )

            if regime is not None:
                for rule in regime.rules:
                    if not rule.selector.matches(
                        relation_node_id=seed.base_relation_node_id,
                        head_id=seed.head_id,
                        tail_id=seed.tail_id,
                        relation_type=seed.relation_type,
                    ):
                        continue
                    matched_rules.add(rule.rule_id)
                    applied_rules.append(rule.rule_id)
                    regime_multiplier *= rule.multiplier
                    dependencies.append(
                        ProjectionDependency(
                            kind="regime_applicability",
                            relation_id=seed.relation_id,
                            input_id=rule.rule_id,
                            detail=f"multiplier={rule.multiplier}",
                        )
                    )

            scenario_multiplier = _clamp(scenario_multiplier, 0.0, 2.0)
            regime_multiplier = _clamp(regime_multiplier, 0.0, 2.0)
            weight = (
                _clamp(
                    seed.base_weight
                    * regime_multiplier
                    * scenario_multiplier,
                    0.0,
                    1.0,
                )
                if active
                else 0.0
            )
            projected.append(
                ProjectedRelation(
                    relation_id=seed.relation_id,
                    head_id=seed.head_id,
                    head_name=seed.head_name,
                    tail_id=seed.tail_id,
                    tail_name=seed.tail_name,
                    relation_type=seed.relation_type,
                    origin=seed.origin,
                    base_relation_node_id=seed.base_relation_node_id,
                    base_sign=seed.base_sign,
                    projected_sign=projected_sign,
                    evidence_score=seed.evidence_score,
                    base_weight=round(seed.base_weight, 12),
                    regime_multiplier=round(regime_multiplier, 12),
                    scenario_multiplier=round(scenario_multiplier, 12),
                    active=active,
                    projected_weight=round(weight, 12),
                    applied_regime_rule_ids=tuple(sorted(applied_rules)),
                    applied_assumption_ids=tuple(sorted(applied_assumptions)),
                    support_assertion_ids=seed.support_assertion_ids,
                    conflict_assertion_ids=seed.conflict_assertion_ids,
                    source_refs=seed.source_refs,
                )
            )

        warnings = []
        for assumption in scenario.assumptions:
            if assumption.assumption_id not in matched_assumptions:
                warnings.append(
                    f"scenario assumption matched no relation: {assumption.assumption_id}"
                )
        if regime is not None:
            for rule in regime.rules:
                if rule.rule_id not in matched_rules:
                    warnings.append(
                        f"regime rule matched no relation: {rule.rule_id}"
                    )

        impacts, shock_dependencies, shock_warnings = self._propagate(
            projected,
            scenario,
            entity_ids,
        )
        dependencies.extend(shock_dependencies)
        warnings.extend(shock_warnings)
        node_summaries = self._summaries(impacts)
        sensitivity = self._sensitivity(projected, scenario)

        projection_id = "proj_" + hash_value(
            {
                "engine_version": self.engine_version,
                "base": base.metadata.identity_dict(),
                "scenario_spec_id": scenario.scenario_spec_id,
                "regime_spec_id": (
                    regime.regime_spec_id if regime is not None else None
                ),
            }
        )
        dependencies_tuple = tuple(
            sorted(
                dependencies,
                key=lambda item: (
                    item.kind,
                    item.relation_id or "",
                    item.input_id,
                    item.detail,
                ),
            )
        )
        warnings_tuple = tuple(sorted(set(warnings)))
        relations_tuple = tuple(
            sorted(projected, key=lambda item: item.relation_id)
        )
        impacts_tuple = tuple(
            sorted(
                impacts,
                key=lambda item: (
                    item.shock_id,
                    item.path_entities,
                    item.path_relation_ids,
                ),
            )
        )
        summaries_tuple = tuple(
            sorted(node_summaries, key=lambda item: item.entity_id)
        )
        sensitivity_tuple = tuple(
            sorted(
                sensitivity,
                key=lambda item: (item.threshold_id, item.relation_id),
            )
        )
        core = {
            "projection_id": projection_id,
            "engine_version": self.engine_version,
            "base": base.metadata.to_dict(),
            "scenario_spec_id": scenario.scenario_spec_id,
            "regime_spec_id": (
                regime.regime_spec_id if regime is not None else None
            ),
            "relations": [item.to_dict() for item in relations_tuple],
            "impacts": [item.to_dict() for item in impacts_tuple],
            "node_summaries": [item.to_dict() for item in summaries_tuple],
            "sensitivity": [item.to_dict() for item in sensitivity_tuple],
            "dependencies": [item.to_dict() for item in dependencies_tuple],
            "warnings": list(warnings_tuple),
        }
        output_digest = hash_value(core)
        evidence_versions = tuple(
            sorted(
                {
                    relation.evidence_score.version
                    for relation in relations_tuple
                    if relation.evidence_score is not None
                }
            )
        )
        trace = ProjectionTrace(
            projection_id=projection_id,
            engine_version=self.engine_version,
            base=base.metadata,
            scenario_spec_id=scenario.scenario_spec_id,
            regime_spec_id=(
                regime.regime_spec_id if regime is not None else None
            ),
            evidence_score_versions=evidence_versions,
            dependencies=dependencies_tuple,
            warnings=warnings_tuple,
            output_digest=output_digest,
        )
        return ScenarioProjection(
            projection_id=projection_id,
            engine_version=self.engine_version,
            base=base.metadata,
            scenario_spec_id=scenario.scenario_spec_id,
            regime_spec_id=(
                regime.regime_spec_id if regime is not None else None
            ),
            relations=relations_tuple,
            impacts=impacts_tuple,
            node_summaries=summaries_tuple,
            sensitivity=sensitivity_tuple,
            trace=trace,
            output_digest=output_digest,
        )

    @staticmethod
    def _entity_catalog(graph, evidence_relations):
        entity_ids: set[str] = set()
        entity_names: dict[str, str] = {}
        for node in graph.nodes:
            if node.kind != NodeKind.ENTITY:
                continue
            stable_key = node.properties.get("stable_key")
            if stable_key is None and node.id.startswith("entity:"):
                stable_key = node.id[len("entity:") :]
            if stable_key:
                stable_key = str(stable_key)
                entity_ids.add(stable_key)
                entity_names[stable_key] = (
                    node.properties.get("canonical_name")
                    or node.label
                    or stable_key
                )
        for relation in evidence_relations:
            entity_ids.update((relation.head_id, relation.tail_id))
            entity_names.setdefault(relation.head_id, relation.head_name)
            entity_names.setdefault(relation.tail_id, relation.tail_name)
        return entity_ids, entity_names

    def _propagate(
        self,
        relations: list[ProjectedRelation],
        scenario: ScenarioSpec,
        entity_ids: set[str],
    ):
        outgoing: dict[str, list[ProjectedRelation]] = defaultdict(list)
        for relation in relations:
            if (
                relation.active
                and relation.projected_weight > 0.0
                and relation.projected_sign in {"+", "-"}
            ):
                outgoing[relation.head_id].append(relation)
        for values in outgoing.values():
            values.sort(key=lambda item: item.relation_id)

        impacts: list[ProjectedImpact] = []
        dependencies: list[ProjectionDependency] = []
        warnings: list[str] = []

        for shock in scenario.shocks:
            if shock.target_entity_id not in entity_ids:
                raise ValueError(
                    f"shock references unknown entity: {shock.target_entity_id}"
                )
            emitted_before = len(impacts)
            queue = deque(
                [
                    (
                        shock.target_entity_id,
                        (shock.target_entity_id,),
                        (),
                        shock.direction,
                        shock.magnitude,
                        (),
                        (),
                    )
                ]
            )
            while queue and len(impacts) < scenario.max_paths:
                (
                    current,
                    path_entities,
                    path_relations,
                    path_sign,
                    path_value,
                    path_rules,
                    path_assumptions,
                ) = queue.popleft()
                depth = len(path_relations)
                if depth >= scenario.max_depth:
                    continue

                for relation in outgoing.get(current, []):
                    if relation.tail_id in path_entities:
                        continue
                    new_sign = _combine_sign(
                        path_sign,
                        relation.projected_sign,
                    )
                    if new_sign == "unknown":
                        continue
                    new_value = round(
                        path_value * relation.projected_weight,
                        12,
                    )
                    new_entities = (*path_entities, relation.tail_id)
                    new_relations = (*path_relations, relation.relation_id)
                    new_rules = tuple(
                        sorted(
                            set(path_rules).union(
                                relation.applied_regime_rule_ids
                            )
                        )
                    )
                    new_assumptions = tuple(
                        sorted(
                            set(path_assumptions).union(
                                relation.applied_assumption_ids
                            )
                        )
                    )
                    impact = ProjectedImpact(
                        shock_id=shock.shock_id,
                        target_entity_id=relation.tail_id,
                        direction=new_sign,
                        impact_value=new_value,
                        path_entities=new_entities,
                        path_relation_ids=new_relations,
                        applied_rule_ids=new_rules,
                        applied_assumption_ids=new_assumptions,
                    )
                    impacts.append(impact)
                    dependencies.append(
                        ProjectionDependency(
                            kind="shock_path",
                            relation_id=relation.relation_id,
                            input_id=shock.shock_id,
                            detail="->".join(new_relations),
                        )
                    )
                    if len(impacts) >= scenario.max_paths:
                        break
                    queue.append(
                        (
                            relation.tail_id,
                            new_entities,
                            new_relations,
                            new_sign,
                            new_value,
                            new_rules,
                            new_assumptions,
                        )
                    )
            if len(impacts) == emitted_before:
                warnings.append(
                    f"shock produced no propagating path: {shock.shock_id}"
                )
        return impacts, dependencies, warnings

    @staticmethod
    def _summaries(
        impacts: Iterable[ProjectedImpact],
    ) -> list[NodeImpactSummary]:
        grouped: dict[str, list[ProjectedImpact]] = defaultdict(list)
        for impact in impacts:
            grouped[impact.target_entity_id].append(impact)

        summaries = []
        for entity_id, values in grouped.items():
            positive = max(
                (
                    item.impact_value
                    for item in values
                    if item.direction == "+"
                ),
                default=0.0,
            )
            negative = max(
                (
                    item.impact_value
                    for item in values
                    if item.direction == "-"
                ),
                default=0.0,
            )
            summaries.append(
                NodeImpactSummary(
                    entity_id=entity_id,
                    strongest_positive=round(positive, 12),
                    strongest_negative=round(negative, 12),
                    net_score=round(positive - negative, 12),
                )
            )
        return summaries

    @staticmethod
    def _sensitivity(
        relations: list[ProjectedRelation],
        scenario: ScenarioSpec,
    ) -> list[SensitivityResult]:
        result: list[SensitivityResult] = []
        for threshold in scenario.sensitivity_thresholds:
            for relation in relations:
                if not threshold.selector.matches(
                    relation_node_id=relation.base_relation_node_id,
                    head_id=relation.head_id,
                    tail_id=relation.tail_id,
                    relation_type=relation.relation_type,
                ):
                    continue
                value = relation.projected_weight
                operator = threshold.operator
                triggered = {
                    "<": value < threshold.threshold,
                    "<=": value <= threshold.threshold,
                    ">": value > threshold.threshold,
                    ">=": value >= threshold.threshold,
                }[operator]
                result.append(
                    SensitivityResult(
                        threshold_id=threshold.threshold_id,
                        relation_id=relation.relation_id,
                        projected_weight=value,
                        operator=operator,
                        threshold=threshold.threshold,
                        triggered=triggered,
                    )
                )
        return result
