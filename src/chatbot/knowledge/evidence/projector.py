"""Project extraction/validation/domain results into evidence provenance graph."""

from __future__ import annotations

from typing import Iterable

from chatbot.knowledge.domain.models import DomainProcessResult
from chatbot.knowledge.shared.models import ExtractionResult, ResolvedEntity
from chatbot.knowledge.validation.models import ValidationResult
from chatbot.knowledge.workspace.hashing import hash_value
from chatbot.knowledge.workspace.invariants import validate_graph
from chatbot.knowledge.workspace.models import (
    Confidence,
    MetaRelation,
    NodeKind,
    SourceRef,
    WorkspaceEdge,
    WorkspaceGraph,
    WorkspaceNode,
)

from .models import EvidenceProjection


def _edge(
    source_id: str,
    target_id: str,
    relation: MetaRelation,
    source_ref: SourceRef,
    *,
    confidence: Confidence,
    semantic_type: str | None = None,
) -> WorkspaceEdge:
    payload = {
        "source": source_id,
        "target": target_id,
        "relation": relation.value,
        "semantic_type": semantic_type,
    }
    return WorkspaceEdge(
        source=source_id,
        target=target_id,
        relation=relation,
        confidence=confidence,
        content_hash=hash_value(payload),
        sources=[source_ref],
        semantic_type=semantic_type,
    )


def _enum_value(value):
    return getattr(value, "value", value)


def _validation_properties(
    result: ValidationResult | None,
) -> dict:
    if result is None:
        return {"validation_state": "not_run"}

    props = {
        "validation_state": (
            "passed" if result.validation_passed else "rejected"
        ),
        "combined_conf": result.combined_conf,
        "student_conf": result.student_conf,
        "sign_score": result.sign_score,
        "semantic_conf": result.semantic_conf,
        "rejection_reason": result.rejection_reason,
    }
    if result.semantic_result is not None:
        props["semantic_tag"] = _enum_value(
            result.semantic_result.semantic_tag
        )
        props["domain_conflict"] = result.semantic_result.domain_conflict
    if result.sign_result is not None:
        props["polarity_final"] = result.sign_result.polarity_final
        props["sign_tag"] = _enum_value(result.sign_result.sign_tag)
    return props


def _domain_properties(
    result: DomainProcessResult | None,
) -> dict:
    if result is None:
        return {"domain_state": "not_run"}

    props = {
        "domain_state": result.final_destination,
    }
    if result.static_result is not None:
        props["static_conflict"] = result.static_result.static_conflict
        props["static_action"] = _enum_value(result.static_result.action)
    return props


class EvidenceProjector:
    """Build deterministic document evidence graphs from runtime pipeline results."""

    def project(
        self,
        extraction: ExtractionResult,
        *,
        validation_results: Iterable[ValidationResult] = (),
        domain_results: Iterable[DomainProcessResult] = (),
    ) -> EvidenceProjection:
        if not extraction.source_uri or not extraction.source_hash:
            raise ValueError(
                "ExtractionResult must include source_uri and source_hash"
            )
        if len(extraction.source_hash) != 64:
            raise ValueError("source_hash must be a SHA-256 hex digest")
        try:
            int(extraction.source_hash, 16)
        except ValueError as exc:
            raise ValueError(
                "source_hash must be a SHA-256 hex digest"
            ) from exc

        source_ref = SourceRef(
            path=extraction.source_uri,
            hash=extraction.source_hash,
        )
        validation_by_edge = {
            item.edge_id: item
            for item in validation_results
        }
        domain_by_edge = {
            item.raw_edge_id: item
            for item in domain_results
        }
        resolved_by_temp = {
            item.entity_id: item
            for item in extraction.resolved_entities
        }

        document_node_id = (
            "document:"
            + hash_value({"source_uri": extraction.source_uri})[:20]
        )
        document_props = {
            "doc_id": extraction.doc_id,
            "source_uri": extraction.source_uri,
        }
        nodes: list[WorkspaceNode] = [
            WorkspaceNode(
                id=document_node_id,
                label=extraction.doc_id,
                kind=NodeKind.DOCUMENT,
                content_hash=extraction.source_hash,
                sources=[source_ref],
                properties=document_props,
            )
        ]
        edges: list[WorkspaceEdge] = []
        fragment_nodes: dict[str, str] = {}

        for fragment in extraction.fragments:
            fragment_props = {
                "doc_id": fragment.doc_id,
                "text": fragment.text,
                "source_start": fragment.source_start,
                "source_end": fragment.source_end,
                "quality_tag": _enum_value(fragment.quality_tag),
            }
            fragment_key = {
                "source_hash": extraction.source_hash,
                **fragment_props,
            }
            fragment_node_id = (
                "fragment:" + hash_value(fragment_key)[:24]
            )
            fragment_nodes[fragment.fragment_id] = fragment_node_id
            nodes.append(
                WorkspaceNode(
                    id=fragment_node_id,
                    label=fragment.text[:80] or fragment_node_id,
                    kind=NodeKind.FRAGMENT,
                    content_hash=hash_value(fragment_props),
                    sources=[source_ref],
                    properties=fragment_props,
                )
            )
            edges.append(
                _edge(
                    document_node_id,
                    fragment_node_id,
                    MetaRelation.PRODUCES,
                    source_ref,
                    confidence=Confidence.EXTRACTED,
                )
            )

        for raw_edge in extraction.raw_edges:
            fragment_node_id = fragment_nodes.get(raw_edge.fragment_id)
            if fragment_node_id is None:
                continue

            head_node, head_key = self._entity_node(
                raw_edge.head_entity_id,
                raw_edge.head_canonical_name,
                resolved_by_temp,
                source_ref,
            )
            tail_node, tail_key = self._entity_node(
                raw_edge.tail_entity_id,
                raw_edge.tail_canonical_name,
                resolved_by_temp,
                source_ref,
            )
            nodes.extend([head_node, tail_node])

            validation = validation_by_edge.get(raw_edge.raw_edge_id)
            domain = domain_by_edge.get(raw_edge.raw_edge_id)

            assertion_props = {
                "relation_type": raw_edge.relation_type,
                "polarity_guess": _enum_value(raw_edge.polarity_guess),
                "student_conf": raw_edge.student_conf,
                "condition_text": raw_edge.condition_text,
                "head": head_key,
                "tail": tail_key,
                **_validation_properties(validation),
                **_domain_properties(domain),
            }
            assertion_key = {
                "source_hash": extraction.source_hash,
                "fragment_node_id": fragment_node_id,
                "head": head_key,
                "tail": tail_key,
                "relation_type": raw_edge.relation_type,
                "polarity_guess": _enum_value(raw_edge.polarity_guess),
                "condition_text": raw_edge.condition_text,
            }
            assertion_node_id = (
                "assertion:" + hash_value(assertion_key)[:24]
            )
            assertion_confidence = (
                Confidence.VALIDATED
                if validation is not None and validation.validation_passed
                else Confidence.EXTRACTED
            )
            nodes.append(
                WorkspaceNode(
                    id=assertion_node_id,
                    label=(
                        f"{head_key} {raw_edge.relation_type} {tail_key}"
                    ),
                    kind=NodeKind.ASSERTION,
                    content_hash=hash_value(assertion_props),
                    sources=[source_ref],
                    properties=assertion_props,
                )
            )
            edges.extend(
                [
                    _edge(
                        fragment_node_id,
                        assertion_node_id,
                        MetaRelation.PRODUCES,
                        source_ref,
                        confidence=Confidence.EXTRACTED,
                    ),
                    _edge(
                        assertion_node_id,
                        head_node.id,
                        MetaRelation.DEPENDS_ON,
                        source_ref,
                        confidence=Confidence.DERIVED,
                    ),
                    _edge(
                        assertion_node_id,
                        tail_node.id,
                        MetaRelation.DEPENDS_ON,
                        source_ref,
                        confidence=Confidence.DERIVED,
                    ),
                ]
            )

            disposition = self._relation_disposition(
                validation,
                domain,
            )
            if disposition is None:
                continue

            relation_props = {
                "head": head_key,
                "tail": tail_key,
                "relation_type": raw_edge.relation_type,
            }
            relation_node_id = (
                "domain-relation:"
                + hash_value(relation_props)[:24]
            )
            nodes.append(
                WorkspaceNode(
                    id=relation_node_id,
                    label=(
                        f"{head_key} {raw_edge.relation_type} {tail_key}"
                    ),
                    kind=NodeKind.DOMAIN_RELATION,
                    content_hash=hash_value(relation_props),
                    sources=[source_ref],
                    properties=relation_props,
                )
            )
            edges.extend(
                [
                    _edge(
                        relation_node_id,
                        head_node.id,
                        MetaRelation.DEPENDS_ON,
                        source_ref,
                        confidence=Confidence.DERIVED,
                    ),
                    _edge(
                        relation_node_id,
                        tail_node.id,
                        MetaRelation.DEPENDS_ON,
                        source_ref,
                        confidence=Confidence.DERIVED,
                    ),
                    _edge(
                        relation_node_id,
                        assertion_node_id,
                        disposition,
                        source_ref,
                        confidence=Confidence.VALIDATED,
                    ),
                ]
            )
            if disposition == MetaRelation.SUPPORTED_BY:
                edges.append(
                    _edge(
                        head_node.id,
                        tail_node.id,
                        MetaRelation.DOMAIN_RELATION,
                        source_ref,
                        confidence=Confidence.VALIDATED,
                        semantic_type=raw_edge.relation_type,
                    )
                )

        graph = self._deduplicate(nodes, edges)
        validate_graph(graph)
        projection_hash = hash_value(graph.to_dict())
        return EvidenceProjection(
            source_uri=extraction.source_uri,
            source_hash=extraction.source_hash,
            document_node_id=document_node_id,
            projection_hash=projection_hash,
            graph=graph,
        )

    @staticmethod
    def _relation_disposition(
        validation: ValidationResult | None,
        domain: DomainProcessResult | None,
    ) -> MetaRelation | None:
        if domain is not None:
            if (
                domain.static_result is not None
                and domain.static_result.static_conflict
            ):
                return MetaRelation.CONTRADICTED_BY
            if domain.final_destination == "domain":
                return MetaRelation.SUPPORTED_BY
            return None

        if validation is None:
            return None
        if (
            validation.semantic_result is not None
            and validation.semantic_result.domain_conflict
        ):
            return MetaRelation.CONTRADICTED_BY
        if validation.validation_passed:
            return MetaRelation.SUPPORTED_BY
        return None

    @staticmethod
    def _entity_node(
        temporary_id: str,
        canonical_name_hint: str | None,
        resolved_by_temp: dict[str, ResolvedEntity],
        source_ref: SourceRef,
    ) -> tuple[WorkspaceNode, str]:
        resolved = resolved_by_temp.get(temporary_id)
        canonical_id = (
            resolved.canonical_id if resolved is not None else None
        )
        canonical_name = (
            resolved.canonical_name if resolved is not None else None
        ) or canonical_name_hint
        canonical_type = (
            resolved.canonical_type if resolved is not None else None
        )

        if canonical_id:
            node_id = f"entity:{canonical_id}"
            stable_key = canonical_id
            label = canonical_id
        else:
            stable_key = (
                canonical_name
                or (
                    resolved.surface_text
                    if resolved is not None
                    else temporary_id
                )
            )
            node_id = "entity-ref:" + hash_value(
                {"stable_key": stable_key}
            )[:20]
            label = stable_key

        props = {
            "canonical_id": canonical_id,
            "canonical_type": canonical_type,
            "stable_key": stable_key,
        }
        return (
            WorkspaceNode(
                id=node_id,
                label=label,
                kind=NodeKind.ENTITY_REF,
                content_hash=hash_value(props),
                sources=[source_ref],
                properties=props,
            ),
            stable_key,
        )

    @staticmethod
    def _deduplicate(
        nodes: list[WorkspaceNode],
        edges: list[WorkspaceEdge],
    ) -> WorkspaceGraph:
        node_map: dict[str, WorkspaceNode] = {}
        for node in sorted(nodes, key=lambda item: item.id):
            current = node_map.get(node.id)
            if current is None:
                node_map[node.id] = node
                continue
            if (
                current.kind != node.kind
                or current.content_hash != node.content_hash
                or current.properties != node.properties
            ):
                raise ValueError(
                    f"conflicting evidence node id: {node.id}"
                )

        edge_map: dict[tuple[str, str, str, str], WorkspaceEdge] = {}
        for edge in sorted(edges, key=lambda item: item.identity):
            current = edge_map.get(edge.identity)
            if current is None:
                edge_map[edge.identity] = edge
                continue
            if (
                current.content_hash != edge.content_hash
                or current.confidence != edge.confidence
                or current.properties != edge.properties
            ):
                raise ValueError(
                    f"conflicting evidence edge: {edge.identity}"
                )

        return WorkspaceGraph(
            nodes=[node_map[key] for key in sorted(node_map)],
            edges=[edge_map[key] for key in sorted(edge_map)],
        )
