"""
Reasoning Pipeline: Query -> Retrieval -> Fusion -> Path Reasoning -> Conclusion -> ScenarioReport
"""
import logging
from typing import Optional, Dict
from datetime import datetime

from src.reasoning.models import (
    ParsedQuery,
    RetrievalResult,
    ReasoningResult,
)
from src.reasoning.query_parser import QueryParser
from src.reasoning.graph_retrieval import GraphRetrieval
from src.reasoning.edge_fusion import EdgeWeightFusion
from src.reasoning.path_reasoning import PathReasoningEngine
from src.reasoning.conclusion import ConclusionSynthesizer
from src.reasoning import policy_injector
from src.reasoning.trace_contracts import ReasoningTrace, PathTrace, EdgeTrace

from src.extraction.ner_student import NERStudent
from src.extraction.entity_resolver import EntityResolver
from src.domain.dynamic_update import DynamicDomainUpdate
from src.personal.pkg_update import PersonalKGUpdate
from src.llm.ollama_client import OllamaClient
from src.policy.policy_registry import get_policy_registry
from src.scenario.scenario_builder import ScenarioBuilder
from src.delta.delta_detector import DeltaDetector
from src.scenario.break_conditions import BreakConditionBuilder

logger = logging.getLogger(__name__)


class ReasoningPipeline:
    """
    Reasoning Sector 파이프라인
    """

    def __init__(
        self,
        domain: Optional[DynamicDomainUpdate] = None,
        personal: Optional[PersonalKGUpdate] = None,
        llm_client: Optional[OllamaClient] = None,
        ner: Optional[NERStudent] = None,
        resolver: Optional[EntityResolver] = None,
    ):
        self.query_parser = QueryParser(
            ner_student=ner,
            entity_resolver=resolver,
            llm_client=llm_client,
        )
        self.graph_retrieval = GraphRetrieval(domain=domain, personal=personal)
        self.edge_fusion = EdgeWeightFusion()
        self.path_reasoning = PathReasoningEngine()
        self.conclusion = ConclusionSynthesizer(llm_client=llm_client)

        self.domain = domain
        self.personal = personal
        self.llm_client = llm_client

        self._stats = {
            "queries_processed": 0,
            "avg_paths_used": 0.0,
            "avg_confidence": 0.0,
        }
        self._last_domain_conf: Dict[str, float] = {}

    def reason(
        self,
        query: str,
        as_of: Optional[datetime] = None,
        context: Optional[Dict] = None,
    ):
        """
        질의에 대한 추론을 실행하고 ScenarioReport를 반환한다.
        """
        policy = get_policy_registry().get_active_policy()
        policy_injector.apply(
            policy,
            edge_fusion=self.edge_fusion,
            graph_retrieval=self.graph_retrieval,
            confidence_filter=getattr(self, "confidence_filter", None),
        )
        logger.info(f"[Reasoning] policy_id={policy.policy_id}, version={policy.version}")

        self._stats["queries_processed"] += 1

        parsed_query = self.query_parser.parse(query)
        # Pass as_of and context to retrieval
        retrieval_result = self.graph_retrieval.retrieve(parsed_query, as_of=as_of, context=context)
        all_paths = retrieval_result.direct_paths + retrieval_result.indirect_paths

        fused_paths = self.edge_fusion.fuse_multiple_paths(all_paths)
        reasoning_result = self.path_reasoning.reason(fused_paths, parsed_query.query_id)

        self._update_stats(reasoning_result)

        conclusion = self.conclusion.synthesize(parsed_query, reasoning_result)
        trace = self._build_trace(fused_paths, reasoning_result, parsed_query, policy.policy_id)

        # Delta Detection: Prefer snapshot from context to avoid graph loading
        current_conf = {}
        if context and "snapshot" in context and context["snapshot"]:
            snapshot = context["snapshot"]
            # Use encapsulated method
            if hasattr(snapshot, "to_conf_map"):
                current_conf = snapshot.to_conf_map()
        else:
            # Fallback to domain adapter (use graph_retrieval.domain as it might be patched)
            # Use graph_retrieval.domain instead of self.domain to respect adapter injection
            current_domain = self.graph_retrieval.domain or self.domain
            if current_domain:
                current_conf = {
                    rel_id: rel.domain_conf for rel_id, rel in current_domain.get_all_relations().items()
                }

        execution_time = as_of if as_of else datetime.utcnow()
        delta_report = DeltaDetector().detect(as_of=execution_time, current=current_conf, previous=self._last_domain_conf)
        self._last_domain_conf = current_conf
        break_report = BreakConditionBuilder().build(trace)
        scenario = ScenarioBuilder().build(
            conclusion=conclusion,
            trace=trace,
            delta=delta_report,
            break_conditions=break_report,
            meta={
                "policy_id": policy.policy_id,
                "policy_version": policy.version,
                "as_of": execution_time.isoformat(),
            },
        )
        return scenario

    def reason_detailed(self, query: str) -> Dict:
        """상세 추론 결과 반환"""
        parsed = self.query_parser.parse(query)
        retrieval = self.graph_retrieval.retrieve(parsed)
        all_paths = retrieval.direct_paths + retrieval.indirect_paths
        fused = self.edge_fusion.fuse_multiple_paths(all_paths)
        reasoning = self.path_reasoning.reason(fused, parsed.query_id)
        conclusion = self.conclusion.synthesize(parsed, reasoning)
        trace = self._build_trace(fused, reasoning, parsed, policy_id=None)
        return {
            "query": query,
            "parsed": {
                "entities": parsed.query_entities,
                "type": parsed.query_type.value,
                "head": parsed.head_entity,
                "tail": parsed.tail_entity,
            },
            "retrieval": {
                "direct_paths": len(retrieval.direct_paths),
                "indirect_paths": len(retrieval.indirect_paths),
                "domain_count": retrieval.domain_paths_count,
                "personal_count": retrieval.personal_paths_count,
            },
            "reasoning": {
                "direction": reasoning.direction.value,
                "confidence": reasoning.confidence,
                "positive_evidence": reasoning.positive_evidence,
                "negative_evidence": reasoning.negative_evidence,
                "paths_used": len(reasoning.paths_used),
                "conflicting": reasoning.conflicting_paths,
            },
            "conclusion": {
                "text": conclusion.conclusion_text,
                "explanation": conclusion.explanation_text,
                "strongest_path": conclusion.strongest_path_description,
            },
            "trace": trace,
        }

    def _update_stats(self, result: ReasoningResult):
        n = self._stats["queries_processed"]
        old_avg_paths = self._stats["avg_paths_used"]
        new_paths = len(result.paths_used)
        self._stats["avg_paths_used"] = (old_avg_paths * (n - 1) + new_paths) / n
        old_avg_conf = self._stats["avg_confidence"]
        self._stats["avg_confidence"] = (old_avg_conf * (n - 1) + result.confidence) / n

    def get_stats(self) -> Dict:
        return self._stats.copy()

    def reset_stats(self):
        self._stats = {
            "queries_processed": 0,
            "avg_paths_used": 0.0,
            "avg_confidence": 0.0,
        }

    def _build_trace(
        self,
        fused_paths,
        reasoning_result,
        parsed_query: ParsedQuery,
        policy_id: Optional[str],
    ) -> ReasoningTrace:
        path_map = {fp.path_id: fp for fp in fused_paths}
        candidate_traces = []
        selected_path_trace = None

        for path_res in reasoning_result.paths_used:
            fused = path_map.get(path_res.path_id)
            edges = []
            if fused:
                for e in fused.fused_edges:
                    edges.append(
                        EdgeTrace(
                            head_id=e.head_id,
                            tail_id=e.tail_id,
                            relation_type=e.relation_type,
                            polarity=e.sign,
                            final_weight=e.final_weight,
                            domain_conf=e.domain_conf,
                            pcs=e.pcs_score,
                            semantic_score=e.semantic_score,
                        )
                    )
                pt = PathTrace(
                    nodes=fused.nodes,
                    edges=edges,
                    path_weight=fused.path_weight,
                    sign_product=fused.path_sign,
                    why_selected=None,
                )
            else:
                pt = PathTrace(
                    nodes=path_res.nodes,
                    edges=[],
                    path_weight=path_res.path_strength,
                    sign_product=path_res.combined_sign,
                )
            candidate_traces.append(pt)
            if reasoning_result.strongest_path and reasoning_result.strongest_path.path_id == path_res.path_id:
                selected_path_trace = pt

        return ReasoningTrace(
            query_entities=parsed_query.query_entities,
            candidate_paths=candidate_traces,
            selected_path=selected_path_trace or (candidate_traces[0] if candidate_traces else None),
            policy_id=policy_id,
            as_of=datetime.utcnow(),
            trace_id=parsed_query.query_id,
        )
