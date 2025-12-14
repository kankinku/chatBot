from typing import Any, Dict, Optional
from datetime import datetime

from src.scenario.contracts import (
    ScenarioReport, ChainReport, ChainLink,
    EvidenceReport, EvidenceItem, ConclusionReport,
    MetaReport, DeltaReport
)
from src.reasoning.trace_contracts import ReasoningTrace
from src.reasoning.models import ReasoningConclusion

class ScenarioBuilder:
    """
    Constructs the 5-part constrained ScenarioReport.
    """
    
    def build(
        self,
        conclusion: ReasoningConclusion,
        trace: ReasoningTrace,
        delta: DeltaReport,
        break_conditions: Any, # BreakConditionsReport
        meta: Dict[str, Any],
    ) -> ScenarioReport:
        
        # 1. Chain Report
        chain_links = []
        chain_desc = ""
        if trace and trace.selected_path:
            for edge in trace.selected_path.edges:
                chain_links.append(ChainLink(
                    source_entity=edge.head_id,
                    target_entity=edge.tail_id,
                    relation=edge.relation_type,
                    weight=edge.final_weight,
                    sign=edge.polarity
                ))
            chain_desc = f"Path of length {len(chain_links)} selected with weight {trace.selected_path.path_weight:.2f}"
            
        chain_report = ChainReport(links=chain_links, description=chain_desc)
        
        # 2. Evidence Report (derived from ReasoningResult/Trace)
        pos_items = []
        neg_items = []
        # In a real impl, we'd dig into 'supporting_fragment_ids' from EdgeTrace
        # For now, we use conclusion's evidences if available, or trace edges.
        if hasattr(conclusion, "positive_evidence"):
             for e in conclusion.positive_evidence:
                 pos_items.append(EvidenceItem("POSITIVE", 1.0, str(e), "unknown"))
        if hasattr(conclusion, "negative_evidence"):
             for e in conclusion.negative_evidence:
                 neg_items.append(EvidenceItem("NEGATIVE", 1.0, str(e), "unknown"))

        evidence_report = EvidenceReport(
            positive_items=pos_items,
            negative_items=neg_items,
            total_score=0.0 # Calc if needed
        )
        
        # 3. Conclusion Report
        conc_report = ConclusionReport(
            direction=getattr(conclusion, "direction", "FLAT"),
            confidence=getattr(conclusion, "confidence", 0.0),
            text=getattr(conclusion, "conclusion_text", ""),
            rationale=getattr(conclusion, "explanation_text", "")
        )
        
        # 4. Meta
        meta_report = MetaReport(
            policy_id=meta.get("policy_id", "unknown"),
            policy_version=meta.get("policy_version", "v0"),
            as_of=datetime.fromisoformat(meta["as_of"]) if isinstance(meta.get("as_of"), str) else datetime.utcnow(),
            snapshot_id=meta.get("snapshot_id"),
            seed=meta.get("seed"),
            trace_id=trace.trace_id if trace else None
        )
        
        return ScenarioReport(
            delta=delta,
            chain=chain_report,
            evidence=evidence_report,
            conclusion=conc_report,
            break_conditions=break_conditions,
            meta=meta_report
        )
