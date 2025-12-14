from typing import List, Dict, Any, Optional
from src.scenario.contracts import BreakConditionsReport, BreakCondition
from src.reasoning.trace_contracts import ReasoningTrace, PathTrace

class BreakConditionBuilder:
    """
    Builds BreakConditionsReport from reasoning traces and delta reports.
    Identifies conditions under which the current conclusion would be invalidated.
    """
    
    def build(self, trace: Any) -> BreakConditionsReport:
        # Trace can be ReasoningTrace or dict (legacy). We support ReasoningTrace.
        # If dict, we try to wrap it or extract info.
        
        conditions = []
        
        selected_path = None
        if isinstance(trace, ReasoningTrace):
            selected_path = trace.selected_path
        elif isinstance(trace, dict):
             # Legacy/Dict support if needed
             pass
             
        if selected_path:
            # 1. Critical Edge Dependency (Weight-based)
            # Find the weakest link in the strongest path? 
            # Or the edge with highest weight that if dropped changes result?
            # For MVP: Pick edge with lowest non-zero weight (easiest to break) 
            # or highest weight (most critical).
            # The prompt suggested: "edge_weight < w_min"
            
            # Let's verify edges.
            sorted_edges = sorted(selected_path.edges, key=lambda e: e.final_weight, reverse=True)
            
            if sorted_edges:
                # Top contributor (Highest Weight)
                top_edge = sorted_edges[0]
                conditions.append(BreakCondition(
                    trigger_type="EDGE_WEIGHT_DROP",
                    threshold=0.25, # Arbitrary MVP threshold
                    current_value=top_edge.final_weight,
                    target_component=f"{top_edge.head_id}->{top_edge.tail_id} ({top_edge.relation_type})",
                    rationale="This edge provides the strongest evidence. If its confidence drops below 0.25, the chain breaks."
                ))
                
                # Weakest link (Lowest non-zero)
                weak_edges = [e for e in sorted_edges if e.final_weight > 0.0]
                if weak_edges:
                    weak_edge = weak_edges[-1]
                    if weak_edge != top_edge:
                         conditions.append(BreakCondition(
                            trigger_type="EDGE_WEIGHT_DROP",
                            threshold=0.10,
                            current_value=weak_edge.final_weight,
                            target_component=f"{weak_edge.head_id}->{weak_edge.tail_id}",
                            rationale="This is the weakest link. Any further drop invalidates the path."
                         ))

        # 2. Evidence Support (Stub)
        # If we had access to raw evidence scores here, we would add EVIDENCE_DROP trigger.
        
        return BreakConditionsReport(conditions=conditions)
