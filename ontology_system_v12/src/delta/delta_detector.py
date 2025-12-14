from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from src.scenario.contracts import DeltaReport, ScalarDeltaItem, TopologyDeltaItem
from src.delta.topology_digest import digest, diff, TopologyDigest

class DeltaDetector:
    """
    Enhanced Delta Detector: Scalar + Topology
    """

    def __init__(self, threshold: float = 0.1, lookback_days: int = 1):
        self.threshold = threshold
        self.lookback_days = lookback_days
        self._last_topology: Optional[TopologyDigest] = None

    def detect(
        self,
        as_of: datetime,
        current: Optional[Dict[str, float]] = None,
        previous: Optional[Dict[str, float]] = None,
        # Optional: Pass graph objects for topology detection
        current_graph: Optional[Any] = None, 
    ) -> DeltaReport:
        scalar_items = []
        topology_items = []
        
        # 1. Scalar Detection
        if current and previous:
            for key, curr_val in current.items():
                prev_val = previous.get(key)
                if prev_val is None:
                    continue
                delta_val = curr_val - prev_val
                delta_pct = (delta_val / prev_val) if prev_val != 0 else None
                
                if delta_pct is not None and abs(delta_pct) >= self.threshold:
                    scalar_items.append(
                        ScalarDeltaItem(
                            id=key,
                            feature_name="confidence",
                            prev_value=prev_val,
                            curr_value=curr_val,
                            delta_val=delta_val,
                            delta_pct=delta_pct,
                            severity="HIGH" if abs(delta_pct) >= self.threshold * 2 else "MEDIUM"
                        )
                    )

        # 2. Topology Detection
        if current_graph:
            curr_topo = digest(current_graph)
            if self._last_topology:
                topo_changes = diff(self._last_topology, curr_topo)
                topology_items.extend(topo_changes)
            self._last_topology = curr_topo
        
        summary = f"Detected {len(scalar_items)} scalar changes and {len(topology_items)} topology changes."
        
        return DeltaReport(
            scalar_items=scalar_items,
            topology_items=topology_items,
            summary=summary
        )
