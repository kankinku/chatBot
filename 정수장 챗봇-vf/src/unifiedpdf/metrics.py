from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MetricsCollector:
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)

    def aggregate(self) -> Dict[str, Any]:
        if not self.rows:
            return {}
        def getf(k: str) -> List[float]:
            return [float(r.get(k, 0.0)) for r in self.rows]
        agg = {
            "count": len(self.rows),
            "p50_total_time_ms": percentile(getf("total_time_ms"), 50),
            "p95_total_time_ms": percentile(getf("total_time_ms"), 95),
            "mean_filter_pass_rate": statistics.mean(getf("filter_pass_rate")) if self.rows else 0.0,
            "mean_context_filled": statistics.mean(getf("context_filled")) if self.rows else 0.0,
            "mean_numeric_preservation": statistics.mean(getf("numeric_preservation")) if self.rows else 0.0,
            "mean_overlap_ratio": statistics.mean(getf("overlap_ratio")) if self.rows else 0.0,
            "no_answer_rate": sum(1 for r in self.rows if r.get("no_answer", 0) == 1) / len(self.rows),
        }
        return agg

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rows": self.rows, "aggregate": self.aggregate()}, f, ensure_ascii=False, indent=2)

    def to_csv(self, path: str) -> None:
        if not self.rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write("")
            return
        keys = sorted({k for r in self.rows for k in r.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)

