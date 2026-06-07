"""
Latency metrics for Guardian SDK benchmark suite.

Computes: p50, p99, mean, min, max, standard deviation, histogram.
All values in milliseconds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Per-run latency statistics (all values in ms)."""

    n_requests: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    histogram: Dict[str, int] = field(default_factory=dict)   # bucket → count

    def as_dict(self) -> dict:
        return {
            "n_requests": self.n_requests,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "std_ms": round(self.std_ms, 2),
            "histogram": self.histogram,
        }


def compute_latency_metrics(latencies_ms: List[float]) -> LatencyMetrics:
    """
    Compute latency statistics from a list of per-request latencies.

    Args:
        latencies_ms: Per-request latencies in milliseconds.

    Returns:
        LatencyMetrics with p50, p95, p99, mean, min, max, std, histogram.
    """
    if not latencies_ms:
        return LatencyMetrics()

    arr = np.array(latencies_ms, dtype=float)

    # Histogram buckets (ms)
    buckets = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 500, float("inf")]
    labels = [
        "<5ms", "5-10ms", "10-20ms", "20-30ms", "30-50ms",
        "50-75ms", "75-100ms", "100-150ms", "150-200ms", "200-500ms", ">500ms",
    ]
    histogram: Dict[str, int] = {}
    for i, label in enumerate(labels):
        lo, hi = buckets[i], buckets[i + 1]
        count = int(((arr >= lo) & (arr < hi)).sum())
        histogram[label] = count

    metrics = LatencyMetrics(
        n_requests=len(arr),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        std_ms=float(arr.std()),
        histogram=histogram,
    )

    logger.info(
        "Latency: p50=%.1fms p95=%.1fms p99=%.1fms mean=%.1fms",
        metrics.p50_ms, metrics.p95_ms, metrics.p99_ms, metrics.mean_ms,
    )
    return metrics
