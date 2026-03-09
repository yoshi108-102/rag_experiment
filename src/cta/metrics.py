"""Performance metrics helpers for CTA Sprint 2."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from src.cta.models import CTATurnRecord


def _percentile(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    rank = ceil((p / 100) * len(sorted_values))
    idx = max(0, min(rank - 1, len(sorted_values) - 1))
    return sorted_values[idx]


@dataclass(frozen=True)
class PerformanceSummary:
    count: int
    mean_ms: float
    p50_ms: int
    p95_ms: int
    max_ms: int
    target_p95_ms: int
    within_target: bool


def summarize_turn_latency(
    turns: list[CTATurnRecord],
    target_p95_ms: int = 5000,
) -> PerformanceSummary:
    latencies = [max(0, int(turn.processing_latency_ms)) for turn in turns]
    if not latencies:
        return PerformanceSummary(
            count=0,
            mean_ms=0.0,
            p50_ms=0,
            p95_ms=0,
            max_ms=0,
            target_p95_ms=target_p95_ms,
            within_target=True,
        )

    sorted_latencies = sorted(latencies)
    mean_ms = sum(sorted_latencies) / len(sorted_latencies)
    p50_ms = _percentile(sorted_latencies, 50)
    p95_ms = _percentile(sorted_latencies, 95)
    max_ms = sorted_latencies[-1]

    return PerformanceSummary(
        count=len(sorted_latencies),
        mean_ms=round(mean_ms, 2),
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        max_ms=max_ms,
        target_p95_ms=target_p95_ms,
        within_target=p95_ms <= target_p95_ms,
    )

