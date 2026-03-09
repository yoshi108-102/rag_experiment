from __future__ import annotations

from src.cta.metrics import summarize_turn_latency
from src.cta.models import CTATurnRecord


def _turn(latency_ms: int) -> CTATurnRecord:
    return CTATurnRecord(
        turn_index=1,
        user_text="u",
        assistant_text="a",
        question_type="STD4",
        backchannel_type="BC4",
        generation_mode="TEMPLATE_RANDOM",
        fallback_used=False,
        decision_reason="test",
        processing_latency_ms=latency_ms,
        subject_name="s",
        topic_name="t",
    )


def test_summarize_turn_latency_basic_stats() -> None:
    turns = [_turn(100), _turn(200), _turn(300), _turn(400), _turn(500)]
    summary = summarize_turn_latency(turns, target_p95_ms=450)

    assert summary.count == 5
    assert summary.p50_ms == 300
    assert summary.p95_ms == 500
    assert summary.max_ms == 500
    assert summary.mean_ms == 300.0
    assert summary.within_target is False


def test_summarize_turn_latency_empty() -> None:
    summary = summarize_turn_latency([], target_p95_ms=1000)

    assert summary.count == 0
    assert summary.p50_ms == 0
    assert summary.p95_ms == 0
    assert summary.within_target is True

