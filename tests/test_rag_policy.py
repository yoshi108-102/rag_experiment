from __future__ import annotations

from types import SimpleNamespace

from src.chat_ui.rag_policy import should_run_rag


def _session_state(
    *,
    turn_count: int,
    last_rag_turn: int,
    idea_buffer: list[dict[str, str]],
) -> SimpleNamespace:
    return SimpleNamespace(
        rag_meta={
            "turn_count": turn_count,
            "last_rag_turn": last_rag_turn,
            "last_rag_signature": None,
        },
        idea_buffer=idea_buffer,
    )


def test_should_run_rag_skips_boundary_routes():
    state = _session_state(
        turn_count=5,
        last_rag_turn=2,
        idea_buffer=[{"user_input": "今日はここまででいい", "route": "FINISH"}],
    )

    should_run, reason = should_run_rag(state, "FINISH")

    assert should_run is False
    assert reason == "boundary-skip"


def test_should_run_rag_runs_on_streak_when_not_in_cooldown():
    state = _session_state(
        turn_count=7,
        last_rag_turn=3,
        idea_buffer=[
            {"user_input": "見づらかった", "route": "CLARIFY"},
            {"user_input": "奥側が難しい", "route": "DEEPEN"},
            {"user_input": "押し量が迷う", "route": "CLARIFY"},
        ],
    )

    should_run, reason = should_run_rag(state, "CLARIFY")

    assert should_run is True
    assert reason == "streak"
