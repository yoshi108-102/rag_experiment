from __future__ import annotations

import collections
from types import SimpleNamespace

from src.chat_ui.cta_state import build_initial_cta_state
from src.chat_ui.turn_handler import handle_user_turn
from src.core.models import GateDecision


def _session_state() -> SimpleNamespace:
    return SimpleNamespace(
        llm_context=collections.deque(
            [{"role": "assistant", "content": "今日はどうでしたか？"}],
            maxlen=10,
        ),
        idea_buffer=[],
        rag_meta={
            "turn_count": 0,
            "last_rag_turn": -999,
            "last_rag_signature": None,
        },
        cta_state=build_initial_cta_state(),
    )


def test_handle_user_turn_applies_response_refinement(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.chat_ui.turn_handler.analyze_input",
        lambda prompt, chat_context, user_images=None: (
            GateDecision(
                route="CLARIFY",
                reason="Need detail",
                first_question="どの場面でそう感じましたか？",
            ),
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        "src.chat_ui.turn_handler.execute_route",
        lambda decision: decision.first_question,
    )
    monkeypatch.setattr(
        "src.chat_ui.turn_handler.refine_route_response",
        lambda draft_response, route, user_input, chat_context=None: SimpleNamespace(
            text="気になっていたんですね。どの場面でそう感じましたか？",
            enabled=True,
            fallback_used=False,
            model_name="gpt-4o-mini",
            error=None,
            draft_response=draft_response,
        ),
    )
    monkeypatch.setattr(
        "src.chat_ui.turn_handler._build_rag_debug",
        lambda session_state, route: {
            "enabled": False,
            "skipped_reason": "not-triggered",
            "trigger": None,
            "query": None,
        },
    )

    result = handle_user_turn("見えづらかったです", _session_state())

    assert result.response == "気になっていたんですね。どの場面でそう感じましたか？"
    assert result.debug_info["response_refinement"]["enabled"] is True
    assert result.debug_info["response_refinement"]["fallback_used"] is False
    assert result.debug_info["response_refinement"]["draft_response"] == "どの場面でそう感じましたか？"
