from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.agents.response_refiner import (
    RESPONSE_REFINER_ENABLED_ENV,
    RESPONSE_REFINER_MODEL_ENV,
    load_response_refiner_prompt,
    refine_route_response,
)


def test_load_response_refiner_prompt() -> None:
    prompt = load_response_refiner_prompt()

    assert "後段エージェントB" in prompt
    assert "質問数は元の文面から増やさない" in prompt


def test_refine_route_response_returns_original_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv(RESPONSE_REFINER_ENABLED_ENV, "0")

    result = refine_route_response(
        "どの場面でそう感じましたか？",
        route="CLARIFY",
        user_input="見えづらかったです",
    )

    assert result.text == "どの場面でそう感じましたか？"
    assert result.enabled is False
    assert result.fallback_used is False


@patch("src.agents.response_refiner.ChatOpenAI")
def test_refine_route_response_calls_llm_when_enabled(mock_chat_openai, monkeypatch) -> None:
    monkeypatch.setenv(RESPONSE_REFINER_ENABLED_ENV, "1")
    monkeypatch.setenv(RESPONSE_REFINER_MODEL_ENV, "gpt-4o-mini")

    mock_llm = mock_chat_openai.return_value
    mock_llm.invoke.return_value = SimpleNamespace(
        content="気になっていたんですね。どの場面でそう感じましたか？"
    )

    result = refine_route_response(
        "どの場面でそう感じましたか？",
        route="CLARIFY",
        user_input="見えづらかったです",
        chat_context=[
            {"role": "assistant", "content": "今日はどうでしたか？"},
            {"role": "user", "content": "見えづらかったです"},
        ],
    )

    assert result.text == "気になっていたんですね。どの場面でそう感じましたか？"
    assert result.enabled is True
    assert result.fallback_used is False
    mock_chat_openai.assert_called_once_with(
        model="gpt-4o-mini",
        temperature=0.2,
        timeout=15.0,
    )
    mock_llm.invoke.assert_called_once()


@patch("src.agents.response_refiner.ChatOpenAI")
def test_refine_route_response_falls_back_when_multiple_questions_added(
    mock_chat_openai,
    monkeypatch,
) -> None:
    monkeypatch.setenv(RESPONSE_REFINER_ENABLED_ENV, "1")

    mock_llm = mock_chat_openai.return_value
    mock_llm.invoke.return_value = SimpleNamespace(
        content="気になっていたんですね。どの場面でしたか？そのとき何が見えていましたか？"
    )

    result = refine_route_response(
        "どの場面でそう感じましたか？",
        route="CLARIFY",
        user_input="見えづらかったです",
    )

    assert result.text == "どの場面でそう感じましたか？"
    assert result.enabled is True
    assert result.fallback_used is True
    assert result.error is not None
