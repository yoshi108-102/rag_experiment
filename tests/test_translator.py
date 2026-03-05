from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.agents.translator import (
    TRANSLATION_ENABLED_ENV,
    TRANSLATION_MODEL_ENV,
    translate_reasoning_to_japanese,
)


def test_translate_reasoning_returns_original_when_disabled(monkeypatch):
    monkeypatch.setenv(TRANSLATION_ENABLED_ENV, "0")

    result = translate_reasoning_to_japanese("Need broad gather first.")

    assert result == "Need broad gather first."


@patch("src.agents.translator.ChatOpenAI")
def test_translate_reasoning_calls_llm_when_enabled(mock_chat_openai, monkeypatch):
    monkeypatch.setenv(TRANSLATION_ENABLED_ENV, "1")
    monkeypatch.setenv(TRANSLATION_MODEL_ENV, "gpt-4o-mini")

    mock_llm = mock_chat_openai.return_value
    mock_llm.invoke.return_value = SimpleNamespace(content="まずは広く聞くべきです。")

    result = translate_reasoning_to_japanese("Need broad gather first.")

    assert result == "まずは広く聞くべきです。"
    mock_chat_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0.3)
    mock_llm.invoke.assert_called_once()
