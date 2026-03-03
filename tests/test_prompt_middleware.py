from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from src.middleware.prompt_middleware import (
    OVERALL_CONTEXT_MODE_ENV,
    _should_include_overall_context,
)


def _request_with_human(content: str | list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(messages=[HumanMessage(content=content)])


def test_should_include_overall_context_auto_with_domain_keyword(monkeypatch):
    monkeypatch.setenv(OVERALL_CONTEXT_MODE_ENV, "auto")
    request = _request_with_human("パイプの曲がりが見づらいです")

    assert _should_include_overall_context(request) is True


def test_should_include_overall_context_auto_without_domain_keyword(monkeypatch):
    monkeypatch.setenv(OVERALL_CONTEXT_MODE_ENV, "auto")
    request = _request_with_human("今日は雑談だけしたいです")

    assert _should_include_overall_context(request) is False


def test_should_include_overall_context_always(monkeypatch):
    monkeypatch.setenv(OVERALL_CONTEXT_MODE_ENV, "always")
    request = _request_with_human("今日は雑談だけしたいです")

    assert _should_include_overall_context(request) is True


def test_should_include_overall_context_auto_when_image_is_attached(monkeypatch):
    monkeypatch.setenv(OVERALL_CONTEXT_MODE_ENV, "auto")
    request = _request_with_human(
        [
            {"type": "text", "text": "画像です"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ]
    )

    assert _should_include_overall_context(request) is True
