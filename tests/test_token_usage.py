from src.core.token_usage import (
    default_gate_model_name,
    estimate_messages_tokens,
    estimate_text_tokens,
    resolve_context_window_limit,
    extract_token_usage,
    usage_ratio,
)


def test_extract_token_usage_from_usage_metadata():
    class _Response:
        usage_metadata = {
            "input_tokens": 111,
            "output_tokens": 22,
            "total_tokens": 133,
        }
        response_metadata = {}

    usage = extract_token_usage(_Response())
    assert usage == {
        "input_tokens": 111,
        "output_tokens": 22,
        "total_tokens": 133,
    }


def test_extract_token_usage_from_response_metadata_token_usage():
    class _Response:
        usage_metadata = None
        response_metadata = {
            "token_usage": {
                "prompt_tokens": 91,
                "completion_tokens": 9,
            }
        }

    usage = extract_token_usage(_Response())
    assert usage == {
        "input_tokens": 91,
        "output_tokens": 9,
        "total_tokens": 100,
    }


def test_estimate_text_tokens_for_japanese_and_ascii():
    assert estimate_text_tokens("こんにちは") > 0
    assert estimate_text_tokens("hello world") > 0
    assert estimate_text_tokens("") == 0


def test_estimate_messages_tokens_and_ratio():
    messages = [
        {"role": "assistant", "content": "最初の案内です"},
        {"role": "user", "content": "この設計の意図を整理したい"},
    ]
    estimated = estimate_messages_tokens(messages)

    assert estimated > 0
    assert usage_ratio(estimated, estimated * 2) == 0.5
    assert usage_ratio(estimated, 0) == 0.0


def test_resolve_context_window_limit_from_model_default(monkeypatch):
    monkeypatch.delenv("GATE_CONTEXT_WINDOW_TOKENS", raising=False)

    info = resolve_context_window_limit("gpt-5.2")
    assert info.max_tokens == 400_000
    assert info.source == "model-default"


def test_resolve_context_window_limit_from_env_override(monkeypatch):
    monkeypatch.setenv("GATE_CONTEXT_WINDOW_TOKENS", "123456")

    info = resolve_context_window_limit("gpt-5.2")
    assert info.max_tokens == 123_456
    assert info.source == "env"


def test_default_gate_model_name_uses_env(monkeypatch):
    monkeypatch.setenv("GATE_MODEL", "gpt-5.2-chat-latest")
    assert default_gate_model_name() == "gpt-5.2-chat-latest"
