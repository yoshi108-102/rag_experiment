"""モデル名・コンテキスト上限・token usage整形を扱う共通ユーティリティ。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
from typing import Any, Iterable, Mapping


DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000

_CJK_CHAR_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_GATE_MODEL_DEFAULT = "gpt-5.2"

_KNOWN_MODEL_CONTEXT_WINDOWS = {
    # Sources: OpenAI official model docs (checked 2026-02-27).
    "gpt-5.2": 400_000,
    "gpt-5.2-codex": 400_000,
    "gpt-5.2-pro": 400_000,
    "gpt-5.2-chat-latest": 128_000,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
}


@dataclass(frozen=True)
class ContextWindowLimit:
    """モデルのコンテキスト上限値と、その根拠情報を保持する。"""

    model: str
    max_tokens: int
    source: str


def default_gate_model_name() -> str:
    """Gate判定で利用するモデル名を環境変数込みで決定する。"""
    return (os.getenv("GATE_MODEL") or _GATE_MODEL_DEFAULT).strip()


def resolve_context_window_limit(model_name: str | None = None) -> ContextWindowLimit:
    """モデルの想定コンテキスト上限を、環境変数優先で解決する。"""
    override = _parse_positive_int(os.getenv("GATE_CONTEXT_WINDOW_TOKENS"))
    resolved_model = (model_name or default_gate_model_name()).strip()

    if override is not None:
        return ContextWindowLimit(
            model=resolved_model,
            max_tokens=override,
            source="env",
        )

    hinted = _lookup_known_context_window(resolved_model)
    if hinted is not None:
        return ContextWindowLimit(
            model=resolved_model,
            max_tokens=hinted,
            source="model-default",
        )

    return ContextWindowLimit(
        model=resolved_model,
        max_tokens=DEFAULT_CONTEXT_WINDOW_TOKENS,
        source="fallback",
    )


def context_limit_source_label(source: str) -> str:
    """上限値のソース識別子を、UI表示向けの日本語ラベルへ変換する。"""
    if source == "env":
        return "環境変数 (GATE_CONTEXT_WINDOW_TOKENS)"
    if source == "model-default":
        return "モデル既定値"
    return "既定フォールバック"


def estimate_text_tokens(text: str) -> int:
    """テキスト長からtoken数を簡易推定する。"""
    normalized = _WHITESPACE_PATTERN.sub(" ", text or "").strip()
    if not normalized:
        return 0

    cjk_count = len(_CJK_CHAR_PATTERN.findall(normalized))
    non_cjk_count = max(len(normalized) - cjk_count, 0)
    # CJK chars are often near 1 token, latin text is often ~4 chars/token.
    estimate = math.ceil((cjk_count * 1.1) + (non_cjk_count / 4))
    return max(estimate, 1)


def estimate_messages_tokens(
    messages: Iterable[Mapping[str, Any]],
    per_message_overhead: int = 4,
) -> int:
    """メッセージ配列全体のtoken数をオーバーヘッド込みで推定する。"""
    total = 0
    for message in messages:
        total += per_message_overhead
        total += estimate_text_tokens(str(message.get("content", "")))
        total += estimate_text_tokens(str(message.get("role", "")))
    return total


def usage_ratio(used_tokens: int, max_tokens: int) -> float:
    """使用量を0.0〜1.0の範囲に収めた比率で返す。"""
    if max_tokens <= 0:
        return 0.0
    ratio = used_tokens / max_tokens
    return max(0.0, min(ratio, 1.0))


def extract_token_usage(response: Any) -> dict[str, int] | None:
    """LangChainレスポンスからtoken使用量を抽出して正規化する。

    OpenAI Responses APIラッパーで一般的な複数格納先
    (`usage_metadata`, `response_metadata.token_usage`, `response_metadata.usage`)
    を順に探索する。
    """
    candidates: list[Mapping[str, Any]] = []

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, Mapping):
        candidates.append(usage_metadata)

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, Mapping):
            candidates.append(token_usage)

        usage = response_metadata.get("usage")
        if isinstance(usage, Mapping):
            candidates.append(usage)

    for candidate in candidates:
        normalized = _normalize_token_usage(candidate)
        if normalized is not None:
            return normalized
    return None


def _normalize_token_usage(payload: Mapping[str, Any]) -> dict[str, int] | None:
    """token使用量ペイロードを`input/output/total`形式へ正規化する。"""
    input_tokens = _coerce_int(payload.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _coerce_int(payload.get("prompt_tokens"))

    output_tokens = _coerce_int(payload.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _coerce_int(payload.get("completion_tokens"))

    total_tokens = _coerce_int(payload.get("total_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    usage: dict[str, int] = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    return usage


def _coerce_int(value: Any) -> int | None:
    """値を安全に`int`へ変換し、変換不能時は`None`を返す。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _parse_positive_int(value: str | None) -> int | None:
    """正の整数文字列をパースし、無効値なら`None`を返す。"""
    if not value:
        return None
    normalized = value.strip()
    if not normalized.isdigit():
        return None
    parsed = int(normalized)
    if parsed <= 0:
        return None
    return parsed


def _lookup_known_context_window(model_name: str) -> int | None:
    """既知モデル名またはスナップショット接頭辞から上限値を引く。"""
    if not model_name:
        return None
    exact = _KNOWN_MODEL_CONTEXT_WINDOWS.get(model_name)
    if exact is not None:
        return exact

    # Snapshot aliases like "gpt-5.2-2025-12-11".
    for prefix, max_tokens in _KNOWN_MODEL_CONTEXT_WINDOWS.items():
        if model_name.startswith(prefix + "-"):
            return max_tokens
    return None
