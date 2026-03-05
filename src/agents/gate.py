"""Gate関連処理の後方互換ファサード。"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.agents.translator import translate_reasoning_to_japanese
from src.chains.gate_classifier import GateClassifierChain
from src.core.gate_trace_logging import log_gate_agent_trace
from src.core.models import GateDecision
from src.middleware.decision_guard import (
    apply_decision_overrides,
    build_clarify_completion_json as _build_clarify_completion_json,
)
from src.middleware.prompt_middleware import load_gate_prompt as _load_gate_prompt


def load_gate_prompt() -> str:
    """Gate判定に使用するsystem prompt本文を読み込む。"""
    return _load_gate_prompt()


def build_clarify_completion_json(
    user_input: str,
    chat_context: list | None = None,
) -> dict[str, str | bool | None]:
    """後方互換のために公開し続けるslot抽出ヘルパ。"""
    return _build_clarify_completion_json(user_input, chat_context)


def _apply_decision_overrides(
    decision: GateDecision,
    user_input: str,
    chat_context: list | None = None,
) -> GateDecision:
    """既存テスト互換のために残す decision override フック。"""
    return apply_decision_overrides(decision, user_input, chat_context)


def analyze_input(
    user_input: str,
    chat_context: list | None = None,
    user_images: list[dict[str, str]] | None = None,
) -> tuple[GateDecision, str | None, dict[str, int] | None]:
    """ユーザー入力を解析し、`GateDecision`と補助情報を返す。"""
    chain = GateClassifierChain(
        llm_factory=ChatOpenAI,
        reasoning_translator=translate_reasoning_to_japanese,
        trace_logger=log_gate_agent_trace,
    )
    return chain.classify(
        user_input=user_input,
        chat_context=chat_context,
        user_images=user_images,
    )
