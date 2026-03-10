"""1ターン分の入力処理を束ね、判定・応答生成・RAGデバッグを構築する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.gate import analyze_input, build_clarify_completion_json
from src.agents.response_refiner import refine_route_response
from src.chat_ui.constants import BOUNDARY_ROUTES, RAG_ELIGIBLE_ROUTES
from src.chat_ui.cta_state import update_cta_state
from src.chat_ui.rag_policy import (
    build_buffered_idea_query,
    clear_idea_buffer_if_boundary,
    finalize_rag_run,
    should_run_rag,
    should_skip_same_query,
    update_idea_buffer,
)
from src.routing.router import execute_route
from src.tools.rag_tools import run_reflection_context_lookup


@dataclass(frozen=True)
class TurnResult:
    """1ターン処理の出力をまとめた不変データ。"""

    response: str
    reasoning: str | None
    debug_info: dict[str, Any]
    is_finished: bool


def handle_user_turn(
    prompt: str,
    session_state: Any,
    user_images: list[dict[str, str]] | None = None,
) -> TurnResult:
    """ユーザー入力1件を処理し、UI描画に必要な結果を返す。"""
    decision, reasoning, token_usage = analyze_input(
        prompt,
        list(session_state.llm_context),
        user_images=user_images,
    )
    draft_response = execute_route(decision)
    refinement = refine_route_response(
        draft_response,
        route=decision.route,
        user_input=prompt,
        chat_context=list(session_state.llm_context),
    )
    response = refinement.text
    clarify_json = build_clarify_completion_json(prompt, list(session_state.llm_context))
    cta_state = update_cta_state(session_state, clarify_json, decision.route)

    update_idea_buffer(session_state, prompt, decision.route)
    rag_debug = _build_rag_debug(session_state, decision.route)

    debug_info = {
        "route": decision.route,
        "reason": decision.reason,
        "reasoning": reasoning,
        "rag": rag_debug,
        "clarify_json": clarify_json,
        "cta_state": cta_state,
        "token_usage": token_usage,
        "response_refinement": {
            "enabled": refinement.enabled,
            "fallback_used": refinement.fallback_used,
            "model_name": refinement.model_name,
            "error": refinement.error,
            "draft_response": refinement.draft_response,
        },
    }

    return TurnResult(
        response=response,
        reasoning=reasoning,
        debug_info=debug_info,
        is_finished=decision.route == "FINISH",
    )


def _build_rag_debug(session_state: Any, route: str) -> dict[str, Any]:
    """現在ターンのRAG実行可否を判定し、デバッグ表示用情報を構築する。"""
    rag_debug: dict[str, Any] = {
        "enabled": False,
        "skipped_reason": "not-triggered",
        "trigger": None,
        "query": None,
    }

    should_rag, rag_trigger = should_run_rag(session_state, route)
    if not should_rag:
        rag_debug["skipped_reason"] = rag_trigger
        clear_idea_buffer_if_boundary(session_state, route)
        return rag_debug

    rag_query = build_buffered_idea_query(session_state)
    rag_debug["trigger"] = rag_trigger
    rag_debug["query"] = rag_query

    if should_skip_same_query(session_state, rag_query):
        rag_debug["skipped_reason"] = "same-or-too-short-query"
        clear_idea_buffer_if_boundary(session_state, route)
        return rag_debug

    context_analysis = run_reflection_context_lookup(
        rag_query,
        route,
        allowed_routes=RAG_ELIGIBLE_ROUTES,
    )
    rag_debug = context_analysis.to_dict()
    rag_debug["trigger"] = rag_trigger
    rag_debug["query"] = rag_query

    finalize_rag_run(
        session_state,
        rag_query,
        clear_buffer=route in BOUNDARY_ROUTES,
    )
    return rag_debug
