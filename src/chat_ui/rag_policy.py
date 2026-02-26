from __future__ import annotations

import hashlib
from typing import Any

from src.chat_ui.constants import (
    BOUNDARY_ROUTES,
    RAG_BUFFER_MAX_ITEMS,
    RAG_COOLDOWN_TURNS,
    RAG_STREAK_TRIGGER,
)


def update_idea_buffer(session_state: Any, user_input: str, route: str) -> None:
    session_state.rag_meta["turn_count"] += 1
    session_state.idea_buffer.append(
        {
            "user_input": user_input,
            "route": route,
        }
    )
    session_state.idea_buffer = session_state.idea_buffer[-RAG_BUFFER_MAX_ITEMS:]


def build_buffered_idea_query(session_state: Any) -> str:
    items = session_state.idea_buffer[-RAG_BUFFER_MAX_ITEMS:]
    if not items:
        return ""

    return "\n".join(f"[{item['route']}] {item['user_input']}" for item in items)


def recent_deepen_clarify_streak(session_state: Any) -> int:
    streak = 0
    for item in reversed(session_state.idea_buffer):
        if item["route"] in {"DEEPEN", "CLARIFY"}:
            streak += 1
            continue
        break
    return streak


def should_run_rag(session_state: Any, current_route: str) -> tuple[bool, str]:
    meta = session_state.rag_meta

    if current_route in BOUNDARY_ROUTES:
        return True, "boundary"

    turns_since_last_rag = meta["turn_count"] - meta["last_rag_turn"]
    if turns_since_last_rag <= RAG_COOLDOWN_TURNS:
        return False, "cooldown"

    if (
        current_route in {"DEEPEN", "CLARIFY"}
        and recent_deepen_clarify_streak(session_state) >= RAG_STREAK_TRIGGER
    ):
        return True, "streak"

    return False, "not-triggered"


def finalize_rag_run(session_state: Any, query: str, clear_buffer: bool = False) -> None:
    meta = session_state.rag_meta
    meta["last_rag_turn"] = meta["turn_count"]
    meta["last_rag_signature"] = hashlib.sha1(query.encode("utf-8")).hexdigest()
    if clear_buffer:
        session_state.idea_buffer = []


def clear_idea_buffer_if_boundary(session_state: Any, route: str) -> None:
    if route in BOUNDARY_ROUTES:
        session_state.idea_buffer = []


def should_skip_same_query(session_state: Any, query: str) -> bool:
    if not query.strip():
        return True
    if len(query.replace("\n", "").strip()) < 15:
        return True

    signature = hashlib.sha1(query.encode("utf-8")).hexdigest()
    return signature == session_state.rag_meta.get("last_rag_signature")
