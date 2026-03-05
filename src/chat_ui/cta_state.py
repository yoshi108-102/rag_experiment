"""CTAインタビュー進捗を`session_state`で保持する補助処理。"""

from __future__ import annotations

from typing import Any


CTA_SLOT_ORDER = ("situation", "perception", "decision", "action", "difficulty")


def build_initial_cta_state() -> dict[str, Any]:
    """CTA進捗の初期状態を返す。"""
    slots = {slot: None for slot in CTA_SLOT_ORDER}
    return {
        "slots": slots,
        "filled_count": 0,
        "completion_ratio": 0.0,
        "next_focus": "situation",
        "is_complete": False,
        "turn_count": 0,
        "last_route": None,
        "history": [],
    }


def ensure_cta_state(session_state: Any) -> None:
    """`session_state`にCTA進捗領域がなければ初期化する。"""
    has_key = False
    try:
        has_key = "cta_state" in session_state
    except TypeError:
        has_key = hasattr(session_state, "cta_state")

    if not has_key:
        session_state.cta_state = build_initial_cta_state()


def update_cta_state(
    session_state: Any,
    clarify_json: dict[str, Any] | None,
    route: str,
) -> dict[str, Any]:
    """抽出結果をマージしてCTA進捗を更新し、最新状態を返す。"""
    ensure_cta_state(session_state)
    state = session_state.cta_state
    state["turn_count"] = int(state.get("turn_count", 0)) + 1
    state["last_route"] = route

    if not clarify_json:
        return state

    latest_slots = clarify_json.get("cta_slots") or {}
    merged_slots = dict(state.get("slots") or {})
    for slot in CTA_SLOT_ORDER:
        value = latest_slots.get(slot)
        if isinstance(value, str) and value.strip():
            merged_slots[slot] = value.strip()
    state["slots"] = merged_slots

    filled_count = sum(1 for slot in CTA_SLOT_ORDER if merged_slots.get(slot))
    state["filled_count"] = filled_count
    state["completion_ratio"] = round(filled_count / len(CTA_SLOT_ORDER), 2)
    state["next_focus"] = clarify_json.get("cta_next_focus")
    state["is_complete"] = bool(clarify_json.get("cta_is_complete"))

    history = list(state.get("history") or [])
    history.append(
        {
            "route": route,
            "filled_count": filled_count,
            "next_focus": state["next_focus"],
            "is_complete": state["is_complete"],
        }
    )
    state["history"] = history[-20:]
    return state
