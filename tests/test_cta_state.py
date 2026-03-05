from types import SimpleNamespace

from src.agents.gate import build_clarify_completion_json
from src.chat_ui.cta_state import build_initial_cta_state, update_cta_state


def test_build_clarify_completion_json_contains_cta_fields():
    payload = build_clarify_completion_json(
        "押した後の確認場面で見え方がぶれて、どこを押すか迷って難しかった"
    )

    assert "cta_slots" in payload
    assert "cta_next_focus" in payload
    assert "cta_is_complete" in payload
    assert payload["cta_filled_count"] >= 2
    assert payload["cta_completion_ratio"] > 0


def test_update_cta_state_merges_slots_without_dropping_previous_values():
    state = SimpleNamespace(cta_state=build_initial_cta_state())

    first_payload = {
        "cta_slots": {
            "situation": "押した後の確認場面",
            "perception": None,
            "decision": None,
            "action": "押した後に回して確認した",
            "difficulty": "見え方がぶれて難しかった",
        },
        "cta_next_focus": "perception",
        "cta_is_complete": False,
    }
    update_cta_state(state, first_payload, "CLARIFY")

    second_payload = {
        "cta_slots": {
            "situation": None,
            "perception": "奥側が二重に見えた",
            "decision": "先に奥側を優先して見ることにした",
            "action": None,
            "difficulty": None,
        },
        "cta_next_focus": None,
        "cta_is_complete": True,
    }
    merged = update_cta_state(state, second_payload, "CLARIFY")

    assert merged["slots"]["situation"] == "押した後の確認場面"
    assert merged["slots"]["action"] == "押した後に回して確認した"
    assert merged["slots"]["perception"] == "奥側が二重に見えた"
    assert merged["slots"]["decision"] == "先に奥側を優先して見ることにした"
    assert merged["is_complete"] is True
