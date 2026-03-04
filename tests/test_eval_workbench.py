from __future__ import annotations

import json
from pathlib import Path

from src.evals.workbench import (
    apply_conversation_to_case,
    build_custom_case,
    case_to_conversation,
    ensure_case_defaults,
    export_cases_to_jsonl,
    initial_user_question,
    load_workbench_state,
    merge_base_cases_with_state,
    normalize_conversation,
    parse_context_lines,
    render_context_lines,
    save_workbench_state,
    upsert_case_in_state,
)


def test_parse_and_render_context_lines_roundtrip():
    raw = "user: 今日はどう？\nassistant: どこが気になる？\nそのままの行"
    context = parse_context_lines(raw)
    rendered = render_context_lines(context)

    assert context == [
        {"role": "user", "content": "今日はどう？"},
        {"role": "assistant", "content": "どこが気になる？"},
        {"role": "user", "content": "そのままの行"},
    ]
    assert "user: 今日はどう？" in rendered
    assert "assistant: どこが気になる？" in rendered


def test_state_save_and_load(tmp_path):
    state_path = tmp_path / "state.json"
    case = ensure_case_defaults(
        {
            "case_id": "case-1",
            "source": {"is_custom": False},
            "input": {"context": [], "user_input": "u"},
            "output": {"assistant_output": "a", "predicted_route": "CLARIFY"},
            "metadata": {},
            "labels": {},
        }
    )
    state = {"cases": {}}
    upsert_case_in_state(case, state)
    save_workbench_state(state_path, state)
    loaded = load_workbench_state(state_path)

    assert "case-1" in loaded["cases"]
    assert loaded["cases"]["case-1"]["metadata"]["dataset_type"] == "route_eval"


def test_merge_base_cases_with_state_prefers_saved_case():
    base = [
        ensure_case_defaults(
            {
                "case_id": "base-1",
                "source": {"assistant_timestamp": "2026-03-01T00:00:00Z", "is_custom": False},
                "input": {"context": [], "user_input": "u1"},
                "output": {"assistant_output": "a1", "predicted_route": "CLARIFY"},
                "metadata": {},
                "labels": {},
            }
        )
    ]
    saved = ensure_case_defaults(
        {
            "case_id": "base-1",
            "source": {"assistant_timestamp": "2026-03-01T00:00:00Z", "is_custom": False},
            "input": {"context": [], "user_input": "edited user"},
            "output": {"assistant_output": "edited assistant", "predicted_route": "DEEPEN"},
            "metadata": {"dataset_type": "regression", "edited": True},
            "labels": {"expected_route": "DEEPEN"},
        }
    )
    state = {"cases": {"base-1": saved}}

    merged = merge_base_cases_with_state(base, state)

    assert len(merged) == 1
    assert merged[0]["input"]["user_input"] == "edited user"
    assert merged[0]["metadata"]["dataset_type"] == "regression"


def test_custom_case_is_added_in_merge_when_not_in_base():
    state = {"cases": {}}
    custom = build_custom_case(
        dataset_type="good_response",
        user_input="u",
        assistant_output="a",
        context=[],
    )
    state["cases"][custom["case_id"]] = custom

    merged = merge_base_cases_with_state([], state)

    assert len(merged) == 1
    assert merged[0]["source"]["is_custom"] is True


def test_export_cases_to_jsonl_filters_edited(tmp_path):
    out_path = tmp_path / "cases.jsonl"
    cases = [
        ensure_case_defaults(
            {
                "case_id": "c1",
                "source": {"is_custom": False},
                "input": {"context": [], "user_input": "u"},
                "output": {"assistant_output": "a", "predicted_route": "CLARIFY"},
                "metadata": {"edited": True, "dataset_type": "route_eval"},
                "labels": {},
            }
        ),
        ensure_case_defaults(
            {
                "case_id": "c2",
                "source": {"is_custom": False},
                "input": {"context": [], "user_input": "u2"},
                "output": {"assistant_output": "a2", "predicted_route": "DEEPEN"},
                "metadata": {"edited": False, "dataset_type": "route_eval"},
                "labels": {},
            }
        ),
    ]

    exported_count = export_cases_to_jsonl(cases, out_path, only_edited=True)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert exported_count == 1
    assert len(rows) == 1
    assert rows[0]["case_id"] == "c1"


def test_case_to_conversation_builds_from_blocks():
    case = ensure_case_defaults(
        {
            "case_id": "c-conv",
            "source": {"is_custom": False},
            "input": {
                "context": [{"role": "assistant", "content": "最初の確認"}],
                "user_input": "ユーザーの疑問",
            },
            "output": {"assistant_output": "回答", "predicted_route": "CLARIFY"},
            "metadata": {},
            "labels": {},
        }
    )

    conversation = case_to_conversation(case)

    assert conversation == [
        {"role": "assistant", "content": "最初の確認"},
        {"role": "user", "content": "ユーザーの疑問"},
        {"role": "assistant", "content": "回答"},
    ]


def test_apply_conversation_to_case_updates_input_output():
    case = ensure_case_defaults(
        {
            "case_id": "c-apply",
            "source": {"is_custom": False},
            "input": {"context": [], "user_input": "old user"},
            "output": {"assistant_output": "old assistant", "predicted_route": "DEEPEN"},
            "metadata": {},
            "labels": {},
        }
    )
    conversation = [
        {"role": "assistant", "content": "前置き"},
        {"role": "user", "content": "新しい疑問"},
        {"role": "assistant", "content": "新しい返答"},
    ]

    updated = apply_conversation_to_case(case, conversation)

    assert updated["input"]["context"] == [{"role": "assistant", "content": "前置き"}]
    assert updated["input"]["user_input"] == "新しい疑問"
    assert updated["output"]["assistant_output"] == "新しい返答"
    assert updated["metadata"]["conversation"] == conversation


def test_initial_user_question_takes_first_user_message():
    case = ensure_case_defaults(
        {
            "case_id": "c-question",
            "source": {"is_custom": False},
            "input": {
                "context": [
                    {"role": "assistant", "content": "冒頭"},
                    {"role": "user", "content": "最初の疑問"},
                ],
                "user_input": "後段の疑問",
            },
            "output": {"assistant_output": "回答", "predicted_route": "CLARIFY"},
            "metadata": {},
            "labels": {},
        }
    )

    assert initial_user_question(case) == "最初の疑問"


def test_normalize_conversation_drops_empty_and_unknown_role():
    normalized = normalize_conversation(
        [
            {"role": "system", "content": "x"},
            {"role": "assistant", "content": "  "},
            {"role": "user", "content": "ok"},
        ]
    )

    assert normalized == [
        {"role": "user", "content": "x"},
        {"role": "user", "content": "ok"},
    ]
