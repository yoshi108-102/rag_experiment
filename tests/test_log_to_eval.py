from __future__ import annotations

import json
from pathlib import Path

from src.evals.log_to_eval import (
    build_eval_dataset,
    dedupe_drafts,
    extract_eval_case_drafts,
    parse_route_quota,
    sample_drafts,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _message(
    *,
    session_id: str,
    timestamp: str,
    role: str,
    content: str,
    message_index: int,
    debug_info: dict | None = None,
) -> dict:
    payload = {
        "role": role,
        "content": content,
        "message_index": message_index,
    }
    if debug_info is not None:
        payload["debug_info"] = debug_info
    return {
        "timestamp": timestamp,
        "session_id": session_id,
        "app_name": "streamlit",
        "event_type": "message",
        "payload": payload,
    }


def test_extract_eval_case_drafts_pairs_user_with_next_assistant(tmp_path):
    log_path = tmp_path / "sample.jsonl"
    rows = [
        _message(
            session_id="s-1",
            timestamp="2026-03-01T00:00:00Z",
            role="assistant",
            content="hello",
            message_index=0,
        ),
        _message(
            session_id="s-1",
            timestamp="2026-03-01T00:00:01Z",
            role="user",
            content="曲がりが見づらい",
            message_index=1,
        ),
        _message(
            session_id="s-1",
            timestamp="2026-03-01T00:00:02Z",
            role="assistant",
            content="どの場面で見づらかった？",
            message_index=2,
            debug_info={"route": "CLARIFY", "reason": "need detail", "token_usage": {"total_tokens": 123}},
        ),
    ]
    _write_jsonl(log_path, rows)

    drafts = extract_eval_case_drafts([log_path], context_turns=1, min_user_chars=2)

    assert len(drafts) == 1
    assert drafts[0].predicted_route == "CLARIFY"
    assert drafts[0].user_input == "曲がりが見づらい"
    assert drafts[0].assistant_output == "どの場面で見づらかった？"
    assert len(drafts[0].context) == 1


def test_dedupe_drafts_user_and_route_mode(tmp_path):
    log_path = tmp_path / "sample.jsonl"
    rows = [
        _message(
            session_id="s-2",
            timestamp="2026-03-01T00:00:00Z",
            role="user",
            content="同じ内容です",
            message_index=1,
        ),
        _message(
            session_id="s-2",
            timestamp="2026-03-01T00:00:01Z",
            role="assistant",
            content="どこが困った？",
            message_index=2,
            debug_info={"route": "CLARIFY", "reason": "x"},
        ),
        _message(
            session_id="s-2",
            timestamp="2026-03-01T00:00:02Z",
            role="user",
            content="同じ内容です",
            message_index=3,
        ),
        _message(
            session_id="s-2",
            timestamp="2026-03-01T00:00:03Z",
            role="assistant",
            content="追加で教えて",
            message_index=4,
            debug_info={"route": "CLARIFY", "reason": "y"},
        ),
    ]
    _write_jsonl(log_path, rows)

    drafts = extract_eval_case_drafts([log_path], context_turns=0, min_user_chars=2)
    deduped = dedupe_drafts(drafts, mode="user_and_route")

    assert len(drafts) == 2
    assert len(deduped) == 1


def test_parse_route_quota():
    quota = parse_route_quota("CLARIFY=50,DEEPEN=20,FINISH=20,PARK=10")

    assert quota == {
        "CLARIFY": 50,
        "DEEPEN": 20,
        "FINISH": 20,
        "PARK": 10,
    }


def test_sample_drafts_with_quota(tmp_path):
    log_path = tmp_path / "sample.jsonl"
    rows = []
    for i in range(3):
        rows.append(
            _message(
                session_id="s-3",
                timestamp=f"2026-03-01T00:00:0{i}Z",
                role="user",
                content=f"clarify {i}",
                message_index=i * 2 + 1,
            )
        )
        rows.append(
            _message(
                session_id="s-3",
                timestamp=f"2026-03-01T00:00:1{i}Z",
                role="assistant",
                content="c",
                message_index=i * 2 + 2,
                debug_info={"route": "CLARIFY", "reason": "c"},
            )
        )
    rows.extend(
        [
            _message(
                session_id="s-3",
                timestamp="2026-03-01T00:00:20Z",
                role="user",
                content="finish case",
                message_index=99,
            ),
            _message(
                session_id="s-3",
                timestamp="2026-03-01T00:00:21Z",
                role="assistant",
                content="f",
                message_index=100,
                debug_info={"route": "FINISH", "reason": "f"},
            ),
        ]
    )
    _write_jsonl(log_path, rows)

    drafts = extract_eval_case_drafts([log_path], context_turns=0, min_user_chars=2)
    selected = sample_drafts(
        drafts,
        max_cases=3,
        seed=123,
        route_quota={"CLARIFY": 2, "FINISH": 1},
    )
    routes = sorted(item.predicted_route for item in selected)

    assert routes == ["CLARIFY", "CLARIFY", "FINISH"]


def test_build_eval_dataset_returns_unlabeled_cases(tmp_path):
    log_path = tmp_path / "sample.jsonl"
    rows = [
        _message(
            session_id="s-4",
            timestamp="2026-03-01T00:00:00Z",
            role="user",
            content="評価したい",
            message_index=1,
        ),
        _message(
            session_id="s-4",
            timestamp="2026-03-01T00:00:01Z",
            role="assistant",
            content="どこが気になった？",
            message_index=2,
            debug_info={"route": "DEEPEN", "reason": "x"},
        ),
    ]
    _write_jsonl(log_path, rows)

    cases, result = build_eval_dataset([log_path], max_cases=10)

    assert result.total_candidates == 1
    assert result.selected == 1
    assert cases[0]["labels"]["expected_route"] is None
    assert cases[0]["labels"]["label_status"] == "unlabeled"


def test_build_eval_dataset_compacts_rag_payload(tmp_path):
    log_path = tmp_path / "sample.jsonl"
    rows = [
        _message(
            session_id="s-5",
            timestamp="2026-03-01T00:00:00Z",
            role="user",
            content="テストです",
            message_index=1,
        ),
        _message(
            session_id="s-5",
            timestamp="2026-03-01T00:00:01Z",
            role="assistant",
            content="質問です",
            message_index=2,
            debug_info={
                "route": "CLARIFY",
                "reason": "x",
                "rag": {
                    "enabled": True,
                    "trigger": "streak",
                    "retrieved": [{"score": 0.56789, "record": {"topic": "x"}}],
                    "novelty": {"is_novel": False, "confidence": 0.8, "reason": "r"},
                    "saved_pending": False,
                },
            },
        ),
    ]
    _write_jsonl(log_path, rows)

    cases, _ = build_eval_dataset([log_path], max_cases=10)
    rag = cases[0]["metadata"]["rag"]

    assert rag["enabled"] is True
    assert rag["trigger"] == "streak"
    assert rag["top_score"] == 0.5679
    assert rag["novelty"]["is_novel"] is False
    assert "retrieved" not in rag
