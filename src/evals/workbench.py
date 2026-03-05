"""Evalデータセット編集ワークベンチ向けの共通処理。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any


DEFAULT_DATASET_TYPE = "route_eval"
DATASET_TYPE_OPTIONS = [
    "route_eval",
    "good_response",
    "regression",
    "adversarial",
    "custom",
]
ROUTE_OPTIONS = ["", "DEEPEN", "CLARIFY", "PARK", "FINISH"]
CHAT_ROLE_OPTIONS = ["user", "assistant"]


def load_workbench_state(path: Path) -> dict[str, Any]:
    """ワークベンチ状態JSONを読み込む。壊れている場合は空状態を返す。"""
    if not path.exists():
        return _empty_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_state()
    if not isinstance(payload, dict):
        return _empty_state()

    cases = payload.get("cases")
    if not isinstance(cases, dict):
        cases = {}

    return {
        "version": 1,
        "updated_at": str(payload.get("updated_at") or ""),
        "cases": cases,
    }


def save_workbench_state(path: Path, state: dict[str, Any]) -> None:
    """ワークベンチ状態JSONを保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": _utc_now_iso(),
        "cases": state.get("cases", {}),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_base_cases_with_state(
    base_cases: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """ログ由来ケースと保存済み編集ケースをマージする。"""
    state_cases = state.get("cases") or {}
    if not isinstance(state_cases, dict):
        state_cases = {}

    merged: list[dict[str, Any]] = []
    base_ids: set[str] = set()
    for case in base_cases:
        case_id = str(case.get("case_id") or "")
        if not case_id:
            continue
        base_ids.add(case_id)
        if case_id in state_cases and isinstance(state_cases[case_id], dict):
            merged.append(ensure_case_defaults(state_cases[case_id]))
        else:
            merged.append(ensure_case_defaults(case))

    for case_id, saved_case in state_cases.items():
        if case_id in base_ids:
            continue
        if not isinstance(saved_case, dict):
            continue
        source = saved_case.get("source") or {}
        if isinstance(source, dict) and source.get("is_custom"):
            merged.append(ensure_case_defaults(saved_case))

    merged.sort(
        key=lambda case: _sort_key_for_case(case),
        reverse=True,
    )
    return merged


def ensure_case_defaults(case: dict[str, Any]) -> dict[str, Any]:
    """ケース辞書にワークベンチ用の既定項目を補う。"""
    normalized = json.loads(json.dumps(case, ensure_ascii=False))

    metadata = normalized.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        normalized["metadata"] = metadata
    metadata.setdefault("dataset_type", DEFAULT_DATASET_TYPE)
    metadata.setdefault("edited", False)
    metadata.setdefault("favorite", False)
    metadata.setdefault("favorite_note", None)

    labels = normalized.setdefault("labels", {})
    if not isinstance(labels, dict):
        labels = {}
        normalized["labels"] = labels
    labels.setdefault("expected_route", None)
    labels.setdefault("label_status", "unlabeled")
    labels.setdefault("labeler", None)
    labels.setdefault("label_note", None)

    input_block = normalized.setdefault("input", {})
    if not isinstance(input_block, dict):
        input_block = {}
        normalized["input"] = input_block
    input_block.setdefault("context", [])
    input_block.setdefault("user_input", "")

    output_block = normalized.setdefault("output", {})
    if not isinstance(output_block, dict):
        output_block = {}
        normalized["output"] = output_block
    output_block.setdefault("assistant_output", "")
    output_block.setdefault("predicted_route", "")
    output_block.setdefault("predicted_reason", "")

    raw_conversation = metadata.get("conversation")
    if isinstance(raw_conversation, list):
        metadata["conversation"] = normalize_conversation(raw_conversation)
    else:
        metadata["conversation"] = _build_conversation_from_blocks(input_block, output_block)

    source = normalized.setdefault("source", {})
    if not isinstance(source, dict):
        source = {}
        normalized["source"] = source
    source.setdefault("is_custom", False)

    if not normalized.get("case_id"):
        normalized["case_id"] = f"generated-{uuid.uuid4().hex[:16]}"

    return normalized


def build_custom_case(
    *,
    dataset_type: str,
    user_input: str,
    assistant_output: str,
    context: list[dict[str, str]],
    predicted_route: str | None = None,
    expected_route: str | None = None,
    label_note: str | None = None,
) -> dict[str, Any]:
    """手動作成ケースをEvalケース形式で生成する。"""
    case_id = f"custom-{uuid.uuid4().hex[:16]}"
    now = _utc_now_iso()
    case = {
        "case_id": case_id,
        "source": {
            "session_id": None,
            "log_path": None,
            "user_message_index": None,
            "assistant_message_index": None,
            "user_timestamp": now,
            "assistant_timestamp": now,
            "is_custom": True,
        },
        "input": {
            "context": context,
            "user_input": user_input,
        },
        "output": {
            "assistant_output": assistant_output,
            "predicted_route": predicted_route or "",
            "predicted_reason": "",
        },
        "metadata": {
            "dataset_type": dataset_type or DEFAULT_DATASET_TYPE,
            "edited": True,
            "token_usage": None,
            "rag": None,
            "conversation": [*context, {"role": "user", "content": user_input}, {"role": "assistant", "content": assistant_output}],
            "created_at": now,
        },
        "labels": {
            "expected_route": expected_route,
            "label_status": "labeled" if expected_route else "unlabeled",
            "labeler": None,
            "label_note": label_note or None,
        },
    }
    return ensure_case_defaults(case)


def upsert_case_in_state(case: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """状態辞書にケースを上書き保存する。"""
    cases = state.setdefault("cases", {})
    if not isinstance(cases, dict):
        cases = {}
        state["cases"] = cases
    normalized = ensure_case_defaults(case)
    cases[str(normalized["case_id"])] = normalized
    return state


def delete_case_from_state(case_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """状態辞書からケースを削除する。"""
    cases = state.get("cases")
    if isinstance(cases, dict):
        cases.pop(case_id, None)
    return state


def parse_context_lines(raw: str) -> list[dict[str, str]]:
    """`user: ...` / `assistant: ...` 形式の複数行をcontext配列へ変換する。"""
    contexts: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        role = "user"
        content = line
        if ":" in line:
            left, right = line.split(":", 1)
            left_normalized = left.strip().lower()
            if left_normalized in {"user", "assistant"}:
                role = left_normalized
                content = right.strip()
        if not content:
            continue
        contexts.append({"role": role, "content": content})
    return contexts


def render_context_lines(context: list[dict[str, str]] | None) -> str:
    """context配列を編集用テキストへシリアライズする。"""
    if not context:
        return ""
    lines: list[str] = []
    for item in context:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower() or "user"
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def export_cases_to_jsonl(
    cases: list[dict[str, Any]],
    out_path: Path,
    *,
    only_edited: bool = True,
    dataset_types: list[str] | None = None,
) -> int:
    """ケース配列を条件付きでJSONLへ出力する。"""
    filtered: list[dict[str, Any]] = []
    selected_types = set(dataset_types or [])
    for case in cases:
        metadata = case.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        if only_edited and not bool(metadata.get("edited")):
            continue
        dataset_type = str(metadata.get("dataset_type") or "")
        if selected_types and dataset_type not in selected_types:
            continue
        filtered.append(case)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(filtered)


def build_conversation_jsonl_payload(conversation: list[Any]) -> str:
    """会話（role/content）だけを1行JSONL向けJSON文字列にする。"""
    normalized = normalize_conversation(conversation)
    payload = {
        "conversation": [
            {
                "role": item["role"],
                "content": item["content"],
            }
            for item in normalized
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


def case_to_conversation(case: dict[str, Any]) -> list[dict[str, str]]:
    """ケースから会話配列を取得する。"""
    metadata = case.get("metadata") or {}
    if isinstance(metadata, dict):
        conversation = metadata.get("conversation")
        if isinstance(conversation, list):
            return normalize_conversation(conversation)
    input_block = case.get("input") or {}
    output_block = case.get("output") or {}
    return _build_conversation_from_blocks(input_block, output_block)


def apply_conversation_to_case(case: dict[str, Any], conversation: list[dict[str, str]]) -> dict[str, Any]:
    """編集済み会話配列をケースの input/output/metadata へ反映する。"""
    normalized_case = ensure_case_defaults(case)
    normalized_conversation = normalize_conversation(conversation)
    metadata = normalized_case.setdefault("metadata", {})
    input_block = normalized_case.setdefault("input", {})
    output_block = normalized_case.setdefault("output", {})

    metadata["conversation"] = normalized_conversation

    if not normalized_conversation:
        input_block["context"] = []
        input_block["user_input"] = ""
        output_block["assistant_output"] = ""
        return normalized_case

    last_user_index = None
    for idx in range(len(normalized_conversation) - 1, -1, -1):
        if normalized_conversation[idx]["role"] == "user":
            last_user_index = idx
            break

    if last_user_index is None:
        input_block["context"] = normalized_conversation[:-1]
        input_block["user_input"] = normalized_conversation[-1]["content"]
    else:
        input_block["context"] = normalized_conversation[:last_user_index]
        input_block["user_input"] = normalized_conversation[last_user_index]["content"]

    assistant_output = ""
    if last_user_index is not None:
        for idx in range(last_user_index + 1, len(normalized_conversation)):
            if normalized_conversation[idx]["role"] == "assistant":
                assistant_output = normalized_conversation[idx]["content"]
                break
    if not assistant_output:
        for idx in range(len(normalized_conversation) - 1, -1, -1):
            if normalized_conversation[idx]["role"] == "assistant":
                assistant_output = normalized_conversation[idx]["content"]
                break
    output_block["assistant_output"] = assistant_output
    return normalized_case


def delete_conversation_messages_by_index(
    conversation: list[Any],
    delete_indexes: set[int] | list[int] | tuple[int, ...],
) -> list[dict[str, str]]:
    """会話配列から、指定indexのメッセージだけを削除して正規化する。"""
    delete_set: set[int] = set()
    for raw_index in delete_indexes:
        if isinstance(raw_index, bool):
            continue
        if isinstance(raw_index, int):
            delete_set.add(raw_index)

    rebuilt: list[dict[str, str]] = []
    for index, message in enumerate(conversation):
        if index in delete_set:
            continue
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "user").strip().lower()
        if role not in CHAT_ROLE_OPTIONS:
            role = "user"
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        rebuilt.append({"role": role, "content": content})

    return rebuilt


def initial_user_question(case: dict[str, Any]) -> str:
    """ボード表示用に、会話内の最初の user 発話を返す。"""
    for message in case_to_conversation(case):
        if message["role"] == "user":
            return message["content"]
    return str((case.get("input") or {}).get("user_input") or "")


def normalize_conversation(conversation: list[Any]) -> list[dict[str, str]]:
    """会話配列を role/content の最小構造へ正規化する。"""
    normalized: list[dict[str, str]] = []
    for message in conversation:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower()
        if role not in CHAT_ROLE_OPTIONS:
            role = "user"
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _sort_key_for_case(case: dict[str, Any]) -> str:
    source = case.get("source") or {}
    if not isinstance(source, dict):
        return ""
    return str(
        source.get("assistant_timestamp")
        or source.get("user_timestamp")
        or case.get("case_id")
        or ""
    )


def _empty_state() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": "",
        "cases": {},
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _build_conversation_from_blocks(
    input_block: dict[str, Any] | Any,
    output_block: dict[str, Any] | Any,
) -> list[dict[str, str]]:
    context = input_block.get("context") if isinstance(input_block, dict) else []
    user_input = input_block.get("user_input") if isinstance(input_block, dict) else ""
    assistant_output = output_block.get("assistant_output") if isinstance(output_block, dict) else ""

    conversation: list[dict[str, str]] = normalize_conversation(context if isinstance(context, list) else [])
    if str(user_input or "").strip():
        conversation.append({"role": "user", "content": str(user_input).strip()})
    if str(assistant_output or "").strip():
        conversation.append({"role": "assistant", "content": str(assistant_output).strip()})
    return conversation
