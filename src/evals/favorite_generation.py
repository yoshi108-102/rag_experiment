"""favoriteケースから手動修正前提の下書きケースを生成する処理。"""

from __future__ import annotations

from datetime import datetime, timezone
import random
from pathlib import Path
from typing import Any, Callable
import uuid

from src.evals.workbench import (
    DEFAULT_DATASET_TYPE,
    apply_conversation_to_case,
    case_to_conversation,
    ensure_case_defaults,
)

DEFAULT_DRAFT_COUNT = 10
FAVORITE_DEFAULT_EXPORT_DIR = "evals/cases/favorite"
GENERATED_DEFAULT_EXPORT_DIR = "evals/cases/generated"


def is_favorite_case(case: dict[str, Any]) -> bool:
    """caseがfavoriteとしてマークされているかを返す。"""
    metadata = case.get("metadata") or {}
    return isinstance(metadata, dict) and bool(metadata.get("favorite"))


def collect_favorite_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """ケース配列からfavoriteのみ抽出して正規化する。"""
    favorites: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        if not is_favorite_case(case):
            continue
        favorites.append(ensure_case_defaults(case))
    return favorites


def generate_cases_from_favorites(
    favorite_cases: list[dict[str, Any]],
    *,
    total_count: int = DEFAULT_DRAFT_COUNT,
    seed: int = 42,
    batch_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """favoriteケースを種に、手動修正向けの下書きケースを生成する。"""
    if total_count <= 0:
        return [], {
            "batch_id": batch_id or build_generation_batch_id(),
            "requested": total_count,
            "generated": 0,
            "favorite_count": 0,
            "parent_case_ids": [],
            "strategy_counts": {},
        }

    normalized_favorites = collect_favorite_cases(favorite_cases)
    if not normalized_favorites:
        raise ValueError("favorite case is required")

    resolved_batch_id = batch_id or build_generation_batch_id()
    rng = random.Random(seed)

    parent_pool = list(normalized_favorites)
    strategy_pool = list(_GENERATION_STRATEGIES)
    rng.shuffle(parent_pool)
    rng.shuffle(strategy_pool)

    generated_cases: list[dict[str, Any]] = []
    parent_case_ids: list[str] = []
    strategy_counts: dict[str, int] = {}

    for index in range(total_count):
        parent_case = parent_pool[index % len(parent_pool)]
        strategy_name, transform = strategy_pool[index % len(strategy_pool)]
        generated = _build_generated_case(
            parent_case=parent_case,
            strategy_name=strategy_name,
            transform=transform,
            batch_id=resolved_batch_id,
            variant_index=index,
        )
        generated_cases.append(generated)
        parent_case_id = str(parent_case.get("case_id") or "")
        parent_case_ids.append(parent_case_id)
        strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1

    summary = {
        "batch_id": resolved_batch_id,
        "requested": total_count,
        "generated": len(generated_cases),
        "favorite_count": len(normalized_favorites),
        "parent_case_ids": sorted({case_id for case_id in parent_case_ids if case_id}),
        "strategy_counts": strategy_counts,
    }
    return generated_cases, summary


def write_jsonl_cases(cases: list[dict[str, Any]], out_path: Path) -> None:
    """ケース配列をJSONLで保存する。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(f"{_to_json(case)}\n")


def build_generation_batch_id() -> str:
    """生成バッチIDを作る。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:6]}"


def _build_generated_case(
    *,
    parent_case: dict[str, Any],
    strategy_name: str,
    transform: Callable[[str], str],
    batch_id: str,
    variant_index: int,
) -> dict[str, Any]:
    parent = ensure_case_defaults(parent_case)
    parent_case_id = str(parent.get("case_id") or "")
    parent_metadata = parent.get("metadata") or {}
    parent_output = parent.get("output") or {}

    conversation = case_to_conversation(parent)
    variant_conversation = _apply_strategy_to_conversation(conversation, transform)

    now = _utc_now_iso()
    generated_case = {
        "case_id": f"gen-{uuid.uuid4().hex[:16]}",
        "source": {
            "session_id": None,
            "log_path": None,
            "user_message_index": None,
            "assistant_message_index": None,
            "user_timestamp": now,
            "assistant_timestamp": now,
            "is_custom": True,
            "generated_from_favorite": True,
            "parent_case_id": parent_case_id or None,
        },
        "input": {"context": [], "user_input": ""},
        "output": {
            "assistant_output": "",
            "predicted_route": str(parent_output.get("predicted_route") or ""),
            "predicted_reason": "favorite-draft-generated",
        },
        "metadata": {
            "dataset_type": str(parent_metadata.get("dataset_type") or DEFAULT_DATASET_TYPE),
            "edited": False,
            "favorite": False,
            "favorite_note": None,
            "token_usage": None,
            "rag": None,
            "conversation": variant_conversation,
            "created_at": now,
            "generation": {
                "batch_id": batch_id,
                "parent_case_id": parent_case_id or None,
                "strategy": strategy_name,
                "variant_index": variant_index,
            },
        },
        "labels": {
            "expected_route": None,
            "label_status": "unlabeled",
            "labeler": None,
            "label_note": f"generated from favorite ({strategy_name})",
        },
    }
    return apply_conversation_to_case(generated_case, variant_conversation)


def _apply_strategy_to_conversation(
    conversation: list[dict[str, str]],
    transform: Callable[[str], str],
) -> list[dict[str, str]]:
    normalized = [
        {"role": str(message.get("role") or "user"), "content": str(message.get("content") or "")}
        for message in conversation
        if isinstance(message, dict)
    ]
    if not normalized:
        return [{"role": "user", "content": transform("作業の振り返りを整理したいです。")}]

    target_index = _last_user_index(normalized)
    if target_index is None:
        normalized.append({"role": "user", "content": transform("作業の振り返りを整理したいです。")})
        return normalized

    base_text = str(normalized[target_index].get("content") or "").strip()
    if not base_text:
        base_text = "作業の振り返りを整理したいです。"
    normalized[target_index]["content"] = transform(base_text)
    return normalized


def _last_user_index(conversation: list[dict[str, str]]) -> int | None:
    for idx in range(len(conversation) - 1, -1, -1):
        if str(conversation[idx].get("role") or "").strip().lower() == "user":
            return idx
    return None


def _to_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _with_time_context(text: str) -> str:
    return f"今日の作業では、{text}"


def _with_constraint(text: str) -> str:
    return f"{text}。ただ、時間制約が厳しい状況でした。"


def _with_counterpoint(text: str) -> str:
    return f"{text}。一方で、うまくいった場面も少しありました。"


def _focus_peak(text: str) -> str:
    return f"{text}。特に難しかった瞬間はどこかを整理したいです。"


def _focus_reproducibility(text: str) -> str:
    return f"{text}。同じミスを再現なく減らす観点で見たいです。"


def _add_uncertainty(text: str) -> str:
    return f"{text}。ただ、表現が曖昧かもしれません。"


def _contrast_progress(text: str) -> str:
    return f"序盤は順調でしたが、途中から{text}"


def _add_body_factor(text: str) -> str:
    return f"{text}。姿勢や手の位置も影響していた気がします。"


def _add_goal(text: str) -> str:
    return f"{text}。最終的には作業の安定化を目指しています。"


def _compact_statement(text: str) -> str:
    compact = text.strip()
    if len(compact) <= 40:
        return f"要点だけ言うと、{compact}"
    return f"要点だけ言うと、{compact[:40]}..."


_GENERATION_STRATEGIES: list[tuple[str, Callable[[str], str]]] = [
    ("time_context", _with_time_context),
    ("constraint", _with_constraint),
    ("counterpoint", _with_counterpoint),
    ("focus_peak", _focus_peak),
    ("focus_reproducibility", _focus_reproducibility),
    ("uncertainty", _add_uncertainty),
    ("contrast_progress", _contrast_progress),
    ("body_factor", _add_body_factor),
    ("goal_oriented", _add_goal),
    ("compact_statement", _compact_statement),
]
