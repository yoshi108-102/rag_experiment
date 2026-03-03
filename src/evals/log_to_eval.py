"""chat_sessions JSONL から評価用ケースを抽出・整形する。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable


ROUTES = ("DEEPEN", "CLARIFY", "PARK", "FINISH")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class EvalCaseDraft:
    """ログ由来の評価候補1件。"""

    session_id: str
    source_log: str
    user_message_index: int | None
    assistant_message_index: int | None
    user_timestamp: str | None
    assistant_timestamp: str | None
    user_input: str
    assistant_output: str
    predicted_route: str
    predicted_reason: str | None
    token_usage: dict[str, int] | None
    rag: dict[str, Any] | None
    context: list[dict[str, str]]

    def to_eval_case(self) -> dict[str, Any]:
        raw_id = (
            f"{self.session_id}|{self.user_message_index}|"
            f"{self.assistant_message_index}|{self.user_input}|{self.assistant_output}"
        )
        case_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
        return {
            "case_id": case_id,
            "source": {
                "session_id": self.session_id,
                "log_path": self.source_log,
                "user_message_index": self.user_message_index,
                "assistant_message_index": self.assistant_message_index,
                "user_timestamp": self.user_timestamp,
                "assistant_timestamp": self.assistant_timestamp,
            },
            "input": {
                "context": self.context,
                "user_input": self.user_input,
            },
            "output": {
                "assistant_output": self.assistant_output,
                "predicted_route": self.predicted_route,
                "predicted_reason": self.predicted_reason,
            },
            "metadata": {
                "token_usage": self.token_usage,
                "rag": self.rag,
                "created_at": _utc_now_iso(),
            },
            "labels": {
                "expected_route": None,
                "label_status": "unlabeled",
                "labeler": None,
                "label_note": None,
            },
        }


@dataclass(frozen=True)
class BuildResult:
    """データセット生成結果のサマリ。"""

    total_candidates: int
    after_dedup: int
    selected: int
    route_counts: dict[str, int]
    output_path: str


def build_eval_dataset(
    log_paths: Iterable[Path],
    *,
    max_cases: int = 100,
    context_turns: int = 2,
    min_user_chars: int = 4,
    dedupe_mode: str = "user_and_route",
    seed: int = 42,
    route_quota: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], BuildResult]:
    """ログ群から評価ケースを構築する。"""
    drafts = extract_eval_case_drafts(
        log_paths,
        context_turns=context_turns,
        min_user_chars=min_user_chars,
    )
    deduped = dedupe_drafts(drafts, mode=dedupe_mode)
    selected = sample_drafts(
        deduped,
        max_cases=max_cases,
        seed=seed,
        route_quota=route_quota,
    )
    cases = [draft.to_eval_case() for draft in selected]
    route_counts = Counter(draft.predicted_route for draft in selected)
    result = BuildResult(
        total_candidates=len(drafts),
        after_dedup=len(deduped),
        selected=len(selected),
        route_counts=dict(route_counts),
        output_path="",
    )
    return cases, result


def extract_eval_case_drafts(
    log_paths: Iterable[Path],
    *,
    context_turns: int = 2,
    min_user_chars: int = 4,
) -> list[EvalCaseDraft]:
    """chat session JSONLから user->assistant ペアを抽出する。"""
    drafts: list[EvalCaseDraft] = []
    context_messages = max(0, context_turns) * 2

    for path in sorted(log_paths):
        records = _read_jsonl(path)
        message_records = [item for item in records if item.get("event_type") == "message"]
        messages: list[dict[str, Any]] = []
        for record in message_records:
            payload = record.get("payload") or {}
            role = str(payload.get("role", ""))
            content = str(payload.get("content", "") or "")
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": str(record.get("timestamp") or ""),
                    "message_index": payload.get("message_index"),
                    "debug_info": payload.get("debug_info"),
                    "session_id": str(record.get("session_id") or ""),
                }
            )

        for idx, current in enumerate(messages):
            if current["role"] != "user":
                continue
            user_text = current["content"].strip()
            if len(_normalize_text(user_text)) < min_user_chars:
                continue

            if idx + 1 >= len(messages):
                continue
            next_message = messages[idx + 1]
            if next_message["role"] != "assistant":
                continue

            debug = next_message.get("debug_info")
            if not isinstance(debug, dict):
                continue

            route = str(debug.get("route") or "")
            if route not in ROUTES:
                continue

            context_slice = messages[max(0, idx - context_messages):idx]
            context = [
                {
                    "role": str(item.get("role") or ""),
                    "content": str(item.get("content") or ""),
                }
                for item in context_slice
            ]

            token_usage = debug.get("token_usage")
            rag = _compact_rag_payload(debug.get("rag"))
            drafts.append(
                EvalCaseDraft(
                    session_id=current["session_id"],
                    source_log=str(path),
                    user_message_index=_safe_int(current.get("message_index")),
                    assistant_message_index=_safe_int(next_message.get("message_index")),
                    user_timestamp=current["timestamp"] or None,
                    assistant_timestamp=next_message["timestamp"] or None,
                    user_input=user_text,
                    assistant_output=str(next_message.get("content") or ""),
                    predicted_route=route,
                    predicted_reason=_as_optional_str(debug.get("reason")),
                    token_usage=token_usage if isinstance(token_usage, dict) else None,
                    rag=rag,
                    context=context,
                )
            )

    return drafts


def dedupe_drafts(drafts: list[EvalCaseDraft], *, mode: str = "user_and_route") -> list[EvalCaseDraft]:
    """重複キーで候補を削減する。"""
    seen: set[str] = set()
    deduped: list[EvalCaseDraft] = []
    for draft in drafts:
        key = dedupe_key_for_draft(draft, mode=mode)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(draft)
    return deduped


def dedupe_key_for_draft(draft: EvalCaseDraft, *, mode: str = "user_and_route") -> str:
    """候補の重複キーを生成する。"""
    user_norm = _normalize_text(draft.user_input)
    response_norm = _normalize_text(draft.assistant_output)

    if mode == "user_only":
        return user_norm
    if mode == "user_and_response":
        return f"{user_norm}|{response_norm}"
    if mode == "user_and_route":
        return f"{draft.predicted_route}|{user_norm}"
    raise ValueError(f"Unknown dedupe mode: {mode}")


def sample_drafts(
    drafts: list[EvalCaseDraft],
    *,
    max_cases: int = 100,
    seed: int = 42,
    route_quota: dict[str, int] | None = None,
) -> list[EvalCaseDraft]:
    """候補群から件数制限とroute配分を適用して抽出する。"""
    if max_cases <= 0 or not drafts:
        return []

    rng = random.Random(seed)

    if route_quota:
        grouped: dict[str, list[EvalCaseDraft]] = defaultdict(list)
        for draft in drafts:
            grouped[draft.predicted_route].append(draft)
        for items in grouped.values():
            rng.shuffle(items)

        selected: list[EvalCaseDraft] = []
        for route in ROUTES:
            quota = route_quota.get(route, 0)
            if quota <= 0:
                continue
            selected.extend(grouped.get(route, [])[:quota])

        if len(selected) < max_cases:
            selected_keys = {
                hashlib.sha1(
                    f"{item.session_id}|{item.user_message_index}|{item.assistant_message_index}".encode("utf-8")
                ).hexdigest()
                for item in selected
            }
            remainder = []
            for draft in drafts:
                key = hashlib.sha1(
                    f"{draft.session_id}|{draft.user_message_index}|{draft.assistant_message_index}".encode("utf-8")
                ).hexdigest()
                if key in selected_keys:
                    continue
                remainder.append(draft)
            rng.shuffle(remainder)
            selected.extend(remainder[: max(0, max_cases - len(selected))])

        return selected[:max_cases]

    items = list(drafts)
    rng.shuffle(items)
    return items[:max_cases]


def parse_route_quota(raw: str | None) -> dict[str, int] | None:
    """`CLARIFY=50,DEEPEN=20` 形式を辞書へ変換する。"""
    if not raw:
        return None
    quota: dict[str, int] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid quota token: {token}")
        route, count_text = token.split("=", 1)
        route = route.strip().upper()
        if route not in ROUTES:
            raise ValueError(f"Unknown route in quota: {route}")
        if not count_text.strip().isdigit():
            raise ValueError(f"Invalid quota count: {token}")
        quota[route] = int(count_text.strip())
    return quota or None


def write_jsonl(items: Iterable[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def list_jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(path.glob("*.jsonl"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _normalize_text(text: str) -> str:
    compact = _WS_RE.sub(" ", text or "").strip().lower()
    return compact.strip("。．!！?？")


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _compact_rag_payload(rag: Any) -> dict[str, Any] | None:
    if not isinstance(rag, dict):
        return None

    top_score = None
    retrieved = rag.get("retrieved")
    if isinstance(retrieved, list) and retrieved:
        first = retrieved[0]
        if isinstance(first, dict):
            raw_score = first.get("score")
            if isinstance(raw_score, (int, float)):
                top_score = round(float(raw_score), 4)

    novelty = rag.get("novelty")
    novelty_info = None
    if isinstance(novelty, dict):
        novelty_info = {
            "is_novel": novelty.get("is_novel"),
            "confidence": novelty.get("confidence"),
            "reason": novelty.get("reason"),
        }

    return {
        "enabled": bool(rag.get("enabled")),
        "trigger": _as_optional_str(rag.get("trigger")),
        "skipped_reason": _as_optional_str(rag.get("skipped_reason")),
        "retrieval_method": _as_optional_str(rag.get("retrieval_method")),
        "top_score": top_score,
        "novelty": novelty_info,
        "saved_pending": bool(rag.get("saved_pending")),
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
