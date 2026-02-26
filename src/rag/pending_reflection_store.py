from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.rag.models import NoveltyDecision, RetrievalResult


def default_pending_reflections_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "datasets"
        / "pending"
        / "pending_reflections.jsonl"
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_existing_signatures(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()

    signatures: set[tuple[str, str]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            route = str(item.get("route", "")).strip()
            user_input = str(item.get("user_input", "")).strip()
            if route and user_input:
                signatures.add((route, user_input))
    return signatures


def store_pending_reflection(
    user_input: str,
    route: str,
    novelty: NoveltyDecision,
    retrieved: list[RetrievalResult],
    path: str | Path | None = None,
) -> bool:
    target = Path(path) if path else default_pending_reflections_path()
    _ensure_parent(target)

    signature = (route.strip(), user_input.strip())
    if not signature[0] or not signature[1]:
        return False

    if signature in _load_existing_signatures(target):
        return False

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "route": signature[0],
        "user_input": signature[1],
        "novelty": novelty.to_dict(),
        "top_matches": [item.to_dict() for item in retrieved[:3]],
        "status": "pending_review",
    }

    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return True


# Backward-compatible aliases while callers migrate.
default_pending_store_path = default_pending_reflections_path
save_pending_reflection = store_pending_reflection
