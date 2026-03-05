"""GateClassifierの詳細トレースをJSONLへ保存するユーティリティ。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping


TRACE_ENABLE_ENV = "GATE_TRACE_LOG_ENABLED"


def _utc_now_iso() -> str:
    """現在UTC時刻をミリ秒精度のISO8601文字列で返す。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _project_root() -> Path:
    """プロジェクトルートディレクトリを返す。"""
    return Path(__file__).resolve().parents[2]


def _to_jsonable(value: Any) -> Any:
    """JSONシリアライズ可能な型へ再帰的に変換する。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value))
        except Exception:
            return str(value)
    return str(value)


def _is_trace_enabled() -> bool:
    """環境変数に応じてGateトレースログ有効/無効を判定する。"""
    raw = os.getenv(TRACE_ENABLE_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _default_trace_log_path(logs_root: Path | None = None) -> Path:
    """Gateトレースログの保存先パスを返す。"""
    root = logs_root or (_project_root() / "logs" / "gate_agent_traces")
    date_prefix = datetime.now(timezone.utc).strftime("%Y%m%d")
    return root / f"{date_prefix}_gate_classifier_trace.jsonl"


def log_gate_agent_trace(
    payload: Mapping[str, Any],
    logs_root: Path | None = None,
) -> None:
    """GateClassifierの1実行分トレースをJSONLへ追記する。"""
    if not _is_trace_enabled():
        return

    log_path = _default_trace_log_path(logs_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": _utc_now_iso(),
        "event_type": "gate_classifier",
        "payload": _to_jsonable(dict(payload)),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
