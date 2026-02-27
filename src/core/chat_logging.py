"""CLI/Streamlitの会話イベントをJSONLへ記録するロギングユーティリティ。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import uuid


def _utc_now_iso() -> str:
    """現在UTC時刻をミリ秒精度のISO8601文字列で返す。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sanitize_filename(text: str) -> str:
    """ログファイル名に使える安全なスラッグへ正規化する。"""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()
    return slug or "chat"


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
    return str(value)


@dataclass(slots=True)
class ChatSessionLogger:
    """1つの会話セッションに紐づくイベントログを追記するロガー。"""

    app_name: str
    session_id: str
    log_path: Path

    @classmethod
    def create(cls, app_name: str, logs_root: Path | None = None) -> "ChatSessionLogger":
        """新規セッション用ロガーを生成し、開始イベントを記録する。"""
        root = logs_root or (_project_root() / "logs" / "chat_sessions")
        root.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{_sanitize_filename(app_name)}-{uuid.uuid4().hex[:12]}"
        filename = f"{created_at}_{_sanitize_filename(app_name)}_{session_id}.jsonl"
        logger = cls(app_name=app_name, session_id=session_id, log_path=root / filename)
        logger.log_event("session_started")
        return logger

    @classmethod
    def from_state(cls, state: dict[str, str]) -> "ChatSessionLogger":
        """永続化済みstate辞書からロガーを復元する。"""
        return cls(
            app_name=state["app_name"],
            session_id=state["session_id"],
            log_path=Path(state["log_path"]),
        )

    def to_state(self) -> dict[str, str]:
        """セッション状態を`st.session_state`保存向け辞書へ変換する。"""
        return {
            "app_name": self.app_name,
            "session_id": self.session_id,
            "log_path": str(self.log_path),
        }

    def log_event(self, event_type: str, **payload: Any) -> None:
        """任意イベントをJSONLとして1行追記する。"""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": _utc_now_iso(),
            "session_id": self.session_id,
            "app_name": self.app_name,
            "event_type": event_type,
            "payload": _to_jsonable(payload),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_message(self, role: str, content: str, **payload: Any) -> None:
        """メッセージイベント（role/content）を記録するショートカット。"""
        self.log_event("message", role=role, content=content, **payload)

    def log_error(self, error: Exception | str, **payload: Any) -> None:
        """例外またはエラー文字列をエラーイベントとして記録する。"""
        self.log_event(
            "error",
            error_type=type(error).__name__ if isinstance(error, Exception) else "Error",
            error_message=str(error),
            **payload,
        )
