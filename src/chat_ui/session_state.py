"""Streamlitの`session_state`に会話履歴とRAG補助状態を保持する処理群。"""

from __future__ import annotations

import collections
from typing import Any

from src.chat_ui.constants import INITIAL_ASSISTANT_MESSAGE


def initialize_session_state(session_state: Any) -> None:
    """チャット開始時に必要なstateキーを初期化する。"""
    if "messages" not in session_state:
        session_state.messages = [
            {
                "role": "assistant",
                "content": INITIAL_ASSISTANT_MESSAGE,
                "debug_info": None,
            }
        ]

    if "llm_context" not in session_state:
        session_state.llm_context = collections.deque(maxlen=10)
        session_state.llm_context.append(
            {
                "role": "assistant",
                "content": INITIAL_ASSISTANT_MESSAGE,
            }
        )

    if "idea_buffer" not in session_state:
        session_state.idea_buffer = []

    if "rag_meta" not in session_state:
        session_state.rag_meta = {
            "turn_count": 0,
            "last_rag_turn": -999,
            "last_rag_signature": None,
        }


def append_user_message(
    session_state: Any,
    prompt: str,
    images: list[dict[str, Any]] | None = None,
    llm_content: str | None = None,
) -> None:
    """ユーザーメッセージを表示履歴とLLMコンテキストへ追記する。"""
    message: dict[str, Any] = {"role": "user", "content": prompt, "debug_info": None}
    if images:
        message["images"] = _sanitize_images_for_state(images)

    session_state.messages.append(message)
    session_state.llm_context.append(
        {"role": "user", "content": llm_content if llm_content is not None else prompt}
    )


def append_assistant_message(
    session_state: Any,
    content: str,
    debug_info: dict[str, Any],
) -> None:
    """アシスタント応答とデバッグ情報を履歴へ追記する。"""
    session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "debug_info": debug_info,
        }
    )
    session_state.llm_context.append({"role": "assistant", "content": content})


def _sanitize_images_for_state(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """永続可能な画像メタ情報のみを抽出してstate保存用に整形する。"""
    sanitized_images: list[dict[str, Any]] = []
    for image in images:
        data = image.get("data")
        if not isinstance(data, (bytes, bytearray)):
            continue
        sanitized_images.append(
            {
                "name": str(image.get("name", "image")),
                "mime_type": str(image.get("mime_type", "")),
                "data": bytes(data),
            }
        )
    return sanitized_images
