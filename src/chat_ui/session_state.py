from __future__ import annotations

import collections
from typing import Any

from src.chat_ui.constants import INITIAL_ASSISTANT_MESSAGE


def initialize_session_state(session_state: Any) -> None:
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


def append_user_message(session_state: Any, prompt: str) -> None:
    session_state.messages.append(
        {"role": "user", "content": prompt, "debug_info": None}
    )
    session_state.llm_context.append({"role": "user", "content": prompt})


def append_assistant_message(
    session_state: Any,
    content: str,
    debug_info: dict[str, Any],
) -> None:
    session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "debug_info": debug_info,
        }
    )
    session_state.llm_context.append({"role": "assistant", "content": content})
