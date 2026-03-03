"""Gate判定用のプロンプト構築とLangChain middleware定義。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage


IMAGE_FALLBACK_TEXT = "添付画像を見て会話を続けたいです。"

class TextBlock(TypedDict):
    """Responses API text block."""

    type: Literal["text"]
    text: str


class ImageURLPayload(TypedDict):
    """Image URL payload for Responses API blocks."""

    url: str


class ImageURLBlock(TypedDict):
    """Responses API image_url block."""

    type: Literal["image_url"]
    image_url: ImageURLPayload


HumanContentBlock = TextBlock | ImageURLBlock


def as_human_message_content(
    content: str | list[HumanContentBlock],
) -> str | list[str | dict]:
    """`HumanMessage` コンストラクタが要求するcontent型へ変換する。"""
    return cast("str | list[str | dict]", content)


def load_gate_prompt() -> str:
    """Gate判定に使用するsystem prompt本文を読み込む。"""
    base_dir = Path(__file__).resolve().parent.parent.parent
    prompt_path = base_dir / "prompts" / "gate_prompt.md"
    overall_path = base_dir / "prompts" / "overall.md"

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    if overall_path.exists():
        with open(overall_path, "r", encoding="utf-8") as f:
            overall_context = f.read()
        prompt += (
            "\n\n[ドメイン知識（前提）]\n"
            "以下の作業概要を前提知識として踏まえた上で、"
            "ユーザーの発言を解釈してください。\n"
            f"{overall_context}"
        )

    return prompt


@dynamic_prompt
def gate_system_prompt_middleware(request: ModelRequest) -> str:
    """LangChain middlewareとしてsystem promptを動的注入する。"""
    del request
    return load_gate_prompt()


def build_human_message_content(
    text: str,
    images: Sequence[Mapping[str, str]] | None = None,
) -> str | list[HumanContentBlock]:
    """テキストと画像群をResponses API向けのHumanMessage形式へ整形する。"""
    normalized_text = (text or "").strip()
    if not images:
        return normalized_text

    text_block: TextBlock = {
        "type": "text",
        "text": normalized_text or IMAGE_FALLBACK_TEXT,
    }
    content_blocks: list[HumanContentBlock] = [text_block]
    for image in images:
        mime_type = image.get("mime_type", "").strip().lower()
        data_base64 = image.get("data_base64", "").strip()
        if not mime_type.startswith("image/"):
            continue
        if not data_base64:
            continue
        image_block: ImageURLBlock = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{data_base64}"},
        }
        content_blocks.append(image_block)

    if len(content_blocks) == 1:
        return text_block["text"]
    return content_blocks


def build_chat_messages(
    user_input: str,
    chat_context: Sequence[Mapping[str, Any]] | None = None,
    user_images: Sequence[Mapping[str, str]] | None = None,
) -> list[HumanMessage | AIMessage]:
    """system/context/current入力からLLM送信用メッセージ列を構築する。"""
    latest_context_message = chat_context[-1] if chat_context else None
    latest_context_is_same_user_input = bool(
        latest_context_message
        and latest_context_message.get("role") == "user"
        and latest_context_message.get("content") == user_input
    )
    latest_context_index = (len(chat_context) - 1) if chat_context else -1

    messages: list[HumanMessage | AIMessage] = []
    if chat_context:
        for idx, msg in enumerate(chat_context):
            role = str(msg.get("role", ""))
            if role == "user":
                msg_images: Sequence[Mapping[str, str]] | None = None
                if (
                    idx == latest_context_index
                    and latest_context_is_same_user_input
                    and user_images
                ):
                    msg_images = user_images

                messages.append(
                    HumanMessage(
                        content=as_human_message_content(
                            build_human_message_content(
                                str(msg.get("content", "")),
                                msg_images,
                            )
                        )
                    )
                )
            elif role == "assistant":
                messages.append(AIMessage(content=str(msg.get("content", ""))))

    if not latest_context_is_same_user_input:
        messages.append(
            HumanMessage(
                content=as_human_message_content(
                    build_human_message_content(user_input, user_images)
                )
            )
        )
    return messages
