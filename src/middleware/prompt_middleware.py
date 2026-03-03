"""Gate判定用のプロンプト構築とLangChain middleware定義。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage


IMAGE_FALLBACK_TEXT = "添付画像を見て会話を続けたいです。"
OVERALL_CONTEXT_MODE_ENV = "OVERALL_CONTEXT_MODE"
OVERALL_CONTEXT_AUTO = "auto"
OVERALL_CONTEXT_ALWAYS = "always"
OVERALL_CONTEXT_OFF = "off"
DOMAIN_HINT_KEYWORDS = (
    "銅合金",
    "パイプ",
    "矯正",
    "曲がり",
    "プレス",
    "油圧",
    "棒材",
    "受け皿",
    "馬台",
    "直線度",
)

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


def load_gate_prompt(include_overall: bool = True) -> str:
    """Gate判定に使用するsystem prompt本文を読み込む。"""
    base_dir = Path(__file__).resolve().parent.parent.parent
    prompt_path = base_dir / "prompts" / "gate_prompt.md"
    overall_path = base_dir / "prompts" / "overall.md"

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    if include_overall and overall_path.exists():
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
    include_overall = _should_include_overall_context(request)
    return load_gate_prompt(include_overall=include_overall)


def _should_include_overall_context(request: ModelRequest) -> bool:
    mode = (os.getenv(OVERALL_CONTEXT_MODE_ENV, OVERALL_CONTEXT_AUTO).strip().lower()
            or OVERALL_CONTEXT_AUTO)
    if mode == OVERALL_CONTEXT_ALWAYS:
        return True
    if mode == OVERALL_CONTEXT_OFF:
        return False

    latest_user_text, has_image = _extract_latest_user_signal(request.messages)
    if has_image:
        return True

    normalized = latest_user_text.lower()
    return any(keyword.lower() in normalized for keyword in DOMAIN_HINT_KEYWORDS)


def _extract_latest_user_signal(messages: Sequence[Any]) -> tuple[str, bool]:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        return _coerce_human_content_to_signal(message.content)
    return "", False


def _coerce_human_content_to_signal(content: Any) -> tuple[str, bool]:
    if isinstance(content, str):
        return content, False
    if not isinstance(content, list):
        return str(content or ""), False

    texts: list[str] = []
    has_image = False
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block_type = str(block.get("type", ""))
        if block_type == "text":
            texts.append(str(block.get("text", "")))
        elif block_type == "image_url":
            has_image = True
    return "\n".join(texts), has_image


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
