"""チャット投稿に含まれる画像ファイルの抽出・正規化・変換処理を担う。"""

from __future__ import annotations

import base64
import mimetypes
from typing import Any

IMAGE_FILE_TYPES = ["png", "jpg", "jpeg", "webp", "gif", "bmp"]
IMAGE_ONLY_DISPLAY_TEXT = "（画像を送信しました）"
IMAGE_ONLY_GATE_PROMPT = "画像を送信しました。画像の内容を踏まえて会話を続けてください。"


def extract_chat_submission(submission: Any) -> tuple[str, list[dict[str, Any]]]:
    """投稿オブジェクトからテキストと画像一覧を抽出する。"""
    if submission is None:
        return "", []

    if isinstance(submission, str):
        return submission.strip(), []

    text = str(getattr(submission, "text", "") or "").strip()
    files = getattr(submission, "files", None) or []
    images = []
    for uploaded_file in files:
        image_payload = _to_image_payload(uploaded_file)
        if image_payload is not None:
            images.append(image_payload)
    return text, images


def normalize_display_text(text: str, images: list[dict[str, Any]]) -> str:
    """画面表示用テキストを正規化し、画像のみ投稿時の代替文言を返す。"""
    normalized = (text or "").strip()
    if normalized:
        return normalized
    if images:
        return IMAGE_ONLY_DISPLAY_TEXT
    return ""


def normalize_gate_text(text: str, images: list[dict[str, Any]]) -> str:
    """Gate判定へ渡す入力テキストを正規化する。"""
    normalized = (text or "").strip()
    if normalized:
        return normalized
    if images:
        return IMAGE_ONLY_GATE_PROMPT
    return ""


def build_image_log_payload(images: list[dict[str, Any]]) -> dict[str, Any]:
    """ログ保存向けに画像メタ情報を集約する。"""
    return {
        "image_count": len(images),
        "image_names": [str(image.get("name", "")) for image in images],
        "image_mime_types": [str(image.get("mime_type", "")) for image in images],
        "image_sizes": [len(bytes(image.get("data", b""))) for image in images],
    }


def build_llm_image_payload(images: list[dict[str, Any]]) -> list[dict[str, str]]:
    """画像バイナリをLLM入力向けBase64ペイロードへ変換する。"""
    payloads: list[dict[str, str]] = []
    for image in images:
        data = image.get("data")
        mime_type = str(image.get("mime_type", "") or "").strip().lower()
        if not isinstance(data, (bytes, bytearray)):
            continue
        if not mime_type.startswith("image/"):
            continue
        payloads.append(
            {
                "name": str(image.get("name", "image")),
                "mime_type": mime_type,
                "data_base64": base64.b64encode(bytes(data)).decode("ascii"),
            }
        )
    return payloads


def _to_image_payload(uploaded_file: Any) -> dict[str, Any] | None:
    """アップロードファイルを内部画像形式へ変換し、非画像は除外する。"""
    if uploaded_file is None:
        return None

    name = str(getattr(uploaded_file, "name", "") or "image")
    mime_type = str(getattr(uploaded_file, "type", "") or "").strip().lower()
    if not mime_type:
        mime_type = (mimetypes.guess_type(name)[0] or "").lower()
    if not mime_type.startswith("image/"):
        return None

    data = uploaded_file.getvalue()
    if not data:
        return None

    return {
        "name": name,
        "mime_type": mime_type,
        "data": data,
    }
