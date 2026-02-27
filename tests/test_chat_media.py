from __future__ import annotations

from src.chat_ui.media import (
    build_llm_image_payload,
    extract_chat_submission,
    normalize_display_text,
    normalize_gate_text,
)


class DummyUploadedFile:
    def __init__(self, name: str, mime_type: str, data: bytes):
        self.name = name
        self.type = mime_type
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class DummySubmission:
    def __init__(self, text: str, files: list[DummyUploadedFile]):
        self.text = text
        self.files = files


def test_extract_chat_submission_filters_non_image_files():
    submission = DummySubmission(
        text="画像つきで相談したいです",
        files=[
            DummyUploadedFile("a.png", "image/png", b"\x89PNG"),
            DummyUploadedFile("note.txt", "text/plain", b"memo"),
        ],
    )

    text, images = extract_chat_submission(submission)

    assert text == "画像つきで相談したいです"
    assert len(images) == 1
    assert images[0]["name"] == "a.png"
    assert images[0]["mime_type"] == "image/png"
    assert images[0]["data"] == b"\x89PNG"


def test_normalize_texts_for_image_only_submission():
    images = [{"name": "photo.jpg", "mime_type": "image/jpeg", "data": b"abc"}]

    assert normalize_display_text("", images) == "（画像を送信しました）"
    assert "画像を送信しました" in normalize_gate_text("", images)


def test_build_llm_image_payload_base64_encoding():
    payload = build_llm_image_payload(
        [{"name": "graph.png", "mime_type": "image/png", "data": b"hello"}]
    )

    assert len(payload) == 1
    assert payload[0]["name"] == "graph.png"
    assert payload[0]["mime_type"] == "image/png"
    assert payload[0]["data_base64"] == "aGVsbG8="
