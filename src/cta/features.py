"""Simple answer feature extraction for CTA transition rules."""

from __future__ import annotations

import re

from src.cta.models import AnswerFeatures


_WORD_RE = re.compile(r"[A-Za-z0-9ぁ-んァ-ヶ一-龥]{2,}")
_CLAUSE_SPLIT_RE = re.compile(r"[。！？!?]")

_STOPWORDS = {
    "それ",
    "これ",
    "あれ",
    "こと",
    "もの",
    "ため",
    "よう",
    "ところ",
    "です",
    "ます",
    "した",
    "して",
    "いる",
    "ある",
}

_POSITIVE_MARKERS = ("はい", "あります", "しました", "そうです", "できます", "ありました")
_NEGATIVE_MARKERS = ("いいえ", "ない", "ありません", "違う", "特にない", "わかりません")
_QUESTION_MARKERS = ("?", "？", "ですか", "でしょうか", "何", "なぜ")

_COGNITIVE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("information", ("情報", "確認", "ニュース", "連絡", "聞い", "調べ")),
    ("situation_awareness", ("状況", "気づ", "違和感", "見えて", "把握")),
    ("decision", ("判断", "決め", "選択", "優先", "基準")),
    ("action", ("行動", "対応", "実施", "避難", "移動", "対処")),
)


class AnswerFeatureExtractor:
    """Extract lightweight features from user text."""

    def extract(self, user_text: str) -> AnswerFeatures:
        text = (user_text or "").strip()
        if not text:
            return AnswerFeatures(
                keywords=[],
                has_positive=False,
                has_negative=False,
                has_question=False,
                clause_count=0,
                cognitive_action_label="other",
            )

        lower_text = text.lower()
        keywords = self._extract_keywords(text)
        clause_count = self._extract_clause_count(text)

        return AnswerFeatures(
            keywords=keywords,
            has_positive=any(marker in text for marker in _POSITIVE_MARKERS),
            has_negative=any(marker in text for marker in _NEGATIVE_MARKERS),
            has_question=any(marker in text for marker in _QUESTION_MARKERS) or "?" in lower_text,
            clause_count=clause_count,
            cognitive_action_label=self._infer_cognitive_label(text),
        )

    def _extract_keywords(self, text: str) -> list[str]:
        candidates: list[str] = []
        for token in _WORD_RE.findall(text):
            if token in _STOPWORDS:
                continue
            candidates.append(token)

        # Preserve order and deduplicate.
        seen: set[str] = set()
        ordered: list[str] = []
        for token in candidates:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered[:5]

    def _extract_clause_count(self, text: str) -> int:
        chunks = [chunk.strip() for chunk in _CLAUSE_SPLIT_RE.split(text) if chunk.strip()]
        if not chunks:
            return 1
        return len(chunks)

    def _infer_cognitive_label(self, text: str) -> str:
        for label, patterns in _COGNITIVE_PATTERNS:
            if any(pattern in text for pattern in patterns):
                return label
        return "other"

