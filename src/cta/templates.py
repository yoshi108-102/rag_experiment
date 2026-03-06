"""Phrase dictionaries and template rendering for CTA Sprint 1."""

from __future__ import annotations

import random
from typing import Mapping


QUESTION_TEMPLATES: dict[str, list[str]] = {
    "STD1": [
        "こんにちは。まずは {subject} について、印象に残った出来事を教えてください。",
    ],
    "STD3": [
        "次は {subject} を伺います。{topic} について印象に残っていることはありますか？",
    ],
    "STD4": [
        "ありがとうございます。具体的にどのような状況だったか、もう少し詳しく教えてください。",
        "もう少し具体例を交えて説明していただけますか？",
    ],
    "STD5": [
        "{keyword} という点ですね。そこをもう少し詳しく教えてください。",
    ],
    "STD7": [
        "話題を少し変えて、{topic} について教えてください。",
    ],
    "STD11": [
        "ありがとうございます。以上でインタビューは終了です。",
    ],
    "CDM1": [
        "そのとき、{probe}",
    ],
    "CDM2": [
        "その場面で、{probe}",
    ],
    "CDM3": [
        "{keyword} について補足すると、{probe}",
    ],
}

BACKCHANNEL_TEMPLATES: dict[str, list[str]] = {
    "BC1": ["なるほど、ありがとうございます。"],
    "BC3": ["そうですか。"],
    "BC4": ["なるほど。"],
    "BC5": ["わかりました。"],
    "BC7": ["似た話を別の方から聞いたことがあります。"],
    "BC8": ["その視点はとても重要ですね。"],
}

CDM_PROBES: dict[str, list[str]] = {
    "information": [
        "どの情報を最初に確認しましたか？",
        "判断材料として最も重要だった情報は何でしたか？",
    ],
    "situation_awareness": [
        "どのサインや違和感に最初に気づきましたか？",
        "状況把握で見落としやすい点はどこでしたか？",
    ],
    "decision": [
        "その判断を選んだ決め手は何でしたか？",
        "代替案と比較して何を優先しましたか？",
    ],
    "action": [
        "最初に取った行動と、その理由を教えてください。",
        "その対応を実行するうえで注意したことは何でしたか？",
    ],
    "other": [
        "その場面で特に重要だったポイントは何でしたか？",
        "振り返って、次に活かせる点は何だと思いますか？",
    ],
}


class TemplateRepository:
    """Dictionary-backed renderer with deterministic randomness."""

    def __init__(self, seed: int | None = 7) -> None:
        self._seed = 7 if seed is None else int(seed)
        self._rng = random.Random(self._seed)

    def render_question(
        self,
        question_type: str,
        subject: str,
        topic: str,
        keyword: str | None,
        probe: str | None,
    ) -> str:
        templates = QUESTION_TEMPLATES.get(question_type)
        if not templates:
            return "もう少し詳しく教えてください。"
        template = self._rng.choice(templates)
        values = {
            "subject": subject,
            "topic": topic,
            "keyword": keyword or topic or "その点",
            "probe": probe or "もう少し詳しく教えてください。",
        }
        return template.format(**values)

    def select_backchannel(self, turn_count: int, question_type: str) -> str | None:
        if question_type in {"STD1", "STD3", "STD7", "STD11"}:
            return None
        if turn_count <= 2:
            return self._rng.choice(["BC7", "BC8"])
        if turn_count <= 5:
            return self._rng.choice(["BC5", "BC4"])
        return self._rng.choice(["BC3", "BC1"])

    def render_backchannel(self, backchannel_type: str | None) -> str | None:
        if backchannel_type is None:
            return None
        candidates = BACKCHANNEL_TEMPLATES.get(backchannel_type)
        if not candidates:
            return None
        return self._rng.choice(candidates)

    def get_cdm_probe(self, cognitive_action_label: str) -> str:
        probes = CDM_PROBES.get(cognitive_action_label) or CDM_PROBES["other"]
        return self._rng.choice(probes)

    def compose_response(self, question_text: str, backchannel_text: str | None) -> str:
        if not backchannel_text:
            return question_text.strip()
        return f"{backchannel_text.strip()} {question_text.strip()}".strip()

    @property
    def template_catalog(self) -> Mapping[str, list[str]]:
        return QUESTION_TEMPLATES

    @property
    def seed(self) -> int:
        return self._seed
