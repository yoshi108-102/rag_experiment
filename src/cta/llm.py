"""LangChain + OpenAI response naturalization for CTA."""

from __future__ import annotations

import os
from typing import Any, Callable

from langchain_openai import ChatOpenAI

from src.cta.models import CTASessionState


class LLMNaturalizer:
    """Naturalize template text while preserving selected question intent."""

    def __init__(
        self,
        model_name: str | None = None,
        llm_factory: Callable[[], Any] | None = None,
        timeout_seconds: float = 20.0,
        prompt_version: str = "cta_sprint1_v1",
    ) -> None:
        self.model_name = model_name or os.getenv("CTA_LLM_MODEL", "gpt-4o-mini")
        self.prompt_version = prompt_version
        self._llm_factory = llm_factory or (
            lambda: ChatOpenAI(
                model=self.model_name,
                temperature=0.2,
                timeout=timeout_seconds,
            )
        )

    def naturalize(
        self,
        template_response: str,
        session: CTASessionState,
        question_type: str,
    ) -> str:
        llm = self._llm_factory()
        prompt = (
            "あなたはCTAインタビューの文面を調整するアシスタントです。"
            "意図や質問タイプを変えず、自然な日本語に短く整えてください。"
            "禁止: 話題変更、質問追加、断定的助言。\n\n"
            f"session_id: {session.session_id}\n"
            f"subject: {session.current_subject_name}\n"
            f"topic: {session.current_topic_name}\n"
            f"question_type: {question_type}\n"
            f"入力文面: {template_response}\n"
            "出力は調整後テキストのみ。"
        )
        result = llm.invoke(prompt)
        text = self._extract_text(result)
        if not text:
            raise ValueError("LLM returned empty text")
        return text

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()

        content = getattr(result, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    candidate = item.get("text")
                    if isinstance(candidate, str) and candidate.strip():
                        chunks.append(candidate.strip())
                elif isinstance(item, str) and item.strip():
                    chunks.append(item.strip())
            return " ".join(chunks).strip()
        return ""

