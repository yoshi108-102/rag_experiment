"""Route応答を、意図を変えずにやわらかく整える後段エージェント。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


RESPONSE_REFINER_ENABLED_ENV = "RESPONSE_REFINER_ENABLED"
RESPONSE_REFINER_MODEL_ENV = "RESPONSE_REFINER_MODEL"
DEFAULT_RESPONSE_REFINER_MODEL = "gpt-4o-mini"

QUESTION_ENDING_PATTERNS = (
    r"[？?]",
    r"(ですか|ますか|でしょうか|でしたか|ます？|です？)",
    r"(どこ|どの|いつ|何が|何を|どう|どんな|どの場面).*(ですか|でしたか|でしょうか)?$",
)


@dataclass(frozen=True)
class ResponseRefinementResult:
    """後段リファインの結果。"""

    text: str
    enabled: bool
    fallback_used: bool
    model_name: str | None
    error: str | None = None
    draft_response: str | None = None


def _is_response_refiner_enabled() -> bool:
    raw = os.getenv(RESPONSE_REFINER_ENABLED_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def load_response_refiner_prompt() -> str:
    """後段エージェントB用のsystem promptを読み込む。"""
    base_dir = Path(__file__).resolve().parents[2]
    prompt_path = base_dir / "prompts" / "response_refiner_prompt.md"
    return prompt_path.read_text(encoding="utf-8")


def _extract_text(result: Any) -> str:
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


def _normalize_sentence_units(text: str) -> list[str]:
    parts = re.split(r"[。.!！\n]+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _question_count(text: str) -> int:
    question_marks = text.count("?") + text.count("？")
    if question_marks > 0:
        return question_marks

    count = 0
    for part in _normalize_sentence_units(text):
        if any(re.search(pattern, part) for pattern in QUESTION_ENDING_PATTERNS):
            count += 1
    return count


def _is_valid_refined_response(draft_response: str, refined_text: str, route: str) -> bool:
    refined = refined_text.strip()
    if not refined:
        return False

    if len(_normalize_sentence_units(refined)) > 3:
        return False

    draft_question_count = _question_count(draft_response)
    refined_question_count = _question_count(refined)

    if refined_question_count > 1:
        return False
    if draft_question_count == 0 and refined_question_count > 0:
        return False
    if route in {"DEEPEN", "CLARIFY"} and draft_question_count >= 1 and refined_question_count == 0:
        return False

    return True


def _format_recent_context(chat_context: list | None, limit: int = 4) -> str:
    if not chat_context:
        return "(none)"

    lines: list[str] = []
    for msg in chat_context[-limit:]:
        role = str(msg.get("role", "unknown"))
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(none)"


def refine_route_response(
    draft_response: str,
    route: str,
    user_input: str,
    chat_context: list | None = None,
) -> ResponseRefinementResult:
    """Aが作った返答文を、routeを保ったままやわらかく整える。"""
    model_name = (
        os.getenv(RESPONSE_REFINER_MODEL_ENV, DEFAULT_RESPONSE_REFINER_MODEL).strip()
        or DEFAULT_RESPONSE_REFINER_MODEL
    )

    if not draft_response.strip():
        return ResponseRefinementResult(
            text=draft_response,
            enabled=False,
            fallback_used=False,
            model_name=model_name,
            draft_response=draft_response,
        )

    if not _is_response_refiner_enabled():
        return ResponseRefinementResult(
            text=draft_response,
            enabled=False,
            fallback_used=False,
            model_name=model_name,
            draft_response=draft_response,
        )

    llm = ChatOpenAI(model=model_name, temperature=0.2, timeout=15.0)
    system_prompt = load_response_refiner_prompt()
    human_prompt = (
        f"route: {route}\n"
        f"latest_user_input: {user_input}\n"
        f"draft_response: {draft_response}\n"
        f"draft_question_count: {_question_count(draft_response)}\n"
        f"recent_context:\n{_format_recent_context(chat_context)}\n\n"
        "出力は調整後テキストのみ。"
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        refined_text = _extract_text(response)
        if not _is_valid_refined_response(draft_response, refined_text, route):
            raise ValueError("Refined response violated guard conditions")
        return ResponseRefinementResult(
            text=refined_text,
            enabled=True,
            fallback_used=False,
            model_name=model_name,
            draft_response=draft_response,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[Response Refiner Error] {error}")
        return ResponseRefinementResult(
            text=draft_response,
            enabled=True,
            fallback_used=True,
            model_name=model_name,
            error=error,
            draft_response=draft_response,
        )
