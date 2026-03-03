"""RAG機能のLangChain toolラッパ。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.rag.reflection_context import analyze_reflection_context


class ReflectionContextToolInput(BaseModel):
    """反省コンテキスト解析ツールの入力。"""

    user_input: str = Field(description="現在のユーザー入力または連結検索クエリ")
    route: str = Field(description="現在ターンのルート名")
    top_k: int = Field(default=3, ge=1, le=10, description="返す類似候補数")


def run_reflection_context_lookup(
    user_input: str,
    route: str,
    *,
    allowed_routes: tuple[str, ...] = ("DEEPEN", "CLARIFY"),
    top_k: int = 3,
):
    """アプリ内部向けの型付きRAG呼び出し。"""
    return analyze_reflection_context(
        user_input,
        route,
        allowed_routes=allowed_routes,
        top_k=top_k,
    )


@tool("analyze_reflection_context", args_schema=ReflectionContextToolInput)
def analyze_reflection_context_tool(
    user_input: str,
    route: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """入力に近い過去知識を検索し、新規性判定を返す。"""
    return analyze_reflection_context(
        user_input,
        route,
        top_k=top_k,
    ).to_dict()


def build_rag_tools() -> list:
    """LangChainのtools配列を返す。"""
    return [analyze_reflection_context_tool]

