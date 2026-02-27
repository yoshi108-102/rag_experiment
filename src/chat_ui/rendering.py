"""Streamlit画面での会話履歴・RAG情報・デバッグ情報を描画する関数群。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.core.token_usage import (
    context_limit_source_label,
    default_gate_model_name,
    estimate_messages_tokens,
    resolve_context_window_limit,
    usage_ratio,
)


def render_rag_panel(rag_info: dict[str, Any] | None) -> None:
    """メインチャット内にRAG結果の要約パネルを表示する。"""
    if not rag_info or not rag_info.get("enabled"):
        return

    with st.expander("過去の近いアイデア・疑問 (RAG)", expanded=False):
        novelty = rag_info.get("novelty") or {}
        if novelty:
            if novelty.get("is_novel"):
                st.write("新規性判定: 新しい内容の可能性あり")
                st.caption(f"理由: {novelty.get('reason', '-')}")
                if rag_info.get("saved_pending"):
                    st.caption("pending保存しました（要レビュー）")
            else:
                st.write("新規性判定: 既存の内容に近い可能性あり")
                st.caption(f"理由: {novelty.get('reason', '-')}")

        retrieved = rag_info.get("retrieved", [])
        if not retrieved:
            st.caption("近い記録は見つかりませんでした。")
            return

        for index, item in enumerate(retrieved, start=1):
            record = item.get("record", {})
            score = item.get("score", 0.0)
            st.markdown(
                f"**{index}. {record.get('topic', 'Unknown')}** "
                f"({record.get('record_type', '-')}, score={score:.2f})"
            )
            st.write(record.get("text", ""))
            tags = record.get("tags") or []
            if tags:
                st.caption("tags: " + ", ".join(tags))
            if record.get("applicable_when"):
                st.caption(f"適用条件: {record['applicable_when']}")


def render_reasoning_panel(reasoning: str | None) -> None:
    """Reasoning文字列がある場合に折りたたみパネルで表示する。"""
    if not reasoning:
        return

    with st.expander("AIの思考プロセス (Reasoning)", expanded=False):
        st.markdown(reasoning)


def render_route_debug_panel(debug_info: dict[str, Any] | None) -> None:
    """route判定やRAGトリガー情報などのデバッグ項目を表示する。"""
    if not debug_info:
        return

    rag_info = debug_info.get("rag") or {}

    with st.expander("AI Routing Info (Debug)", expanded=False):
        st.write(f"**Route:** {debug_info.get('route')}")
        st.write(f"**Reason:** {debug_info.get('reason')}")
        if debug_info.get("clarify_json"):
            st.write("**CLARIFY JSON:**", debug_info["clarify_json"])

        if "trigger" in rag_info:
            st.write(f"**RAG Trigger:** {rag_info.get('trigger')}")

        if rag_info.get("enabled") and rag_info.get("novelty"):
            st.write(f"**RAG Novelty:** {rag_info['novelty']['is_novel']}")
        elif rag_info.get("skipped_reason"):
            st.write(f"**RAG Skipped:** {rag_info['skipped_reason']}")


def render_chat_history(messages: list[dict[str, Any]]) -> None:
    """保存済みメッセージ配列を順に描画し、各種補助パネルも併記する。"""
    for msg in messages:
        with st.chat_message(msg["role"]):
            debug_info = msg.get("debug_info")
            if debug_info and debug_info.get("reasoning"):
                render_reasoning_panel(debug_info["reasoning"])
            st.markdown(msg["content"])
            _render_message_images(msg.get("images") or [])
            if debug_info:
                render_rag_panel(debug_info.get("rag"))
                render_route_debug_panel(debug_info)


def render_rag_sidebar(
    messages: list[dict[str, Any]],
    llm_context: list[dict[str, Any]],
    system_prompt_text: str | None = None,
) -> None:
    """サイドバーに最新RAG詳細とコンテキスト使用量を表示する。"""
    latest_rag = None
    latest_debug = None
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        debug_info = msg.get("debug_info") or {}
        rag = debug_info.get("rag")
        if rag is not None:
            latest_rag = rag
            latest_debug = debug_info
            break

    with st.sidebar:
        _render_context_window_panel(
            llm_context,
            latest_debug,
            system_prompt_text=system_prompt_text,
        )
        st.divider()
        st.subheader("RAG Debug")

        if latest_rag is None:
            st.info("まだRAG実行結果はありません。")
            return

        retrieval_method = latest_rag.get("retrieval_method") or "unknown"
        st.caption(f"retrieval: {retrieval_method}")
        if latest_rag.get("retrieval_note"):
            st.caption(f"note: {latest_rag['retrieval_note']}")

        st.write(f"Route: `{(latest_debug or {}).get('route', '-')}`")
        st.write(f"Trigger: `{latest_rag.get('trigger')}`")
        if latest_rag.get("query"):
            st.text_area("RAG Query", latest_rag["query"], height=120, disabled=True)

        novelty = latest_rag.get("novelty") or {}
        if novelty:
            top_score = None
            retrieved = latest_rag.get("retrieved") or []
            if retrieved:
                top_score = retrieved[0].get("score")
            st.markdown("**Novelty**")
            st.write(
                {
                    "is_novel": novelty.get("is_novel"),
                    "confidence": novelty.get("confidence"),
                    "reason": novelty.get("reason"),
                    "top_score": top_score,
                    "threshold": 0.38,
                }
            )
        elif latest_rag.get("skipped_reason"):
            st.write(f"Skipped: `{latest_rag.get('skipped_reason')}`")

        retrieved = latest_rag.get("retrieved") or []
        if not retrieved:
            st.caption("検索ヒットなし")
            return

        st.markdown("**Retrieved (Top-K)**")
        for idx, item in enumerate(retrieved, start=1):
            record = item.get("record") or {}
            score = item.get("score", 0.0)
            reasons = item.get("reasons") or []
            with st.expander(
                f"{idx}. {record.get('topic', 'Unknown')} ({score:.2f})",
                expanded=(idx == 1),
            ):
                st.caption(f"type={record.get('record_type', '-')}")
                if reasons:
                    st.caption("score reasons: " + ", ".join(reasons))
                if record.get("tags"):
                    st.caption("tags: " + ", ".join(record["tags"]))
                st.write(record.get("text", ""))
                if record.get("applicable_when"):
                    st.caption(f"適用条件: {record['applicable_when']}")


def _render_context_window_panel(
    llm_context: list[dict[str, Any]],
    latest_debug: dict[str, Any] | None,
    system_prompt_text: str | None = None,
) -> None:
    """推定token使用量と実測token usageを表示する。"""
    st.subheader("Context Window")

    gate_model = default_gate_model_name()
    limit_info = resolve_context_window_limit(gate_model)
    st.caption(f"model: `{limit_info.model}`")
    st.caption(f"上限の根拠: {context_limit_source_label(limit_info.source)}")

    max_context_tokens = int(
        st.number_input(
            "想定上限 (tokens)",
            min_value=4_096,
            max_value=1_000_000,
            value=min(1_000_000, max(4_096, limit_info.max_tokens)),
            step=1_024,
            key="context_window_limit_tokens",
            help=(
                "モデル既定値または環境変数から初期化されています。"
                "必要なら手動で調整できます。"
            ),
        )
    )

    estimation_messages = list(llm_context)
    if system_prompt_text:
        estimation_messages = [
            {"role": "system", "content": system_prompt_text},
            *estimation_messages,
        ]

    estimated_tokens = estimate_messages_tokens(estimation_messages)
    ratio = usage_ratio(estimated_tokens, max_context_tokens)
    st.progress(
        ratio,
        text=(
            f"推定: {estimated_tokens:,} / {max_context_tokens:,} "
            f"tokens ({ratio * 100:.1f}%)"
        ),
    )
    st.caption("推定値は system prompt を含む文字数ベースの近似です。")

    token_usage = (latest_debug or {}).get("token_usage") or {}
    input_tokens = token_usage.get("input_tokens")
    output_tokens = token_usage.get("output_tokens")
    total_tokens = token_usage.get("total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        st.caption("実測 token usage は最初の応答生成後に表示されます。")
        return

    st.markdown("**Latest API Usage (実測)**")
    usage_payload: dict[str, int] = {}
    if input_tokens is not None:
        usage_payload["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage_payload["output_tokens"] = output_tokens
    if total_tokens is not None:
        usage_payload["total_tokens"] = total_tokens
    st.write(usage_payload)


def _render_message_images(images: list[dict[str, Any]]) -> None:
    """メッセージ添付画像を表示可能な形式に整形して描画する。"""
    valid_images: list[dict[str, Any]] = []
    for image in images:
        data = image.get("data")
        if not isinstance(data, (bytes, bytearray)):
            continue
        valid_images.append(
            {
                "data": bytes(data),
                "name": str(image.get("name", "image")),
            }
        )

    if not valid_images:
        return

    st.image(
        [image["data"] for image in valid_images],
        caption=[image["name"] for image in valid_images],
        use_container_width=True,
    )
