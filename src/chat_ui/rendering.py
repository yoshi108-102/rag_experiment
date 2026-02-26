from __future__ import annotations

from typing import Any

import streamlit as st


def render_rag_panel(rag_info: dict[str, Any] | None) -> None:
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
    if not reasoning:
        return

    with st.expander("AIの思考プロセス (Reasoning)", expanded=False):
        st.markdown(reasoning)


def render_route_debug_panel(debug_info: dict[str, Any] | None) -> None:
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
    for msg in messages:
        with st.chat_message(msg["role"]):
            debug_info = msg.get("debug_info")
            if debug_info and debug_info.get("reasoning"):
                render_reasoning_panel(debug_info["reasoning"])
            st.markdown(msg["content"])
            if debug_info:
                render_rag_panel(debug_info.get("rag"))
                render_route_debug_panel(debug_info)


def render_rag_sidebar(messages: list[dict[str, Any]]) -> None:
    with st.sidebar:
        st.subheader("RAG Debug")

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
