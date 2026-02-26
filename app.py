import streamlit as st
from dotenv import load_dotenv
from src.agents.gate import analyze_input
from src.routing.router import execute_route
from src.rag import analyze_with_rag
import collections
import hashlib

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Reflective Gate Chat", page_icon="🧠", layout="centered")

st.title("Reflective Gate Chat")
st.markdown("自分の思いを深めるための、会話入口トリアージ型AIチャット")

RAG_STREAK_TRIGGER = 3
RAG_COOLDOWN_TURNS = 2
RAG_BUFFER_MAX_ITEMS = 6


def init_rag_session_state():
    if "idea_buffer" not in st.session_state:
        st.session_state.idea_buffer = []
    if "rag_meta" not in st.session_state:
        st.session_state.rag_meta = {
            "turn_count": 0,
            "last_rag_turn": -999,
            "last_rag_signature": None,
        }


def update_idea_buffer(user_input: str, route: str):
    st.session_state.rag_meta["turn_count"] += 1
    st.session_state.idea_buffer.append(
        {
            "user_input": user_input,
            "route": route,
        }
    )
    st.session_state.idea_buffer = st.session_state.idea_buffer[-RAG_BUFFER_MAX_ITEMS:]


def build_buffered_idea_query() -> str:
    items = st.session_state.idea_buffer[-RAG_BUFFER_MAX_ITEMS:]
    if not items:
        return ""

    lines = []
    for item in items:
        lines.append(f"[{item['route']}] {item['user_input']}")
    return "\n".join(lines)


def _recent_deepen_clarify_streak() -> int:
    streak = 0
    for item in reversed(st.session_state.idea_buffer):
        if item["route"] in {"DEEPEN", "CLARIFY"}:
            streak += 1
            continue
        break
    return streak


def should_run_rag(current_route: str) -> tuple[bool, str]:
    meta = st.session_state.rag_meta
    if current_route in {"PARK", "FINISH"}:
        return True, "boundary"

    turns_since_last_rag = meta["turn_count"] - meta["last_rag_turn"]
    if turns_since_last_rag <= RAG_COOLDOWN_TURNS:
        return False, "cooldown"

    if current_route in {"DEEPEN", "CLARIFY"} and _recent_deepen_clarify_streak() >= RAG_STREAK_TRIGGER:
        return True, "streak"

    return False, "not-triggered"


def finalize_rag_run(query: str, clear_buffer: bool = False):
    meta = st.session_state.rag_meta
    meta["last_rag_turn"] = meta["turn_count"]
    meta["last_rag_signature"] = hashlib.sha1(query.encode("utf-8")).hexdigest()
    if clear_buffer:
        st.session_state.idea_buffer = []


def clear_idea_buffer_if_boundary(route: str):
    if route in {"PARK", "FINISH"}:
        st.session_state.idea_buffer = []


def should_skip_same_query(query: str) -> bool:
    if not query.strip():
        return True
    if len(query.replace("\n", "").strip()) < 15:
        return True
    signature = hashlib.sha1(query.encode("utf-8")).hexdigest()
    return signature == st.session_state.rag_meta.get("last_rag_signature")


def render_rag_panel(rag_info: dict | None):
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial AI greeting
    st.session_state.messages.append({
        "role": "assistant",
        "content": "お疲れ様です！本日の作業はどうでしたか？何か気になったことや、迷った瞬間はありましたか？",
        "debug_info": None
    })

# Initialize LLM context window (deque) directly retaining only role and content
if "llm_context" not in st.session_state:
    st.session_state.llm_context = collections.deque(maxlen=10)
    # Also add the initial AI greeting to the context window
    st.session_state.llm_context.append({
        "role": "assistant",
        "content": "お疲れ様です！本日の作業はどうでしたか？何か気になったことや、迷った瞬間はありましたか？"
    })

init_rag_session_state()

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("debug_info") and msg["debug_info"].get("reasoning"):
            with st.expander("AIの思考プロセス (Reasoning)", expanded=False):
                st.markdown(msg["debug_info"]["reasoning"])
        st.markdown(msg["content"])
        if msg.get("debug_info"):
            render_rag_panel(msg["debug_info"].get("rag"))
        if msg.get("debug_info"):
            with st.expander("AI Routing Info (Debug)", expanded=False):
                st.write(f"**Route:** {msg['debug_info']['route']}")
                st.write(f"**Reason:** {msg['debug_info']['reason']}")

# React to user input
if prompt := st.chat_input("考えたことや悩みを入力してください..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "debug_info": None})
    st.session_state.llm_context.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("思考のトリアージ中..."):
            try:
                # Step 1: Analyze input using the Gate Model
                decision, reasoning = analyze_input(prompt, list(st.session_state.llm_context))
                
                # Step 2: Execute the routing logic Based on the decision
                response = execute_route(decision)
                update_idea_buffer(prompt, decision.route)

                rag_debug = {
                    "enabled": False,
                    "skipped_reason": "not-triggered",
                    "trigger": None,
                    "query": None,
                }
                should_rag, rag_trigger = should_run_rag(decision.route)
                if should_rag:
                    rag_query = build_buffered_idea_query()
                    if should_skip_same_query(rag_query):
                        rag_debug["skipped_reason"] = "same-or-too-short-query"
                        rag_debug["trigger"] = rag_trigger
                        rag_debug["query"] = rag_query
                        clear_idea_buffer_if_boundary(decision.route)
                    else:
                        rag_analysis = analyze_with_rag(
                            rag_query,
                            decision.route,
                            allowed_routes=("DEEPEN", "CLARIFY", "PARK", "FINISH"),
                        )
                        rag_debug = rag_analysis.to_dict()
                        rag_debug["trigger"] = rag_trigger
                        rag_debug["query"] = rag_query
                        finalize_rag_run(
                            rag_query,
                            clear_buffer=decision.route in {"PARK", "FINISH"},
                        )
                else:
                    rag_debug["skipped_reason"] = rag_trigger
                    clear_idea_buffer_if_boundary(decision.route)
                
                # Render reasoning if available
                if reasoning:
                    with st.expander("AIの思考プロセス (Reasoning)", expanded=False):
                        st.markdown(reasoning)
                
                # Render response
                st.markdown(response)
                render_rag_panel(rag_debug)
                
                # Render Debug Info
                debug_info = {
                    "route": decision.route,
                    "reason": decision.reason,
                    "reasoning": reasoning,
                    "rag": rag_debug,
                }
                with st.expander("AI Routing Info (Debug)", expanded=False):
                    st.write(f"**Route:** {debug_info['route']}")
                    st.write(f"**Reason:** {debug_info['reason']}")
                    st.write(f"**RAG Trigger:** {rag_debug.get('trigger')}")
                    if rag_debug.get("enabled") and rag_debug.get("novelty"):
                        st.write(f"**RAG Novelty:** {rag_debug['novelty']['is_novel']}")
                    elif rag_debug.get("skipped_reason"):
                        st.write(f"**RAG Skipped:** {rag_debug['skipped_reason']}")
                    
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "debug_info": debug_info
                })
                st.session_state.llm_context.append({
                    "role": "assistant",
                    "content": response
                })
                
                if decision.route == "FINISH":
                    st.info("対話が終了しました。再開する場合はページをリロードしてください。")
                    
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
