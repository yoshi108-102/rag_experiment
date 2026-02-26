import streamlit as st
from dotenv import load_dotenv

from src.chat_ui.constants import INITIAL_ASSISTANT_MESSAGE
from src.chat_ui.rendering import (
    render_chat_history,
    render_rag_panel,
    render_rag_sidebar,
    render_reasoning_panel,
    render_route_debug_panel,
)
from src.chat_ui.session_state import (
    append_assistant_message,
    append_user_message,
    initialize_session_state,
)
from src.chat_ui.turn_handler import handle_user_turn
from src.core.chat_logging import ChatSessionLogger


load_dotenv()

st.set_page_config(page_title="Reflective Gate Chat", page_icon="🧠", layout="centered")

st.title("Reflective Gate Chat")
st.markdown("自分の思いを深めるための、会話入口トリアージ型AIチャット")


def get_chat_logger() -> ChatSessionLogger:
    state_key = "chat_log_state"
    if state_key not in st.session_state:
        logger = ChatSessionLogger.create(app_name="streamlit")
        st.session_state[state_key] = logger.to_state()
        return logger
    return ChatSessionLogger.from_state(st.session_state[state_key])


chat_logger = get_chat_logger()

is_first_session_render = "messages" not in st.session_state
initialize_session_state(st.session_state)

if is_first_session_render:
    chat_logger.log_message(
        "assistant",
        INITIAL_ASSISTANT_MESSAGE,
        source="initial_greeting",
        message_index=0,
    )

render_rag_sidebar(st.session_state.messages)
render_chat_history(st.session_state.messages)

if prompt := st.chat_input("考えたことや悩みを入力してください..."):
    st.chat_message("user").markdown(prompt)
    append_user_message(st.session_state, prompt)
    chat_logger.log_message(
        "user",
        prompt,
        message_index=len(st.session_state.messages) - 1,
    )

    with st.chat_message("assistant"):
        with st.spinner("思考のトリアージ中..."):
            try:
                turn_result = handle_user_turn(prompt, st.session_state)
                debug_info = turn_result.debug_info
                rag_debug = debug_info.get("rag") or {
                    "enabled": False,
                    "skipped_reason": "no-rag-debug",
                    "trigger": None,
                    "query": None,
                }
                debug_info["rag"] = rag_debug

                render_reasoning_panel(turn_result.reasoning)
                st.markdown(turn_result.response)
                render_rag_panel(rag_debug)
                render_route_debug_panel(debug_info)

                append_assistant_message(
                    st.session_state,
                    turn_result.response,
                    debug_info,
                )
                chat_logger.log_message(
                    "assistant",
                    turn_result.response,
                    message_index=len(st.session_state.messages) - 1,
                    debug_info=debug_info,
                )

                if turn_result.is_finished:
                    chat_logger.log_event(
                        "conversation_finished",
                        route=debug_info["route"],
                        reason=debug_info["reason"],
                    )
                    st.info("いったん区切りです。付け加えることがあればそのまま続けて入力できます。")

            except Exception as e:
                chat_logger.log_error(e, stage="streamlit_response", user_input=prompt)
                st.error(f"エラーが発生しました: {e}")
