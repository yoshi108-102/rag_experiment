import streamlit as st
from dotenv import load_dotenv

from src.chat_ui import (
    append_assistant_message,
    append_user_message,
    handle_user_turn,
    initialize_session_state,
    render_chat_history,
    render_rag_panel,
    render_reasoning_panel,
    render_route_debug_panel,
)

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Reflective Gate Chat", page_icon="🧠", layout="centered")

st.title("Reflective Gate Chat")
st.markdown("自分の思いを深めるための、会話入口トリアージ型AIチャット")

initialize_session_state(st.session_state)
render_chat_history(st.session_state.messages)

# React to user input
if prompt := st.chat_input("考えたことや悩みを入力してください..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    append_user_message(st.session_state, prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("思考のトリアージ中..."):
            try:
                turn_result = handle_user_turn(prompt, st.session_state)

                render_reasoning_panel(turn_result.reasoning)
                st.markdown(turn_result.response)
                render_rag_panel(turn_result.debug_info.get("rag"))
                render_route_debug_panel(turn_result.debug_info)

                append_assistant_message(
                    st.session_state,
                    turn_result.response,
                    turn_result.debug_info,
                )

                if turn_result.is_finished:
                    st.info("対話が終了しました。再開する場合はページをリロードしてください。")
                    
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
