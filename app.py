import streamlit as st
from dotenv import load_dotenv
from src.agents.gate import analyze_input
from src.routing.router import execute_route

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Reflective Gate Chat", page_icon="🧠", layout="centered")

st.title("Reflective Gate Chat")
st.markdown("自分の思いを深めるための、会話入口トリアージ型AIチャット")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial AI greeting
    st.session_state.messages.append({
        "role": "assistant",
        "content": "お疲れ様です！本日の作業はどうでしたか？何か気になったことや、迷った瞬間はありましたか？",
        "debug_info": None
    })

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("debug_info") and msg["debug_info"].get("reasoning"):
            with st.expander("AIの思考プロセス (Reasoning)", expanded=False):
                st.markdown(msg["debug_info"]["reasoning"])
        st.markdown(msg["content"])
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

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("思考のトリアージ中..."):
            try:
                # Step 1: Analyze input using the Gate Model
                decision, reasoning = analyze_input(prompt)
                
                # Step 2: Execute the routing logic Based on the decision
                response = execute_route(decision)
                
                # Render reasoning if available
                if reasoning:
                    with st.expander("AIの思考プロセス (Reasoning)", expanded=False):
                        st.markdown(reasoning)
                
                # Render response
                st.markdown(response)
                
                # Render Debug Info
                debug_info = {
                    "route": decision.route,
                    "reason": decision.reason,
                    "reasoning": reasoning
                }
                with st.expander("AI Routing Info (Debug)", expanded=False):
                    st.write(f"**Route:** {debug_info['route']}")
                    st.write(f"**Reason:** {debug_info['reason']}")
                    
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "debug_info": debug_info
                })
                
                if decision.route == "FINISH":
                    st.info("対話が終了しました。再開する場合はページをリロードしてください。")
                    
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
