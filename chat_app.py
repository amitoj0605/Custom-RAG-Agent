import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import graph as app

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------

st.title("🤖 Agentic RAG Assistant")
st.caption("Ask questions about Generative AI, RAG systems, and AI agents.")

# -----------------------------
# SESSION STATE
# -----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

# -----------------------------
# SIDEBAR (DEBUG PANEL)
# -----------------------------

with st.sidebar:

    st.header("⚙️ Debug Panel")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.debug_logs = []
        st.rerun()

    st.divider()

    st.subheader("Logs")

    for log in st.session_state.debug_logs[-20:]:
        st.text(log)

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------

for role, message in st.session_state.chat_history:

    with st.chat_message(role):
        st.markdown(message)

# -----------------------------
# USER INPUT
# -----------------------------

prompt = st.chat_input("Ask a question...")

if prompt:

    # display user message
    st.session_state.chat_history.append(("user", prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    # run agent
    with st.spinner("Agent thinking..."):

        result = app.invoke({
            "messages": [HumanMessage(content=prompt)]
        })

    # extract final AI response
    ai_message = None

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            ai_message = msg.content

    if ai_message is None:
        ai_message = "I couldn't generate a response."

    # display assistant response
    with st.chat_message("assistant"):
        st.markdown(ai_message)

    # save to history
    st.session_state.chat_history.append(("assistant", ai_message))