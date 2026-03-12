import streamlit as st
from agent.graph import graph

st.set_page_config(page_title="Agentic RAG Chat", page_icon="🤖")

st.title("🤖 Agentic RAG Assistant")

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# user input
if prompt := st.chat_input("Ask something..."):

    # show user message
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # run the graph
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    response = result["messages"][-1].content

    # show assistant message
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})