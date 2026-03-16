from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

from agent.retriever_tool import retriever_tool


# 3b is used ONLY for routing — it's much better at tool calling than 1.5b.
# Answer generation still uses 1.5b so overall speed stays fast.
# num_predict=100 is enough — routing just needs to emit a tool call or short reply.
response_model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    num_predict=100,
)


def generate_query_or_respond(state: MessagesState):
    """
    Decide whether to answer directly or call the retriever tool.
    """

    system_prompt = SystemMessage(
        content=(
            "You are an AI assistant with a retriever_tool to search a knowledge base.\n"
            "ALWAYS call retriever_tool for ANY question about AI, agents, RAG, "
            "agentic AI, generative AI, or any factual topic.\n"
            "Only respond directly for greetings or purely conversational messages."
        )
    )

    messages = [system_prompt] + state["messages"]

    model_with_tools = response_model.bind_tools([retriever_tool])

    response = model_with_tools.invoke(messages)

    return {"messages": [response]}