from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

from agent.retriever_tool import retriever_tool


# Initialize LLM
response_model = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)


def generate_query_or_respond(state: MessagesState):
    """
    Decide whether to answer directly or call the retriever tool.
    """

    system_prompt = SystemMessage(
        content=(
            "You are an AI assistant with access to a document retrieval tool.\n"
            "If the user asks about information that may exist in documents, "
            "use the retriever_tool to search for relevant information.\n"
            "If the question is general or conversational, respond directly."
        )
    )

    # Add system instruction
    messages = [system_prompt] + state["messages"]

    # Bind tool
    model_with_tools = response_model.bind_tools([retriever_tool])

    # Invoke model
    response = model_with_tools.invoke(messages)

    return {"messages": [response]}