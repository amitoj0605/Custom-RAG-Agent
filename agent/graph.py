from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import MessagesState
from agent.generate_query_or_respond import generate_query_or_respond
from agent.grade_documents import grade_documents
from agent.retriever_tool import retriever_tool


workflow = StateGraph(MessagesState)

# Nodes
# generate_answer is intentionally excluded from the graph —
# the final answer is streamed directly in chat_app.py to avoid
# double generation (which was adding ~10s of wasted LLM time).
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("grade_documents", grade_documents)

# Start
workflow.add_edge(START, "generate_query_or_respond")

# Decide whether retrieval is needed
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# Retrieval pipeline — ends after grading, chat_app.py takes over from here
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", END)

# Compile
graph = workflow.compile()