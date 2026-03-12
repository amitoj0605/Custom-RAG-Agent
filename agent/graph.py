from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import MessagesState
from agent.generate_query_or_respond import generate_query_or_respond
from agent.rewrite_question import rewrite_question
from agent.generate_answer import generate_answer
from agent.grade_documents import grade_documents
from agent.retriever_tool import retriever_tool


# Create Graph
workflow = StateGraph(MessagesState)

# Nodes
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

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


# After retrieval → check document relevance
workflow.add_conditional_edges(
    "retrieve",
    grade_documents
)


# If documents are relevant → generate answer
workflow.add_edge("generate_answer", END)


# If documents are irrelevant → rewrite question
workflow.add_edge("rewrite_question", "generate_query_or_respond")


# Compile graph
graph = workflow.compile()