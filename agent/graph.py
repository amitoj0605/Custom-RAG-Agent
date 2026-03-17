from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import MessagesState
from agent.generate_query_or_respond import generate_query_or_respond
from agent.grade_documents import grade_documents
from agent.rewrite_question import rewrite_question
from agent.retriever_tool import retriever_tool


def route_after_grading(state: MessagesState) -> str:
    """
    After grading, decide next step:
    - "relevant"     → END (chat_app.py streams the answer)
    - "not_relevant" → rewrite_question (retry with improved query)
    """
    grade = state.get("doc_grade", "relevant")
    if grade == "not_relevant":
        return "rewrite_question"
    return END


workflow = StateGraph(MessagesState)

# Nodes
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_question", rewrite_question)

# Start
workflow.add_edge(START, "generate_query_or_respond")

# Route: call retriever tool or answer directly
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# After retrieval → grade
workflow.add_edge("retrieve", "grade_documents")

# After grading → answer (END) or rewrite and retry
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "rewrite_question": "rewrite_question",
        END: END,
    },
)

# After rewrite → re-route (will call retriever again with better query)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()