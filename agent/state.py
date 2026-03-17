from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Grader sets this after evaluating retrieved docs.
    # "relevant"  → proceed to answer generation
    # "not_relevant" → rewrite question and retry retrieval
    doc_grade: Optional[str]
    # Tracks how many rewrite attempts have been made.
    # Prevents infinite rewrite loops — max 1 retry.
    rewrite_count: int