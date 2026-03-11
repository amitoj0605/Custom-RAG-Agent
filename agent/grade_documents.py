# agent/grade_documents.py

from typing import Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama


GRADE_PROMPT = """
You are a grader assessing relevance of a retrieved document to a user question.

Retrieved Document:
{context}

User Question:
{question}

If the document contains keyword(s) or semantic meaning related to the question,
grade it as relevant.

Give a binary score 'yes' or 'no'.
"""


class GradeDocuments(BaseModel):
    """Binary relevance score for retrieved documents."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, 'no' if not relevant"
    )


# grading model
grader_model = ChatOllama(
    model="qwen2.5",
    temperature=0
)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """
    Determine whether retrieved documents are relevant.
    """

    print("---GRADING DOCUMENTS---")

    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(
        question=question,
        context=context
    )

    response = grader_model.with_structured_output(
        GradeDocuments
    ).invoke(
        [{"role": "user", "content": prompt}]
    )

    score = response.binary_score

    if score == "yes":
        print("---DOCUMENTS RELEVANT---")
        return "generate_answer"

    else:
        print("---DOCUMENTS NOT RELEVANT---")
        return "rewrite_question"