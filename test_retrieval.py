# test_generate_answer.py

from agent.generate_answer import generate_answer
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


input_state = {
    "messages": [

        HumanMessage(
            content="What is Agentic AI?"
        ),

        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "1",
                    "name": "retrieve_documents",
                    "args": {"query": "agentic ai definition"},
                }
            ],
        ),

        ToolMessage(
            content="Agentic AI refers to artificial intelligence systems that can autonomously plan, reason, and take actions to achieve goals. These systems combine large language models with tools, memory, and decision-making capabilities.",
            tool_call_id="1",
        ),
    ]
}


if __name__ == "__main__":

    result = generate_answer(input_state)

    print("\nGenerated Answer:\n")

    result["messages"][-1].pretty_print()