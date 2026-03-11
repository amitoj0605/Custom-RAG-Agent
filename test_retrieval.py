from agent.grade_documents import grade_documents
from langchain_core.messages import convert_to_messages

input_state = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What is agentic AI?"
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retriever_tool",
                        "args": {"query": "agentic AI"}
                    }
                ]
            },
            {
                "role": "tool",
                "content": "Agentic AI refers to AI systems that can autonomously plan, reason, and take actions.",
                "tool_call_id": "1"
            }
        ]
    )
}

result = grade_documents(input_state)

print(result)