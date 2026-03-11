from agent.generate_query_or_respond import generate_query_or_respond

state = {
    "messages": [
        {"role": "user", "content": "hello"}
    ]
}

result = generate_query_or_respond(state)

message = result["messages"][0]

print("\nMessage Content:")
print(message.content)

print("\nTool Calls:")
print(message.tool_calls)