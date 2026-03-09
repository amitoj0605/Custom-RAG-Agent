from agent.retriever_tool import retriever_tool

query = "What is agentic AI?"

result = retriever_tool.invoke({"query": query})

print("Top relevant chunks:\n")
print(result)