from agent.graph import graph


for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is Agentic AI?"
            }
        ]
    }
):

    for node, update in chunk.items():

        print("Update from node:", node)

        if "messages" in update:
            update["messages"][-1].pretty_print()

        print("\n\n")