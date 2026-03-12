from agent.graph import graph


for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Summarize the agentic AI articles in the dataset"
            }
        ]
    }
):

    for node, update in chunk.items():

        print("Update from node:", node)

        if "messages" in update:
            update["messages"][-1].pretty_print()

        print("\n\n")