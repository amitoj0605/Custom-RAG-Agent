from utils.logger import log

def grade_documents(state):

    log("Grading retrieved documents (fast heuristic)")

    messages = state["messages"]

    # question is always first message
    question = messages[0].content.lower()

    # retriever output is last message
    context = messages[-1].content.lower()

    score = sum(word in context for word in question.split())

    if score >= 2:
        log("Documents considered relevant")
    else:
        log("Documents weakly relevant but passing")

    # return state unchanged
    return {"messages": messages}