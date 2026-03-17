from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from utils.logger import log

response_model = ChatOllama(model="qwen2.5:1.5b", temperature=0, num_predict=128)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state):
    messages = state["messages"]
    rewrite_count = state.get("rewrite_count", 0)

    # Always rewrite from the original question (first message)
    question = messages[0].content

    log(f"Rewriting question (attempt {rewrite_count + 1}): '{question}'")

    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])

    log(f"Rewritten query: '{response.content}'")

    return {
        "messages": [HumanMessage(content=response.content)],
        "rewrite_count": rewrite_count + 1
    }