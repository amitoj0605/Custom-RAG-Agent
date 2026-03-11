from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

response_model = ChatOllama(model="qwen2.5", temperature=0)

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
    question = messages[0].content

    prompt = REWRITE_PROMPT.format(question=question)

    response = response_model.invoke(
        [{"role": "user", "content": prompt}]
    )

    return {"messages": [HumanMessage(content=response.content)]}