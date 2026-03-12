from langchain_ollama import ChatOllama

response_model = ChatOllama(
    model="qwen2.5",
    temperature=0
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state):
    """Generate final answer using retrieved context."""

    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GENERATE_PROMPT.format(
        question=question,
        context=context
    )

    response = response_model.invoke(
        [{"role": "user", "content": prompt}]
    )

    return {"messages": [response]}