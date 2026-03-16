import time
from utils.logger import log
from langchain_ollama import ChatOllama

response_model = ChatOllama(
    model="qwen2.5",
    temperature=0
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If the answer is not contained in the context, say you don't know. "
    "Keep the answer concise (max three sentences).\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}"
)


def generate_answer(state):

    log("Node: generate_answer started")

    start = time.time()

    messages = state["messages"]

    # first message = user question
    question = messages[0].content

    # last message = retriever output
    context = messages[-1].content

    # limit context size for faster generation
    context = context[:800]

    log(f"Context length: {len(context)} characters")

    prompt = GENERATE_PROMPT.format(
        question=question,
        context=context
    )

    response = response_model.invoke(prompt)

    llm_time = time.time() - start
    log(f"LLM response time: {llm_time:.2f}s")

    log("Final answer generated")

    return {"messages": [response]}