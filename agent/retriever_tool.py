import time

from langchain.tools import tool

from vectorstore.faiss_store import FaissRetriever
from embeddings.ollama_embeddings import EmbeddingService

from utils.logger import log


embedding_service = EmbeddingService()
retriever = FaissRetriever(embedding_service)


@tool
def retriever_tool(query: str):
    """
    Search the knowledge base and return relevant document chunks.
    """

    log("Retriever tool invoked")

    start = time.time()

    docs = retriever.retrieve(query, top_k=2)

    retrieval_time = time.time() - start

    log(f"Retrieved {len(docs)} documents")
    log(f"Retrieval latency: {retrieval_time:.2f}s")

    cleaned_chunks = []

    for i, doc in enumerate(docs):

        text = " ".join(doc.page_content.split())[:400]

        log(f"Chunk {i+1} retrieved")

        cleaned_chunks.append(text)

    return {
        "chunks": cleaned_chunks,
        "text": "\n\n".join(cleaned_chunks)
    }