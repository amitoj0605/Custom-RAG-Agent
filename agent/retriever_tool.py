# agent/retriever_tool.py

from langchain.tools import tool
from vectorstore.faiss_store import FaissRetriever
from embeddings.ollama_embeddings import EmbeddingService

# Initialize embedding service
embedding_service = EmbeddingService()

# Load FAISS retriever (index already built and saved)
retriever = FaissRetriever(embedding_service)

# Define retriever tool
@tool
def retriever_tool(query: str) -> str:
    """Search the knowledge base and return relevant information."""

    docs = retriever.retrieve(query, top_k=5)

    cleaned_chunks = []

    for doc in docs:
        text = doc.page_content

        # remove excessive whitespace and newlines
        text = " ".join(text.split())

        cleaned_chunks.append(text)

    return "\n\n".join(cleaned_chunks)
# Export tool for agent usage
retriever_tool = retriever_tool