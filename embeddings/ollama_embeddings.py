# embeddings/ollama_embeddings.py

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document  # correct import for your version
from typing import List

class EmbeddingService:
    """
    Embedding service using Ollama's nomic-embed-text model via the langchain_community module.
    """

    def __init__(self, model_name: str = "nomic-embed-text"):
        # Initialize the Ollama embedding model
        self.embeddings_model = OllamaEmbeddings(model=model_name)

    def get_embeddings(self, docs: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of Document objects.
        """
        texts = [doc.page_content for doc in docs]
        return self.embeddings_model.embed_documents(texts)

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate an embedding vector for a single query string.
        """
        return self.embeddings_model.embed_query(query)