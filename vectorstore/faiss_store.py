# vectorstore/faiss_store.py

import faiss
import pickle
from typing import List
from langchain_core.documents import Document
import numpy as np

class FaissRetriever:
    """
    Load FAISS index and perform semantic search for a query embedding.
    """

    def __init__(self, embedding_service, index_path: str = "vectorstore/faiss.index", mapping_path: str = "vectorstore/doc_mapping.pkl"):
        self.embedding_service = embedding_service

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load doc mapping
        with open(mapping_path, "rb") as f:
            self.doc_mapping: List[Document] = pickle.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve top-k most relevant document chunks for a query.
        """
        # Embed the query
        query_embedding = self.embedding_service.get_query_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')

        # Search FAISS
        distances, indices = self.index.search(query_vector, top_k)

        # Map indices to Document objects
        results = [self.doc_mapping[idx] for idx in indices[0] if idx < len(self.doc_mapping)]
        return results