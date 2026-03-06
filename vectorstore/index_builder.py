# vectorstore/index_builder.py

import faiss
import pickle
from embeddings.ollama_embeddings import EmbeddingService
from langchain_core.documents import Document
from typing import List

class FaissIndexBuilder:
    """
    Build a FAISS index from document embeddings and save it to disk.
    """

    def __init__(self, embedding_service: EmbeddingService, dimension: int = 768):
        """
        :param embedding_service: Your EmbeddingService instance
        :param dimension: Dimensionality of embeddings (nomic-embed-text has 768)
        """
        self.embedding_service = embedding_service
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        self.doc_mapping = []  # to map FAISS vector IDs to Document objects

    def build_index(self, docs: List[Document]):
        """
        Generate embeddings for the docs and add them to FAISS index.
        """
        print(f"Generating embeddings for {len(docs)} chunks...")
        embeddings = self.embedding_service.get_embeddings(docs)

        # Convert to float32 numpy array (FAISS requirement)
        import numpy as np
        embeddings_np = np.array(embeddings).astype('float32')

        # Add embeddings to FAISS
        self.index.add(embeddings_np)
        self.doc_mapping.extend(docs)
        print(f"Added {len(docs)} vectors to FAISS index.")

    def save_index(self, index_path: str = "vectorstore/faiss.index", mapping_path: str = "vectorstore/doc_mapping.pkl"):
        """
        Save FAISS index and document mapping to disk.
        """
        faiss.write_index(self.index, index_path)
        with open(mapping_path, "wb") as f:
            pickle.dump(self.doc_mapping, f)
        print(f"FAISS index saved to {index_path} and mapping to {mapping_path}.")