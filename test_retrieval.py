from ingest.ingestion import load_all_documents
from split.splitter import split_documents
from embeddings.ollama_embeddings import EmbeddingService
from vectorstore.index_builder import FaissIndexBuilder
from vectorstore.faiss_store import FaissRetriever

# 1️⃣ Load and split docs
all_docs = load_all_documents()
chunked_docs = split_documents(all_docs)

# 2️⃣ Create embedding service
embedding_service = EmbeddingService()

# 3️⃣ Build and save FAISS index
index_builder = FaissIndexBuilder(embedding_service)
index_builder.build_index(chunked_docs)
index_builder.save_index()

# 4️⃣ Load retriever and query
retriever = FaissRetriever(embedding_service)
results = retriever.retrieve("What is agentic AI?", top_k=3)

for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content[:200]}...")  # print first 200 chars