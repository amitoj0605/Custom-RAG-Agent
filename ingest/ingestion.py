# ingestion/ingest.py

from loaders.web_loaders import load_web_documents
from loaders.local_loaders import load_local_documents

def load_all_documents():
    print("Loading web documents...")
    web_docs = load_web_documents()

    print("\nLoading local documents...")
    local_docs = load_local_documents()

    all_docs = web_docs + local_docs

    print(f"\nTotal documents loaded: {len(all_docs)}")

    return all_docs