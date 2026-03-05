# ingest/ingestion.py

from typing import List
from langchain_core.documents import Document

# import your loaders
from loaders.local_loaders import load_local_documents
from loaders.web_loaders import load_web_documents


def load_all_documents() -> List[Document]:
    """
    Load all documents from local files and web URLs.
    Returns a unified list of Document objects.
    """

    # 1️⃣ Load local documents
    local_docs = load_local_documents()

    # 2️⃣ Load web documents
    web_docs = load_web_documents()

    # 3️⃣ Combine all documents
    all_docs = local_docs + web_docs

    print(f"Loaded {len(local_docs)} local docs and {len(web_docs)} web docs.")
    print(f"Total documents: {len(all_docs)}")

    return all_docs