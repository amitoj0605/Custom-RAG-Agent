# split/splitter.py

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split loaded documents into smaller chunks for vectorstore indexing.

    Args:
        documents (List[Document]): List of Document objects.

    Returns:
        List[Document]: List of chunked Document objects.
    """

    # Token-aware recursive splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,     # tokens per chunk
        chunk_overlap=100   # tokens overlapping between chunks
    )

    split_docs = text_splitter.split_documents(documents)

    print(f"Documents split into {len(split_docs)} chunks.")

    return split_docs