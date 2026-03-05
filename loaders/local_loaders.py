# loaders/local_loader.py

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 👇 Add your local file paths here
LOCAL_FILES = [

    "data/agentic_ai_notes.txt",
]

def load_local_documents():
    documents = []

    for file_path in LOCAL_FILES:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Loading local file: {file_path}")

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)

        else:
            print(f"Unsupported file type: {file_path}")
            continue

        docs = loader.load()
        documents.extend(docs)

    return documents