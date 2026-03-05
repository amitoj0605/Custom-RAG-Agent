from ingest.ingestion import load_all_documents
from split.splitter import split_documents

# 1️⃣ Load all documents
docs = load_all_documents()

# 2️⃣ Split documents into chunks
chunked_docs = split_documents(docs)

# 3️⃣ Check first chunk
print("First chunk content:\n", chunked_docs[0].page_content.strip())