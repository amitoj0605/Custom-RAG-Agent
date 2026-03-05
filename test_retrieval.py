from ingest.ingestion import load_all_documents

docs = load_all_documents()

print("\nPreview first document:\n")
print(docs[0].page_content[:400])
