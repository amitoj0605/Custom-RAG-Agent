# loaders/web_loader.py
import os

os.environ["USER_AGENT"] = "Mozilla/5.0 (X11; Linux x86_64)"

from langchain_community.document_loaders import WebBaseLoader

URLS = [
    "https://www.ibm.com/think/topics/agentic-ai-vs-generative-ai",
    "https://www.cognigy.com/agentic-ai/generative-ai-vs-agentic-ai",
    "https://business.adobe.com/ai/agentic-ai-vs-generative-ai.html",
    "https://www.geeksforgeeks.org/artificial-intelligence/gen-ai-vs-ai-agents-vs-agentic-ai/"
]

def load_web_documents():
    documents = []

    for url in URLS:
        print(f"Loading: {url}")
        loader = WebBaseLoader(url)
        docs = loader.load()  # returns a list of Document objects
        documents.extend(docs)

    return documents

