# loaders/web_loader.py

import os
from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader

os.environ["USER_AGENT"] = "Mozilla/5.0 (X11; Linux x86_64)"

URLS = [
    "https://www.ibm.com/think/topics/agentic-ai-vs-generative-ai",
    "https://www.cognigy.com/agentic-ai/generative-ai-vs-agentic-ai",
    "https://business.adobe.com/ai/agentic-ai-vs-generative-ai.html",
    "https://www.geeksforgeeks.org/artificial-intelligence/gen-ai-vs-ai-agents-vs-agentic-ai/"
]

def load_web_documents():
    documents = []

    # Only extract useful HTML sections
    strainer = SoupStrainer(["article", "main", "section", "p"])

    for url in URLS:
        print(f"Loading: {url}")

        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs={"parse_only": strainer}
        )

        docs = loader.load()

        # Clean excessive whitespace
        for doc in docs:
            doc.page_content = " ".join(doc.page_content.split())

        documents.extend(docs)

    return documents